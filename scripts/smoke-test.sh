#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Post-deploy smoke test.
#
#   ./scripts/smoke-test.sh https://priceref-staging-frontend.<region>.azurecontainerapps.io
#
# Run against the *frontend* URL, which is the only public endpoint — the backend
# has internal-only ingress. That means this also verifies the nginx reverse
# proxy, not just the API: a deploy where the API is healthy but unreachable
# through the proxy is still a failed deploy.
#
# Shared by the staging and production jobs in .github/workflows/cd.yml so both
# environments are validated identically. Kept as a script rather than inline YAML
# so it can be run by hand against a deployment that is misbehaving.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BASE_URL="${1:-}"
if [ -z "$BASE_URL" ]; then
  echo "usage: $0 <base-url>" >&2
  exit 2
fi
BASE_URL="${BASE_URL%/}"

# A cold backend replica loads ~250 MB of model artifacts before it answers.
READINESS_TIMEOUT="${READINESS_TIMEOUT:-300}"
POLL_INTERVAL=5

pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1" >&2; exit 1; }

# ── Locate a working Python 3 ────────────────────────────────────────────────
# Used further down to parse the prediction JSON and range-check the valuation.
#
# `python3` is right on Linux and macOS, and is what CI uses, but hardcoding it
# made this script unrunnable on Windows: Git Bash finds the Microsoft Store
# "App Execution Alias" for both python3 and python, which prints an advert for
# the Store and exits non-zero. So a deploy could only be smoke-tested from CI,
# defeating the point of keeping this as a hand-runnable script.
#
# Each candidate is probed by actually executing it rather than by `command -v`,
# because the Store stub is a real file on PATH and satisfies that check while
# failing the moment it runs.
PYTHON=""
for candidate in python3 python py; do
  command -v "$candidate" >/dev/null 2>&1 || continue
  if [ "$candidate" = "py" ]; then
    # The Windows launcher needs -3 to guarantee a Python 3 interpreter.
    if py -3 -c 'import sys; sys.exit(0)' >/dev/null 2>&1; then
      PYTHON="py -3"
      break
    fi
  elif "$candidate" -c 'import sys; sys.exit(0 if sys.version_info[0] == 3 else 1)' >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done
[ -n "$PYTHON" ] \
  || fail "no working Python 3 found (tried python3, python, py -3); needed to check the prediction"

echo "==> Smoke testing ${BASE_URL}"

# ── 1. Frontend is serving ───────────────────────────────────────────────────
echo "--> frontend"
status=$(curl -fsS -o /dev/null -w '%{http_code}' "${BASE_URL}/healthz") \
  || fail "frontend /healthz unreachable"
[ "$status" = "200" ] || fail "frontend /healthz returned ${status}"
pass "frontend healthy"

curl -fsS "${BASE_URL}/" | grep -q '<div id="root">' \
  || fail "index.html did not contain the app root element"
pass "SPA shell served"

# ── 2. Runtime config was injected ───────────────────────────────────────────
# The single most likely deployment misconfiguration: if config.js still holds the
# repository's empty placeholder, the browser silently falls back to build-time
# values and sign-in breaks with no server-side error.
echo "--> runtime config"
config=$(curl -fsS "${BASE_URL}/config.js") || fail "config.js unreachable"
echo "$config" | grep -q '__PRICEREF_CONFIG__' || fail "config.js is not the expected shape"
echo "$config" | grep -q 'do not edit' \
  || fail "config.js is still the public/ placeholder — the entrypoint did not run"
pass "runtime config generated at boot"

echo "$config" | grep -qi 'service_role' \
  && fail "config.js appears to contain a service-role key — it must never be client-side"
pass "no service-role key in client config"

cache=$(curl -fsS -o /dev/null -D - "${BASE_URL}/config.js" | tr -d '\r' | grep -i '^cache-control:' || true)
echo "$cache" | grep -qi 'no-store' \
  || fail "config.js is cacheable (${cache:-no header}) — browsers would pin a stale environment"
pass "config.js is not cacheable"

# ── 3. Backend is reachable through the proxy ────────────────────────────────
# The readiness loop waits for two things:
#   a) /health returns 200 (backend reachable through nginx proxy)
#   b) active_variant=final (new revision has fully replaced the old one)
# Breaking on (a) alone is not enough: when Azure Container Apps replaces a
# revision, the old replica stays healthy and in rotation until the new one
# is ready. Without checking (b) the smoke test reports success against the
# old variant_1 image while the new revision is still loading.
echo "--> backend via proxy (up to ${READINESS_TIMEOUT}s for model load)"
deadline=$(( $(date +%s) + READINESS_TIMEOUT ))
health=""
reachable=false
while [ "$(date +%s)" -lt "$deadline" ]; do
  if health=$(curl -fsS "${BASE_URL}/health" 2>/dev/null); then
    reachable=true
    # Keep polling until the new revision (active_variant=final) is live.
    # The old revision may still be healthy and answering during rollover.
    if echo "$health" | grep -q '"active_variant":"final"'; then
      break
    fi
    echo "  (waiting for final revision — currently: $(echo "$health" | grep -o '"active_variant":"[^"]*"' || echo 'unknown'))"
  fi
  sleep "$POLL_INTERVAL"
done
[ "$reachable" = "true" ] || fail "backend never answered /health through the proxy"
pass "backend reachable through the nginx proxy"

echo "$health" | grep -q '"status":"ok"' || fail "unexpected health payload: ${health}"
# A process that is listening but failed to load its ensemble is not healthy;
# treating it as healthy would put it in rotation serving errors.
echo "$health" | grep -q '"model_loaded":true' || fail "model not loaded: ${health}"
echo "$health" | grep -q '"active_variant":"final"' \
  || fail "serving an unexpected model variant: ${health}"
pass "model loaded, serving final"

# ── 4. A real prediction ─────────────────────────────────────────────────────
# The endpoint returning 200 is not sufficient: a broken feature pipeline still
# returns 200 with a nonsense number.
echo "--> prediction"
prediction=$(curl -fsS -X POST "${BASE_URL}/predict" \
  -H 'Content-Type: application/json' \
  -d '{"brand":"Honda","model":"City","variant":"VX","year":2021,
       "fuel_type":"Petrol","transmission":"Manual","odometer_reading":28000,
       "owner_count":1,"condition":"Good","city":"Bangalore","locality":"Indiranagar"}') \
  || fail "/predict request failed"

price=$(printf '%s' "$prediction" | $PYTHON -c '
import json, sys
payload = json.load(sys.stdin)
value = payload.get("market_value") or payload.get("predicted_price")
print(value if value is not None else "")
')
[ -n "$price" ] || fail "no price field in prediction response: ${prediction:0:300}"

# A 2021 Honda City is a mainstream mid-tier car. Outside this band means the
# model or the feature pipeline is broken, not merely differently tuned.
$PYTHON -c "
price = float('${price}')
assert 200_000 < price < 3_000_000, f'implausible valuation: {price:,.0f}'
print(f'    valuation: Rs.{price:,.0f}')
" || fail "prediction outside the plausible band: ${price}"
pass "prediction plausible"

# ── 5. Authenticated endpoints are closed ────────────────────────────────────
echo "--> authorisation"
for endpoint in /api/history /api/profile; do
  code=$(curl -fsS -o /dev/null -w '%{http_code}' "${BASE_URL}${endpoint}" || true)
  case "$code" in
    401|403|503) pass "${endpoint} refuses anonymous callers (${code})" ;;
    *) fail "${endpoint} returned ${code} without a token" ;;
  esac
done

code=$(curl -fsS -o /dev/null -w '%{http_code}' -X POST \
  "${BASE_URL}/api/registry/variant_1/activate" || true)
case "$code" in
  401|403) pass "variant activation refused (${code})" ;;
  *) fail "variant activation returned ${code}; it must never be publicly callable" ;;
esac

# ── 6. Security headers ──────────────────────────────────────────────────────
echo "--> headers"
# All three are asserted, not just one. nginx drops every inherited add_header
# from a location that declares any of its own, so these go missing as a set --
# checking one of them and reporting "security headers present" understated a
# failure that had removed all three from every response.
headers=$(curl -fsS -o /dev/null -D - "${BASE_URL}/" | tr -d '\r')
echo "$headers" | grep -qi '^x-content-type-options: nosniff' \
  || fail "missing X-Content-Type-Options"
echo "$headers" | grep -qi '^x-frame-options:' \
  || fail "missing X-Frame-Options"
echo "$headers" | grep -qi '^referrer-policy:' \
  || fail "missing Referrer-Policy"
pass "security headers present"

echo
echo "==> All smoke tests passed against ${BASE_URL}"
