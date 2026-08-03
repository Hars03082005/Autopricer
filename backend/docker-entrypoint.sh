#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# Container entrypoint for the PriceRef ML API.
#
# Exists rather than a bare uvicorn CMD for three reasons:
#   1. $PORT must be expanded at runtime (Azure Container Apps, Render and
#      Railway all inject it); exec-form CMD does not expand variables.
#   2. Fail fast and loudly on a misconfigured variant, instead of silently
#      falling back to a different model and serving wrong prices.
#   3. exec so uvicorn becomes PID 1 and receives SIGTERM directly, which is
#      what makes ACA's graceful drain on revision swap actually graceful.
# ─────────────────────────────────────────────────────────────────────────────
set -eu

PORT="${PORT:-8000}"
VARIANT="${ACTIVE_VARIANT_ID:-variant_1}"
VARIANT_DIR="/app/model_registry/${VARIANT}"

echo "==> PriceRef API starting | variant=${VARIANT} port=${PORT}"

# The image ships exactly one variant directory (see .dockerignore). If
# ACTIVE_VARIANT_ID points at a variant that was not baked in, resolve_variant_data()
# in backend/main.py would fall through to its model_artifacts/ backup path —
# which is also excluded from the image — and the failure would surface as an
# opaque error on the first prediction rather than at boot.
if [ ! -d "${VARIANT_DIR}" ]; then
  echo "FATAL: ACTIVE_VARIANT_ID=${VARIANT} but ${VARIANT_DIR} is not in this image." >&2
  echo "       Variants present: $(ls /app/model_registry 2>/dev/null | tr '\n' ' ')" >&2
  echo "       Ship a different variant by editing .dockerignore and backend/Dockerfile." >&2
  exit 1
fi

for required in model_metadata.json ensemble_bundle.pkl vehicle_price_catboost.cbm; do
  if [ ! -f "${VARIANT_DIR}/${required}" ]; then
    echo "FATAL: ${VARIANT_DIR}/${required} missing — model artifacts are incomplete." >&2
    exit 1
  fi
done

# Warn on a half-configured database. The API degrades to "valuations work,
# history does not", which is easy to miss in logs otherwise.
#
# The gate is the *anon* key, not the service-role key: user-scoped queries are
# issued with the caller's own JWT plus the anon key so that row-level security
# still applies (see backend/db.py). No request path needs the service-role key.
if [ -z "${SUPABASE_URL:-}" ] || [ -z "${SUPABASE_ANON_KEY:-}" ]; then
  echo "WARN: SUPABASE_URL / SUPABASE_ANON_KEY unset — /api/history and" >&2
  echo "      /api/profile will return 503. ML valuation endpoints are unaffected." >&2
fi

# The service-role key bypasses RLS entirely and nothing here needs it. If it has
# been supplied anyway, say so — an unused secret in the environment is blast
# radius with no benefit.
if [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ]; then
  echo "WARN: SUPABASE_SERVICE_ROLE_KEY is set but no request path uses it." >&2
  echo "      It bypasses row-level security; prefer leaving it unset." >&2
fi

if [ "${CORS_ALLOWED_ORIGINS:-}" = "*" ]; then
  echo "WARN: CORS_ALLOWED_ORIGINS='*' — acceptable for local dev, not for a" >&2
  echo "      deployed environment. Set an explicit origin allowlist." >&2
fi

# --workers 1 is deliberate: each worker loads its own ~250 MB copy of the
# ensemble. Scale with replicas (ACA minReplicas/maxReplicas), never workers.
# --proxy-headers + --forwarded-allow-ips: ACA terminates TLS at its ingress and
# forwards X-Forwarded-*; without these uvicorn reports http:// scheme and the
# ingress IP for every request.
exec uvicorn backend.main:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers 1 \
  --proxy-headers \
  --forwarded-allow-ips='*' \
  --timeout-keep-alive 65 \
  --no-server-header \
  --log-level "${LOG_LEVEL:-info}"
