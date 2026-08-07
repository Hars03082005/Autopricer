#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# Frontend container entrypoint.
#
# Renders two files from the environment before handing off to nginx:
#
#   /usr/share/nginx/html/config.js   runtime config read by the browser
#   /etc/nginx/conf.d/default.conf    nginx config with the backend origin
#
# This is what makes the image environment-agnostic: Vite inlines
# import.meta.env.VITE_* at build time, so without runtime injection the API and
# Supabase URLs would be frozen into the artifact and staging could not be
# promoted to production byte-for-byte.
# ─────────────────────────────────────────────────────────────────────────────
set -eu

HTML_DIR=/usr/share/nginx/html
TEMPLATE=/etc/nginx/nginx.conf.template
RENDERED=/etc/nginx/conf.d/default.conf

# ── Backend origin ───────────────────────────────────────────────────────────
# Must include the scheme. In Azure Container Apps this is the backend's
# *internal* ingress FQDN, e.g. https://priceref-backend.internal.<env-domain>,
# so the ML API has no public endpoint of its own.
BACKEND_ORIGIN="${BACKEND_ORIGIN:-http://backend:8000}"
BACKEND_ORIGIN="$(printf '%s' "$BACKEND_ORIGIN" | sed 's:/*$::')"

case "$BACKEND_ORIGIN" in
  http://*|https://*) ;;
  *)
    echo "FATAL: BACKEND_ORIGIN must start with http:// or https:// (got '$BACKEND_ORIGIN')" >&2
    exit 1
    ;;
esac

# ── DNS resolver ─────────────────────────────────────────────────────────────
# Read from resolv.conf rather than hardcoded: Docker Compose uses 127.0.0.11,
# while Container Apps hands out a cluster DNS address that differs per
# environment. A wrong resolver makes every proxied request fail to resolve.
DNS_RESOLVER="${DNS_RESOLVER:-}"
if [ -z "$DNS_RESOLVER" ]; then
  DNS_RESOLVER="$(awk '/^nameserver/ {print $2; exit}' /etc/resolv.conf 2>/dev/null || true)"
fi
if [ -z "$DNS_RESOLVER" ]; then
  echo "WARN: no nameserver found in /etc/resolv.conf — falling back to 127.0.0.11" >&2
  DNS_RESOLVER="127.0.0.11"
fi

echo "==> PriceRef frontend | backend=${BACKEND_ORIGIN} resolver=${DNS_RESOLVER}"

# ── Render nginx config ──────────────────────────────────────────────────────
# Only the two __PLACEHOLDER__ tokens are substituted, leaving nginx's own
# $variables untouched — which the base image's blanket envsubst step does not
# guarantee.
sed \
  -e "s|__BACKEND_ORIGIN__|${BACKEND_ORIGIN}|g" \
  -e "s|__DNS_RESOLVER__|${DNS_RESOLVER}|g" \
  "$TEMPLATE" > "$RENDERED"

# ── Render browser runtime config ────────────────────────────────────────────
# APP_API_URL is normally left empty so the browser calls the API same-origin
# through the proxy above. Set it only to bypass the proxy deliberately.
#
# Note SUPABASE_ANON_KEY is a *public* key by design — it is safe in client
# JavaScript and is protected by row-level security. The service-role key must
# never appear here; it lives only in the backend container.
cat > "${HTML_DIR}/config.js" <<EOF
// Generated at container start by docker/frontend-entrypoint.sh — do not edit.
window.__PRICEREF_CONFIG__ = {
  apiUrl: "${APP_API_URL:-}",
  supabaseUrl: "${SUPABASE_URL:-}",
  supabaseAnonKey: "${SUPABASE_ANON_KEY:-}",
  environment: "${APP_ENVIRONMENT:-production}",
  release: "${APP_RELEASE:-unknown}"
};
EOF

if [ -z "${SUPABASE_URL:-}" ] || [ -z "${SUPABASE_ANON_KEY:-}" ]; then
  echo "WARN: SUPABASE_URL / SUPABASE_ANON_KEY unset — sign-in is unavailable and" >&2
  echo "      the app runs in guest mode with localStorage history." >&2
fi

# nginx must be PID 1 to receive SIGTERM directly, which is what makes the
# ingress drain connections gracefully on a revision swap.
exec nginx -g 'daemon off;'
