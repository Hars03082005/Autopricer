/**
 * Single source of truth for frontend configuration.
 *
 * Resolution order, highest priority first:
 *   1. window.PriceRef_API_URL     — injected by the Flutter WebView shell after
 *                                    onPageFinished (api url only).
 *   2. window.__PRICEREF_CONFIG__  — written by the container entrypoint at boot
 *                                    from environment variables.
 *   3. import.meta.env.VITE_*      — inlined by Vite at build time; this is what
 *                                    `npm run dev` and .env use.
 *   4. Same-origin (deployed) or http://localhost:8000 (local).
 *
 * Layer 2 is the reason this module exists. Vite substitutes import.meta.env at
 * build time, so a built bundle otherwise has its API and Supabase URLs frozen
 * in — forcing a separate build per environment and making "promote the exact
 * artifact that passed staging" impossible.
 */

function runtime() {
  if (typeof window === 'undefined') return {};
  return window.__PRICEREF_CONFIG__ || {};
}

/** Returns the first non-blank, trimmed string among the arguments, else ''. */
function firstNonBlank(...values) {
  for (const value of values) {
    if (typeof value === 'string' && value.trim() !== '') return value.trim();
  }
  return '';
}

function stripTrailingSlashes(url) {
  return url.replace(/\/+$/, '');
}

function servedFromLocalhost() {
  if (typeof window === 'undefined') return false;
  const host = window.location.hostname;
  return host === 'localhost' || host === '127.0.0.1';
}

/**
 * Base URL of the FastAPI backend, or '' meaning same-origin.
 *
 * Deliberately a function rather than a constant: the Flutter shell injects
 * window.PriceRef_API_URL *after* the module graph has initialised, so a value
 * captured at import time would always miss it.
 *
 * The deployed default is same-origin (''), because the frontend container's
 * nginx reverse-proxies the API paths to the backend (see docker/nginx.conf).
 * That keeps browser traffic single-origin, so production needs no CORS at all;
 * CORS_ALLOWED_ORIGINS on the backend exists for the mobile shell and for local
 * `npm run dev`, which do call it cross-origin.
 */
export function getApiBase() {
  if (typeof window !== 'undefined' && window.PriceRef_API_URL) {
    return stripTrailingSlashes(String(window.PriceRef_API_URL));
  }

  const configured = firstNonBlank(
    runtime().apiUrl,
    import.meta.env.VITE_API_URL,
    import.meta.env.VITE_ML_API_URL,
  );

  if (configured) {
    const pointsAtLocalhost = /localhost|127\.0\.0\.1/.test(configured);
    // A localhost API URL is only usable when the page itself is on localhost.
    // Baked into a deployed bundle it sends every visitor's browser to its own
    // machine, which surfaces as an opaque network error.
    if (!pointsAtLocalhost || servedFromLocalhost()) {
      return stripTrailingSlashes(configured);
    }
  }

  // Same-origin when deployed. This replaces the previous hardcoded
  // *.onrender.com host-sniffing, which pointed any unrecognised hostname at a
  // fixed third-party deployment.
  return servedFromLocalhost() ? 'http://localhost:8000' : '';
}

export const supabaseUrl = firstNonBlank(
  runtime().supabaseUrl,
  import.meta.env.VITE_SUPABASE_URL,
);

export const supabaseAnonKey = firstNonBlank(
  runtime().supabaseAnonKey,
  import.meta.env.VITE_SUPABASE_ANON_KEY,
);

/** Deployment environment name, surfaced in logs and error reports. */
export const environment = firstNonBlank(runtime().environment, 'development');

/** Container image tag / git SHA of this build, for correlating bug reports. */
export const release = firstNonBlank(runtime().release, 'local');
