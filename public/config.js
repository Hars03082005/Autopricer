// Development placeholder for runtime configuration.
//
// Vite copies public/ verbatim into dist/, and the frontend container's
// entrypoint overwrites this file at boot with values from the environment
// (see docker/frontend-entrypoint.sh).
//
// Every key is left empty on purpose: src/lib/runtimeConfig.js falls back to
// import.meta.env.VITE_* when a key is absent or blank, so local `npm run dev`
// keeps reading .env exactly as it did before.
window.__PRICEREF_CONFIG__ = {
  apiUrl: '',
  supabaseUrl: '',
  supabaseAnonKey: '',
  environment: 'development',
  release: 'local',
};
