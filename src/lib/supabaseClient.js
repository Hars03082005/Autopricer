import { createClient } from '@supabase/supabase-js';
import { supabaseUrl, supabaseAnonKey } from './runtimeConfig.js';

/**
 * Supabase client — authentication only.
 *
 * Table access has moved to the FastAPI backend (see src/lib/apiClient.js).
 * This client is now used solely for sign-in, sign-up, session refresh and
 * reading the current access token, which the API client attaches to its
 * requests. Auth stays here because proxying password and session flows through
 * our own service would add risk without adding anything.
 *
 * URL and key come from runtimeConfig so they can be injected at container start
 * rather than baked in at build time. The anon key is public by design: it
 * identifies the project, and row-level security is what protects the data.
 */

export const isSupabaseConfigured = Boolean(
  supabaseUrl &&
  supabaseAnonKey &&
  // Guard against the placeholder values that ship in .env.example being
  // mistaken for real configuration.
  !supabaseUrl.includes('placeholder') &&
  !supabaseUrl.includes('your-project') &&
  !supabaseUrl.includes('your_supabase_url') &&
  !supabaseAnonKey.includes('placeholder')
);

if (!isSupabaseConfigured) {
  console.warn(
    '[PriceRef] Supabase is not configured — running in guest mode. ' +
    'Valuations work normally; history is kept in this browser only.'
  );
}

/**
 * Null when unconfigured, rather than a client pointed at a placeholder project.
 *
 * The previous version constructed a client against a fake URL, so guest-mode
 * calls produced confusing network failures against a domain that does not
 * exist. Callers check `isSupabaseConfigured` and skip the call instead.
 */
export const supabase = isSupabaseConfigured
  ? createClient(supabaseUrl, supabaseAnonKey, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: true,
      },
    })
  : null;
