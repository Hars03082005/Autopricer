import { createClient } from '@supabase/supabase-js';

const defaultUrl  = 'https://placeholder-project.supabase.co';
const defaultAnon = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBsYWNlaG9sZGVyIiwicm9sZSI6ImFub24iLCJpYXQiOjE2MDA0MDAwMDAsImV4cCI6MTkwMDA0MDAwMH0.placeholder';

const envUrl  = import.meta.env.VITE_SUPABASE_URL;
const envAnon = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const isSupabaseConfigured = Boolean(
  envUrl && envAnon && envUrl !== 'your_supabase_url' && envUrl !== 'https://your-project.supabase.co'
);

if (!isSupabaseConfigured) {
  console.warn(
    '[PriceRef] Supabase credentials missing in .env. Running in offline Guest mode.'
  );
}

export const supabase = createClient(
  isSupabaseConfigured ? envUrl : defaultUrl,
  isSupabaseConfigured ? envAnon : defaultAnon
);
