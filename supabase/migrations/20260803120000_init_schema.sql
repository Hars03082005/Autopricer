-- ─────────────────────────────────────────────────────────────────────────────
-- PriceRef initial schema: dealer profiles and valuation history.
--
-- Replaces the copy-paste SQL block that used to live in the README. That block
-- was not just undocumented-as-code, it was wrong in two ways that made cloud
-- history sync fail silently on every write:
--
--   1. `evaluations` had no `variant` or `locality` column, but the frontend's
--      recordToDbRow() sent both — PostgREST rejects unknown columns, so every
--      insert failed and was swallowed by a console.warn.
--   2. `user_id uuid` received the literal string 'guest' for unauthenticated
--      sessions, which is not a valid uuid.
--
-- Guest sessions no longer touch the database at all; they stay in localStorage,
-- which is the honest representation of "not signed in".
--
-- Apply with:  supabase db push      (CI does this — see .github/workflows/cd.yml)
-- ─────────────────────────────────────────────────────────────────────────────

-- gen_random_uuid() lives in pgcrypto on older projects; on current Supabase it
-- is built in. Requesting the extension is idempotent and covers both.
create extension if not exists pgcrypto;

-- ── profiles ────────────────────────────────────────────────────────────────
create table if not exists public.profiles (
  id         uuid        primary key references auth.users (id) on delete cascade,
  name       text        not null check (length(trim(name)) between 1 and 120),
  avatar     text        not null default 'U' check (length(avatar) between 1 and 8),
  role       text        not null default 'Dealer' check (role in ('Dealer', 'Manager', 'Admin', 'Viewer')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

comment on table public.profiles is
  'Dealer profile, one row per auth.users entry. Written only via the API.';

-- ── evaluations ─────────────────────────────────────────────────────────────
-- Column set mirrors backend/routers/history.py EvaluationIn exactly. When one
-- changes, the other must change in the same commit, and
-- tests/test_history_api.py asserts the two agree.
create table if not exists public.evaluations (
  id                  uuid        primary key default gen_random_uuid(),
  user_id             uuid        not null references auth.users (id) on delete cascade,
  created_at          timestamptz not null default now(),

  -- Provenance
  source              text        not null default 'Single Vehicle',
  model_variant       text        not null default '',
  is_ml_powered       boolean     not null default true,

  -- Vehicle identity
  brand               text        not null default 'Unknown',
  model               text        not null default 'Unknown',
  variant             text        not null default '',
  year                integer     not null default 0 check (year between 0 and 2100),
  fuel                text        not null default 'Unknown',
  transmission        text        not null default 'Unknown',
  engine_cc           integer     not null default 0 check (engine_cc between 0 and 20000),

  -- Location
  city                text        not null default 'Unknown',
  locality            text        not null default '',

  -- Condition
  odometer            integer     not null default 0 check (odometer between 0 and 10000000),
  fuel_efficiency     numeric(8,2)  not null default 0 check (fuel_efficiency >= 0),
  owner_count         integer     not null default 1 check (owner_count between 0 and 20),
  condition           text        not null default 'Good',

  -- Valuation output. numeric, never float: these are money, and binary
  -- floating point cannot represent decimal currency exactly.
  seller_asking_price numeric(14,2) not null default 0 check (seller_asking_price >= 0),
  market_value        numeric(14,2) not null default 0 check (market_value >= 0),
  buy_price           numeric(14,2) not null default 0 check (buy_price >= 0),
  sell_price          numeric(14,2) not null default 0 check (sell_price >= 0),
  expected_profit     numeric(14,2) not null default 0,
  margin_pct          numeric(7,3)  not null default 0,

  -- Scores, all on a 0-100 scale
  risk_score          numeric(6,2) not null default 0 check (risk_score between 0 and 100),
  confidence_score    numeric(6,2) not null default 0 check (confidence_score between 0 and 100),
  deal_quality_score  numeric(6,2) not null default 0 check (deal_quality_score between 0 and 100),
  urgency_score       numeric(6,2) not null default 0 check (urgency_score between 0 and 100),

  -- Recommendation
  action              text        not null default 'MANUAL REVIEW',
  positive_factors    jsonb       not null default '[]'::jsonb,
  negative_factors    jsonb       not null default '[]'::jsonb
);

comment on table public.evaluations is
  'One row per completed valuation. Written only via the API, which sets id, '
  'user_id and created_at from the verified access token.';

-- The dashboard's only access pattern is "my rows, newest first", and it is
-- always filtered by user_id (by the API and again by RLS). A composite index in
-- that exact order serves both the filter and the sort from one scan.
create index if not exists evaluations_user_created_idx
  on public.evaluations (user_id, created_at desc);

-- Dashboard filters facet on brand and city within a user's own rows.
create index if not exists evaluations_user_brand_idx
  on public.evaluations (user_id, brand);
create index if not exists evaluations_user_city_idx
  on public.evaluations (user_id, city);

-- ── updated_at maintenance ──────────────────────────────────────────────────
create or replace function public.set_updated_at()
returns trigger
language plpgsql
-- SECURITY INVOKER (the default) plus a pinned search_path: a mutable
-- search_path in a trigger function is a known privilege-escalation vector.
set search_path = ''
as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

drop trigger if exists profiles_set_updated_at on public.profiles;
create trigger profiles_set_updated_at
  before update on public.profiles
  for each row execute function public.set_updated_at();

-- ── Row-level security ──────────────────────────────────────────────────────
-- The API already filters every query by the user id it verified from the JWT,
-- and it talks to PostgREST using the caller's own token rather than the
-- service-role key. RLS is therefore the second, authoritative line of defence:
-- a filtering mistake in application code still cannot cross tenants.
alter table public.profiles    enable row level security;
alter table public.evaluations enable row level security;

-- Force RLS so it also applies to the table owner. Without this, a future
-- migration or maintenance script running as the owner bypasses these policies
-- without anyone noticing.
alter table public.profiles    force row level security;
alter table public.evaluations force row level security;

-- Policies are dropped and recreated so this migration is re-runnable.
drop policy if exists "profiles: read own"   on public.profiles;
drop policy if exists "profiles: insert own" on public.profiles;
drop policy if exists "profiles: update own" on public.profiles;

create policy "profiles: read own"
  on public.profiles for select
  to authenticated
  using (auth.uid() = id);

create policy "profiles: insert own"
  on public.profiles for insert
  to authenticated
  with check (auth.uid() = id);

create policy "profiles: update own"
  on public.profiles for update
  to authenticated
  using (auth.uid() = id)
  with check (auth.uid() = id);

drop policy if exists "evaluations: read own"   on public.evaluations;
drop policy if exists "evaluations: insert own" on public.evaluations;
drop policy if exists "evaluations: delete own" on public.evaluations;

create policy "evaluations: read own"
  on public.evaluations for select
  to authenticated
  using (auth.uid() = user_id);

create policy "evaluations: insert own"
  on public.evaluations for insert
  to authenticated
  with check (auth.uid() = user_id);

create policy "evaluations: delete own"
  on public.evaluations for delete
  to authenticated
  using (auth.uid() = user_id);

-- Note the absence of an UPDATE policy on evaluations. A valuation is a record
-- of what the model said at a point in time; editing one after the fact would
-- make the history untrustworthy as an audit trail. Re-run the valuation instead.

-- ── Grants ──────────────────────────────────────────────────────────────────
-- The anon role gets nothing: every request that reaches these tables carries a
-- signed-in user's token. Guests keep their history in localStorage.
revoke all on public.profiles    from anon;
revoke all on public.evaluations from anon;

grant select, insert, update on public.profiles    to authenticated;
grant select, insert, delete on public.evaluations to authenticated;
