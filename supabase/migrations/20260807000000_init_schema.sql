-- AutoQuant Supabase Schema Migration
CREATE TABLE IF NOT EXISTS public.profiles (
  id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
  name TEXT NOT NULL,
  avatar TEXT NOT NULL DEFAULT 'U',
  role TEXT NOT NULL DEFAULT 'Dealer',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.evaluations (
  id TEXT PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  source TEXT, brand TEXT, model TEXT, year INT,
  fuel TEXT, transmission TEXT, city TEXT,
  odometer INT, owner_count INT, condition TEXT,
  seller_asking_price NUMERIC, market_value NUMERIC,
  buy_price NUMERIC, sell_price NUMERIC,
  expected_profit NUMERIC, margin_pct NUMERIC,
  risk_score NUMERIC, confidence_score NUMERIC,
  action TEXT, is_ml_powered BOOLEAN DEFAULT TRUE,
  positive_factors JSONB, negative_factors JSONB
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.evaluations ENABLE ROW LEVEL SECURITY;