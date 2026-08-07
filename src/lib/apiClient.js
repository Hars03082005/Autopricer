/**
 * Thin fetch wrapper for the PriceRef API.
 *
 * Every authenticated call attaches the current Supabase access token. The token
 * is read from the Supabase client at call time rather than held in a variable,
 * because supabase-js refreshes it in the background — a captured copy goes
 * stale and starts producing 401s an hour into a session.
 */

import { supabase, isSupabaseConfigured } from './supabaseClient.js';
import { getApiBase } from './runtimeConfig.js';

/** Error carrying the HTTP status, so callers can distinguish 401 from 503. */
export class ApiError extends Error {
  constructor(message, { status, detail } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status ?? 0;
    this.detail = detail;
  }

  /** True when the API is reachable but persistence is switched off server-side. */
  get isUnavailable() {
    return this.status === 503;
  }

  get isAuthError() {
    return this.status === 401 || this.status === 403;
  }
}

async function currentAccessToken() {
  if (!isSupabaseConfigured) return null;
  const { data, error } = await supabase.auth.getSession();
  if (error) return null;
  return data?.session?.access_token ?? null;
}

async function request(path, { method = 'GET', body, auth = true, signal } = {}) {
  const headers = { Accept: 'application/json' };
  if (body !== undefined) headers['Content-Type'] = 'application/json';

  if (auth) {
    const token = await currentAccessToken();
    if (!token) {
      // Fail before the network call: without a token the endpoint returns 401
      // regardless, and this way callers get a stable, inspectable error.
      throw new ApiError('Not signed in', { status: 401 });
    }
    headers.Authorization = `Bearer ${token}`;
  }

  let response;
  try {
    response = await fetch(`${getApiBase()}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (cause) {
    if (cause?.name === 'AbortError') throw cause;
    throw new ApiError('Could not reach the PriceRef API', { status: 0, detail: cause?.message });
  }

  if (response.status === 204) return null;

  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { detail: text.slice(0, 500) };
    }
  }

  if (!response.ok) {
    const detail = payload?.detail;
    // FastAPI validation errors arrive as an array of per-field objects; join
    // them into something displayable instead of rendering "[object Object]".
    const message = Array.isArray(detail)
      ? detail.map(d => d?.msg ?? JSON.stringify(d)).join('; ')
      : (typeof detail === 'string' ? detail : `Request failed (${response.status})`);
    throw new ApiError(message, { status: response.status, detail });
  }

  return payload;
}

// ── Valuation history ───────────────────────────────────────────────────────

export async function fetchHistory({ limit = 200, signal } = {}) {
  try {
    const payload = await request(`/api/history?limit=${encodeURIComponent(limit)}`, { signal });
    return payload?.evaluations ?? [];
  } catch (err) {
    if (err instanceof ApiError && err.isUnavailable) return [];
    throw err;
  }
}

export async function createHistoryEntry(record, { signal } = {}) {
  try {
    return await request('/api/history', { method: 'POST', body: record, signal });
  } catch (err) {
    if (err instanceof ApiError && err.isUnavailable) return null;
    throw err;
  }
}

export async function clearHistory({ signal } = {}) {
  try {
    const payload = await request('/api/history', { method: 'DELETE', signal });
    return payload?.deleted ?? 0;
  } catch (err) {
    if (err instanceof ApiError && err.isUnavailable) return 0;
    throw err;
  }
}

export async function deleteHistoryEntry(id, { signal } = {}) {
  try {
    const payload = await request(`/api/history/${encodeURIComponent(id)}`, {
      method: 'DELETE',
      signal,
    });
    return payload?.deleted ?? 0;
  } catch (err) {
    if (err instanceof ApiError && err.isUnavailable) return 0;
    throw err;
  }
}

// ── Dealer profile ──────────────────────────────────────────────────────────

export async function fetchProfile({ signal } = {}) {
  try {
    return await request('/api/profile', { signal });
  } catch (err) {
    if (err instanceof ApiError && err.isUnavailable) return null;
    throw err;
  }
}

export async function saveProfile(profile, { signal } = {}) {
  try {
    return await request('/api/profile', { method: 'PUT', body: profile, signal });
  } catch (err) {
    if (err instanceof ApiError && err.isUnavailable) return null;
    throw err;
  }
}
