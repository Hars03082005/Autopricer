/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react';
import { supabase, isSupabaseConfigured } from '../lib/supabaseClient.js';
import { fetchProfile, saveProfile, ApiError } from '../lib/apiClient.js';

const AuthContext = createContext(null);

// Local demo account bypass (no Supabase needed for a demo walkthrough).
// A demo session is intentionally never persisted server-side: it has no
// auth.users row, so it has no user id to own database records.
const DEMO_ACCOUNTS = {
  'dealer@priceref.ai': { password: 'dealer123', name: 'Ramesh Sharma', role: 'Dealer', avatar: 'RS' },
};

// Set at signup, applied on the first authenticated load. Supabase may require
// email confirmation, in which case signUp() returns no session and there is no
// token to write the profile with yet — so the chosen display name has to
// survive until the user actually signs in.
const PENDING_PROFILE_KEY = 'PriceRef_pending_profile_v1';

function initialsOf(name) {
  return (
    name
      .trim()
      .split(/\s+/)
      .map(word => word[0])
      .join('')
      .toUpperCase()
      .slice(0, 2) || 'U'
  );
}

function buildUser(supabaseUser, profile) {
  const fallbackName = supabaseUser.email?.split('@')[0] || 'Dealer';
  const name = profile?.name || fallbackName;
  return {
    id: supabaseUser.id,
    email: supabaseUser.email,
    name,
    role: profile?.role || 'Dealer',
    avatar: profile?.avatar || initialsOf(name),
  };
}

function readPendingProfile() {
  try {
    const raw = localStorage.getItem(PENDING_PROFILE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function AuthProvider({ children }) {
  const [currentUser, setCurrentUser] = useState(null);
  // Only start in the loading state when there is actually a session to restore.
  // Initialising to `true` unconditionally forced the unconfigured path to call
  // setLoading(false) synchronously inside an effect, which triggers an extra
  // render pass before paint (react-hooks/set-state-in-effect).
  const [loading, setLoading] = useState(isSupabaseConfigured);
  // Surfaces "signed in, but your profile could not be loaded" to the UI instead
  // of failing silently the way the previous empty catch blocks did.
  const [profileError, setProfileError] = useState(null);

  // Guards against a resolved promise writing state after unmount, and against
  // an earlier auth event overwriting a later one.
  const aliveRef = useRef(true);

  /**
   * Load the profile for a signed-in Supabase user and publish the merged user.
   *
   * The profile now comes from GET /api/profile rather than a direct
   * `supabase.from('profiles')` read. The endpoint synthesises a default when no
   * row exists, so a missing profile is no longer an error path here.
   */
  const hydrateUser = useCallback(async (supabaseUser) => {
    const pending = readPendingProfile();

    try {
      if (pending?.name) {
        // First authenticated load after signup: persist the chosen name now
        // that there is a token to write it with.
        const saved = await saveProfile({
          name: pending.name,
          avatar: pending.avatar || initialsOf(pending.name),
          role: 'Dealer',
        });
        localStorage.removeItem(PENDING_PROFILE_KEY);
        if (aliveRef.current) {
          setCurrentUser(buildUser(supabaseUser, saved));
          setProfileError(null);
        }
        return;
      }

      const profile = await fetchProfile();
      if (aliveRef.current) {
        setCurrentUser(buildUser(supabaseUser, profile));
        setProfileError(null);
      }
    } catch (error) {
      // Still sign the user in — the app is usable without a profile row, and
      // blocking on it would turn a cosmetic failure into a login outage.
      if (aliveRef.current) {
        setCurrentUser(buildUser(supabaseUser, null));
        setProfileError(
          error instanceof ApiError && error.isUnavailable
            ? 'Cloud profile is unavailable; using local defaults.'
            : error.message
        );
      }
    }
  }, []);

  // On mount: restore the session, then subscribe to auth changes.
  useEffect(() => {
    aliveRef.current = true;

    if (!isSupabaseConfigured) {
      // `loading` already initialised to false in this case — nothing to do.
      return () => {
        aliveRef.current = false;
      };
    }

    supabase.auth
      .getSession()
      .then(async ({ data }) => {
        const user = data?.session?.user;
        if (user && aliveRef.current) await hydrateUser(user);
      })
      .catch(() => {})
      .finally(() => {
        if (aliveRef.current) setLoading(false);
      });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!aliveRef.current) return;
      if (session?.user) {
        // Not awaited: onAuthStateChange must not block, and supabase-js warns
        // about awaiting other Supabase calls inside this callback.
        void hydrateUser(session.user);
      } else {
        setCurrentUser(null);
        setProfileError(null);
      }
    });

    return () => {
      aliveRef.current = false;
      subscription?.unsubscribe?.();
    };
  }, [hydrateUser]);

  const login = useCallback(async ({ email, password }) => {
    const normalised = email.trim().toLowerCase();

    const demo = DEMO_ACCOUNTS[normalised];
    if (demo && demo.password === password) {
      setCurrentUser({
        id: 'demo',
        email: normalised,
        name: demo.name,
        role: demo.role,
        avatar: demo.avatar,
      });
      return { ok: true };
    }

    if (!isSupabaseConfigured) {
      return {
        ok: false,
        error: 'Sign-in is unavailable: this deployment has no Supabase project configured.',
      };
    }

    const { error } = await supabase.auth.signInWithPassword({
      email: normalised,
      password,
    });
    if (error) return { ok: false, error: error.message };
    return { ok: true };
  }, []);

  const signup = useCallback(async ({ name, email, password }) => {
    const normalised = email.trim().toLowerCase();
    const trimmedName = name.trim();

    if (DEMO_ACCOUNTS[normalised]) return { ok: false, error: 'Email already registered' };

    if (!isSupabaseConfigured) {
      return {
        ok: false,
        error: 'Sign-up is unavailable: this deployment has no Supabase project configured.',
      };
    }

    // emailRedirectTo is required, not optional. Without it Supabase sends the
    // confirmation link to the project's Site URL, which is a single global
    // value — so it can point at the deployed app or at a developer's machine,
    // never both. Left unset, it stayed on the factory default and every
    // confirmation email pointed at http://localhost:3000.
    //
    // window.location.origin sends the user back to wherever they actually
    // signed up from, which is correct for the deployed app, for `npm run dev`
    // on :5173, and for the compose stack on :5173 alike. Each origin still has
    // to be listed under Authentication -> URL Configuration -> Redirect URLs,
    // otherwise Supabase ignores this and falls back to the Site URL.
    const { data, error } = await supabase.auth.signUp({
      email: normalised,
      password,
      options: { emailRedirectTo: window.location.origin },
    });
    if (error) return { ok: false, error: error.message };

    // Stash the display name; hydrateUser() writes it once a session exists.
    // The previous version inserted the profile row straight from the browser,
    // which failed whenever email confirmation was enabled because there was no
    // authenticated session at that moment.
    try {
      localStorage.setItem(
        PENDING_PROFILE_KEY,
        JSON.stringify({ name: trimmedName, avatar: initialsOf(trimmedName) })
      );
    } catch {
      // Private-browsing quota failure: the name defaults from the email instead.
    }

    if (data.session?.user) {
      await hydrateUser(data.session.user);
      return { ok: true };
    }

    return { ok: true, needsEmailConfirmation: true };
  }, [hydrateUser]);

  const logout = useCallback(async () => {
    if (currentUser?.id === 'demo' || !isSupabaseConfigured) {
      setCurrentUser(null);
      return;
    }
    await supabase.auth.signOut();
  }, [currentUser]);

  const updateProfile = useCallback(async (fields) => {
    if (!currentUser || currentUser.id === 'demo') {
      return { ok: false, error: 'Profile changes are not saved for the demo account.' };
    }
    try {
      const saved = await saveProfile(fields);
      setCurrentUser(prev => (prev ? { ...prev, ...saved } : prev));
      return { ok: true };
    } catch (error) {
      return { ok: false, error: error.message };
    }
  }, [currentUser]);

  return (
    <AuthContext.Provider
      value={{ currentUser, loading, profileError, login, signup, logout, updateProfile }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider');
  return ctx;
}
