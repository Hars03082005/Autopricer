/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { supabase } from '../lib/supabaseClient.js';

const AuthContext = createContext(null);

// Local demo account bypass (no Supabase needed for demo)
const DEMO_ACCOUNTS = {
  'dealer@priceref.ai': { password: 'dealer123', name: 'Ramesh Sharma', role: 'Dealer', avatar: 'RS' },
};

function buildUser(supabaseUser, profile) {
  const name   = profile?.name   || supabaseUser.email.split('@')[0];
  const avatar = profile?.avatar || name.slice(0, 2).toUpperCase();
  const role   = profile?.role   || 'Dealer';
  return { id: supabaseUser.id, email: supabaseUser.email, name, role, avatar };
}

export function AuthProvider({ children }) {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading]         = useState(true);

  // On mount: restore session from Supabase and listen for changes
  useEffect(() => {
    // Get current session
    supabase.auth.getSession().then(async ({ data: { session } }) => {
      if (session?.user) {
        const { data: profile } = await supabase
          .from('profiles')
          .select('name, avatar, role')
          .eq('id', session.user.id)
          .single();
        setCurrentUser(buildUser(session.user, profile));
      }
      setLoading(false);
    });

    // Subscribe to auth state changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(async (event, session) => {
      if (session?.user) {
        const { data: profile } = await supabase
          .from('profiles')
          .select('name, avatar, role')
          .eq('id', session.user.id)
          .single();
        setCurrentUser(buildUser(session.user, profile));
      } else {
        setCurrentUser(null);
      }
    });

    return () => subscription.unsubscribe();
  }, []);

  const login = useCallback(async ({ email, password }) => {
    const em = email.trim().toLowerCase();

    // Demo account bypass
    const demo = DEMO_ACCOUNTS[em];
    if (demo && demo.password === password) {
      setCurrentUser({ id: 'demo', email: em, name: demo.name, role: demo.role, avatar: demo.avatar });
      return { ok: true };
    }

    const { error } = await supabase.auth.signInWithPassword({ email: em, password });
    if (error) return { ok: false, error: error.message };
    return { ok: true };
  }, []);

  const signup = useCallback(async ({ name, email, password }) => {
    const em      = email.trim().toLowerCase();
    const trimmed = name.trim();
    const avatar  = trimmed.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2) || 'U';

    if (DEMO_ACCOUNTS[em]) return { ok: false, error: 'Email already registered' };

    const { data, error } = await supabase.auth.signUp({ email: em, password });
    if (error) return { ok: false, error: error.message };

    // Insert profile row
    if (data.user) {
      await supabase.from('profiles').insert({
        id:   data.user.id,
        name: trimmed,
        avatar,
        role: 'Dealer',
      });
    }

    return { ok: true };
  }, []);

  const logout = useCallback(async () => {
    // Demo user: just clear state
    if (currentUser?.id === 'demo') {
      setCurrentUser(null);
      return;
    }
    await supabase.auth.signOut();
  }, [currentUser]);

  return (
    <AuthContext.Provider value={{ currentUser, loading, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider');
  return ctx;
}
