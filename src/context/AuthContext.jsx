/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useCallback } from 'react';

const AuthContext = createContext(null);

// ── Seed demo accounts ──────────────────────────────────────
const DEMO_ACCOUNTS = {
  'dealer@PriceRef.ai': { password: 'dealer123', name: 'Ramesh Sharma', role: 'Dealer', avatar: 'RS' },
};

const STORAGE_KEY = 'PriceRef_auth';
const USERS_KEY   = 'PriceRef_users';

function loadUsers() {
  try {
    return JSON.parse(localStorage.getItem(USERS_KEY) || '{}');
  } catch { return {}; }
}

function saveUsers(users) {
  localStorage.setItem(USERS_KEY, JSON.stringify(users));
}

function loadSession() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
  } catch { return null; }
}

export function AuthProvider({ children }) {
  const [currentUser, setCurrentUser] = useState(() => loadSession());
  const [loading]                     = useState(false);

  const login = useCallback(({ email, password }) => {
    const em = email.trim().toLowerCase();

    // Check demo accounts
    const demo = DEMO_ACCOUNTS[em];
    if (demo && demo.password === password) {
      const user = { email: em, name: demo.name, role: demo.role, avatar: demo.avatar };
      setCurrentUser(user);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
      return { ok: true };
    }

    // Check registered users
    const users = loadUsers();
    const reg = users[em];
    if (reg && reg.password === password) {
      const user = { email: em, name: reg.name, role: 'Dealer', avatar: reg.avatar };
      setCurrentUser(user);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
      return { ok: true };
    }

    return { ok: false, error: 'Invalid email or password' };
  }, []);

  const signup = useCallback(({ name, email, password }) => {
    const em = email.trim().toLowerCase();
    if (DEMO_ACCOUNTS[em]) return { ok: false, error: 'Email already registered' };

    const users = loadUsers();
    if (users[em]) return { ok: false, error: 'Email already registered' };

    const initials = name.trim().split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2);
    const avatar   = initials || 'U';
    const newUser  = { password, name: name.trim(), role: 'Dealer', avatar };
    users[em] = newUser;
    saveUsers(users);

    const user = { email: em, name: name.trim(), role: 'Dealer', avatar };
    setCurrentUser(user);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
    return { ok: true };
  }, []);

  const logout = useCallback(() => {
    setCurrentUser(null);
    localStorage.removeItem(STORAGE_KEY);
  }, []);

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
