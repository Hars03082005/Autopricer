import { useState } from 'react';
import { useAuth } from '../context/AuthContext.jsx';
import Icon from '../components/Icon.jsx';

export default function AuthScreen() {
  const { login, signup } = useAuth();
  
  const [activeTab, setActiveTab] = useState('signin');
  const [signInEmail, setSignInEmail] = useState('');
  const [signInPassword, setSignInPassword] = useState('');
  const [signUpName, setSignUpName] = useState('');
  const [signUpEmail, setSignUpEmail] = useState('');
  const [signUpPassword, setSignUpPassword] = useState('');
  const [showPass, setShowPass] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const fillDemo = () => {
    setActiveTab('signin');
    setSignInEmail('dealer@PriceRef.ai');
    setSignInPassword('dealer123');
    setError('');
  };

  const handleSignIn = async (e) => {
    e.preventDefault();
    if (!signInEmail.trim()) { setError('Email address is required'); return; }
    if (!signInPassword) { setError('Password is required'); return; }
    setLoading(true);
    setError('');
    await new Promise(r => setTimeout(r, 600));
    const result = login({ email: signInEmail, password: signInPassword });
    if (!result.ok) setError(result.error || 'Invalid email or password');
    setLoading(false);
  };

  const handleSignUp = async (e) => {
    e.preventDefault();
    if (!signUpName.trim()) { setError('Name is required'); return; }
    if (!signUpEmail.trim()) { setError('Email address is required'); return; }
    if (!signUpPassword) { setError('Password is required'); return; }
    setLoading(true);
    setError('');
    await new Promise(r => setTimeout(r, 600));
    const result = signup({ name: signUpName, email: signUpEmail, password: signUpPassword });
    if (!result.ok) {
      setError(result.error || 'Registration failed');
    } else {
      setSignUpName(''); setSignUpEmail(''); setSignUpPassword('');
    }
    setLoading(false);
  };

  const switchTab = (tab) => { setActiveTab(tab); setError(''); };

  return (
    <div className="auth-root">

      {/* ── Left Panel (Hero) ── */}
      <div className="auth-left">

        {/* Brand */}
        <div className="auth-left-brand">
          <div className="auth-left-logo">
            <Icon name="car" size={22} color="white" strokeWidth={2} />
          </div>
          <div>
            <div className="auth-left-title">Price<span>Ref</span></div>
            <div className="auth-left-sub">Dealer Intelligence OS</div>
          </div>
        </div>

        {/* Hero copy */}
        <div>
          <div className="auth-left-tagline">
            Know the exact<br />
            price <span>before</span><br />
            you negotiate.
          </div>
          <div className="auth-left-desc">
            ML-powered valuation engine trusted by dealerships across India.
            Get instant buy/sell recommendations with full cost breakdowns.
          </div>

          {/* Stats bar */}
          <div className="auth-stats">
            <div className="auth-stat">
              <div className="auth-stat-value">50<span>K+</span></div>
              <div className="auth-stat-label">Valuations done</div>
            </div>
            <div className="auth-stat">
              <div className="auth-stat-value">99<span>%</span></div>
              <div className="auth-stat-label">Model accuracy</div>
            </div>
            <div className="auth-stat">
              <div className="auth-stat-value">200<span>+</span></div>
              <div className="auth-stat-label">Active dealers</div>
            </div>
          </div>
        </div>

        {/* Value props */}
        <div className="auth-value-props">
          {[
            { icon: 'robot',  text: 'CatBoost + LightGBM + XGBoost ensemble ML' },
            { icon: 'shield', text: 'Segment-aware — Economy, Premium & Luxury models' },
            { icon: 'coins',  text: 'Dealer margin, refurb & registration cost breakdown' },
            { icon: 'chart',  text: 'Live analytics with deal quality & risk scoring' },
          ].map((p, i) => (
            <div key={i} className="auth-prop">
              <div className="auth-prop-icon">
                <Icon name={p.icon} size={16} color="rgba(255,255,255,0.75)" strokeWidth={1.8} />
              </div>
              <div className="auth-prop-text">{p.text}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Right Panel (Form) ── */}
      <div className="auth-right">
        <div className="auth-form-wrap">

          {/* Mobile header */}
          <div className="auth-mobile-header" style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 28 }}>
            <div style={{ width: 36, height: 36, background: 'linear-gradient(135deg,#f75d34,#ff8c5a)', borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 12px rgba(247,93,52,0.3)' }}>
              <Icon name="car" size={18} color="white" strokeWidth={2} />
            </div>
            <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--text-1)', letterSpacing: -0.3 }}>
              Price<span style={{ color: '#f75d34', fontStyle: 'italic' }}>Ref</span>
            </div>
          </div>

          <div className="auth-form-title" style={{ fontSize: 24, fontWeight: 800, color: 'var(--text-1)', letterSpacing: -0.5, marginBottom: 6 }}>
            {activeTab === 'signin' ? 'Welcome back 👋' : 'Create Account'}
          </div>
          <div style={{ fontSize: 14, color: 'var(--text-3)', marginBottom: 28 }}>
            {activeTab === 'signin'
              ? 'Sign in to your dealership dashboard'
              : 'Register to start ML-powered valuations'}
          </div>

          {/* Tabs */}
          <div style={{ display: 'flex', gap: 4, background: 'var(--surface-2)', padding: 4, borderRadius: 12, marginBottom: 24, border: '1px solid var(--border)' }}>
            {['signin', 'signup'].map(tab => (
              <button
                key={tab}
                type="button"
                onClick={() => switchTab(tab)}
                style={{
                  flex: 1,
                  padding: '9px 16px',
                  fontSize: 13.5,
                  fontWeight: 600,
                  borderRadius: 9,
                  cursor: 'pointer',
                  border: 'none',
                  background: activeTab === tab
                    ? 'linear-gradient(135deg,#f75d34,#ff8c5a)'
                    : 'none',
                  color: activeTab === tab ? 'white' : 'var(--text-3)',
                  boxShadow: activeTab === tab ? '0 2px 8px rgba(247,93,52,0.3)' : 'none',
                  transition: 'all 0.2s',
                  letterSpacing: -0.1,
                }}
              >
                {tab === 'signin' ? 'Sign In' : 'Sign Up'}
              </button>
            ))}
          </div>

          {/* Error */}
          {error && (
            <div className="error-banner" style={{ marginBottom: 20, display: 'flex', alignItems: 'center', gap: 8, background: 'rgba(248,113,113,0.1)', border: '1px solid rgba(248,113,113,0.25)', borderRadius: 10, padding: '10px 14px', fontSize: 13, color: '#f87171' }}>
              <Icon name="warning" size={14} color="#f87171" strokeWidth={2} />
              {error}
            </div>
          )}

          {activeTab === 'signin' ? (
            <form onSubmit={handleSignIn} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div>
                <label className="field-label">Email Address</label>
                <input
                  className="field-input"
                  type="email"
                  value={signInEmail}
                  onChange={e => setSignInEmail(e.target.value)}
                  placeholder="name@dealership.com"
                  autoComplete="email"
                  required
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 7 }}>
                  <label className="field-label" style={{ marginBottom: 0 }}>Password</label>
                  <button
                    type="button"
                    onClick={fillDemo}
                    style={{ border: '1px solid var(--accent-border)', background: 'var(--accent-light)', color: 'var(--accent)', fontSize: 11, fontWeight: 700, padding: '3px 9px', borderRadius: 6, cursor: 'pointer', letterSpacing: 0.2 }}
                  >
                    Use Demo →
                  </button>
                </div>
                <div style={{ position: 'relative' }}>
                  <input
                    className="field-input"
                    type={showPass ? 'text' : 'password'}
                    value={signInPassword}
                    onChange={e => setSignInPassword(e.target.value)}
                    placeholder="••••••••"
                    autoComplete="current-password"
                    style={{ paddingRight: 44 }}
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPass(s => !s)}
                    style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center' }}
                  >
                    <Icon name="eye" size={16} color="var(--text-4)" strokeWidth={1.8} />
                  </button>
                </div>
              </div>

              <button type="submit" className="btn btn-primary btn-full btn-lg" disabled={loading} style={{ marginTop: 8 }}>
                {loading ? (
                  <><div className="loading-spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />Signing in…</>
                ) : (
                  <>Sign In<Icon name="arrowRight" size={16} color="white" strokeWidth={2.2} /></>
                )}
              </button>
            </form>
          ) : (
            <form onSubmit={handleSignUp} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div>
                <label className="field-label">Full Name / Dealership Name</label>
                <input
                  className="field-input"
                  type="text"
                  value={signUpName}
                  onChange={e => setSignUpName(e.target.value)}
                  placeholder="e.g. Sharma Motors"
                  required
                />
              </div>

              <div>
                <label className="field-label">Email Address</label>
                <input
                  className="field-input"
                  type="email"
                  value={signUpEmail}
                  onChange={e => setSignUpEmail(e.target.value)}
                  placeholder="name@dealership.com"
                  required
                />
              </div>

              <div>
                <label className="field-label">Password</label>
                <div style={{ position: 'relative' }}>
                  <input
                    className="field-input"
                    type={showPass ? 'text' : 'password'}
                    value={signUpPassword}
                    onChange={e => setSignUpPassword(e.target.value)}
                    placeholder="Minimum 6 characters"
                    style={{ paddingRight: 44 }}
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPass(s => !s)}
                    style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center' }}
                  >
                    <Icon name="eye" size={16} color="var(--text-4)" strokeWidth={1.8} />
                  </button>
                </div>
              </div>

              <button type="submit" className="btn btn-primary btn-full btn-lg" disabled={loading} style={{ marginTop: 8 }}>
                {loading ? (
                  <><div className="loading-spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />Creating account…</>
                ) : (
                  <>Register & Sign In<Icon name="arrowRight" size={16} color="white" strokeWidth={2.2} /></>
                )}
              </button>
            </form>
          )}

          {/* Footer note */}
          <div style={{ marginTop: 24, textAlign: 'center', fontSize: 12, color: 'var(--text-4)', lineHeight: 1.6 }}>
            Demo account: <span style={{ color: 'var(--text-3)', fontWeight: 600 }}>dealer@PriceRef.ai</span><br />
            Password: <span style={{ color: 'var(--text-3)', fontWeight: 600 }}>dealer123</span>
          </div>
        </div>
      </div>
    </div>
  );
}
