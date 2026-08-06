import { useState } from 'react';
import { useAuth } from '../context/AuthContext.jsx';
import Icon from '../components/Icon.jsx';
export default function AuthScreen() {
  const { login, signup } = useAuth();
  // Tab control: 'signin' or 'signup'
  const [activeTab, setActiveTab] = useState('signin');
  // Sign In inputs
  const [signInEmail, setSignInEmail] = useState('');
  const [signInPassword, setSignInPassword] = useState('');
  // Sign Up inputs
  const [signUpName, setSignUpName] = useState('');
  const [signUpEmail, setSignUpEmail] = useState('');
  const [signUpPassword, setSignUpPassword] = useState('');
  const [showPass, setShowPass]   = useState(false);
  const [error, setError]         = useState('');
  const [loading, setLoading]     = useState(false);
  // Quick fill demo account credentials
  const fillDemo = () => {
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
    const result = await login({ email: signInEmail, password: signInPassword });
    if (!result.ok) {
      setError(result.error || 'Invalid email or password');
    }
    setLoading(false);
  };
  const handleSignUp = async (e) => {
    e.preventDefault();
    if (!signUpName.trim()) { setError('Name is required'); return; }
    if (!signUpEmail.trim()) { setError('Email address is required'); return; }
    if (!signUpPassword) { setError('Password is required'); return; }
    setLoading(true);
    setError('');
    const result = await signup({ name: signUpName, email: signUpEmail, password: signUpPassword });
    if (!result.ok) {
      setError(result.error || 'Registration failed');
    } else {
      setSignUpName('');
      setSignUpEmail('');
      setSignUpPassword('');
    }
    setLoading(false);
  };
  return (
    <div className="auth-root">
      <div className="auth-left">
        <div className="auth-left-brand">
          <div className="auth-left-logo">
            <Icon name="car" size={22} color="white" strokeWidth={2} />
          </div>
          <div>
            <div className="auth-left-title">Price<span>Ref</span></div>
            <div className="auth-left-sub">Dealer Decision OS</div>
          </div>
        </div>
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
        </div>
        <div className="auth-value-props">
          {[
            { icon: 'robot', text: 'Ensemble ML engine trained on market transactions' },
            { icon: 'shield', text: 'Segment-aware pricing — Economy, Premium & Luxury models' },
            { icon: 'coins', text: 'Realistic dealer margin with full cost breakdown' },
            { icon: 'chart', text: 'Live analytics dashboard with deal quality scoring' },
          ].map((p, i) => (
            <div key={i} className="auth-prop">
              <div className="auth-prop-icon">
                <Icon name={p.icon} size={16} color="rgba(255,255,255,0.7)" strokeWidth={1.8} />
              </div>
              <div className="auth-prop-text">{p.text}</div>
            </div>
          ))}
        </div>
      </div>
      <div className="auth-right">
        <div className="auth-form-wrap">
          <div className="auth-mobile-header">
            <Icon name="car" size={28} color="#f75d34" strokeWidth={2.2} />
            <div className="auth-mobile-title">PriceRef</div>
          </div>
          <div className="auth-form-title">
            {activeTab === 'signin' ? 'Welcome back' : 'Create Dealer Account'}
          </div>
          <div className="auth-form-sub" style={{ marginBottom: 24 }}>
            {activeTab === 'signin'
              ? 'Sign in to access your valuations & analytics'
              : 'Register your dealership to get started'}
          </div>
          <div className="auth-tabs" style={{ display: 'flex', gap: 4, background: '#f1f5f9', padding: 4, borderRadius: 10, marginBottom: 24 }}>
            <button
              type="button"
              className={`auth-tab-btn ${activeTab === 'signin' ? 'active' : ''}`}
              onClick={() => { setActiveTab('signin'); setError(''); }}
              style={{
                flex: 1,
                border: 'none',
                padding: '8px 16px',
                fontSize: 13.5,
                fontWeight: 600,
                borderRadius: 8,
                cursor: 'pointer',
                background: activeTab === 'signin' ? '#ffffff' : 'none',
                color: activeTab === 'signin' ? '#0f172a' : '#64748b',
                boxShadow: activeTab === 'signin' ? '0 1px 3px rgba(0,0,0,0.06)' : 'none',
                transition: 'all 0.15s'
              }}
            >
              Sign In
            </button>
            <button
              type="button"
              className={`auth-tab-btn ${activeTab === 'signup' ? 'active' : ''}`}
              onClick={() => { setActiveTab('signup'); setError(''); }}
              style={{
                flex: 1,
                border: 'none',
                padding: '8px 16px',
                fontSize: 13.5,
                fontWeight: 600,
                borderRadius: 8,
                cursor: 'pointer',
                background: activeTab === 'signup' ? '#ffffff' : 'none',
                color: activeTab === 'signup' ? '#0f172a' : '#64748b',
                boxShadow: activeTab === 'signup' ? '0 1px 3px rgba(0,0,0,0.06)' : 'none',
                transition: 'all 0.15s'
              }}
            >
              Sign Up
            </button>
          </div>
          {error && (
            <div className="error-banner" style={{ marginBottom: 20 }}>
              <Icon name="warning" size={14} color="#dc2626" strokeWidth={2} />
              {error}
            </div>
          )}
          {activeTab === 'signin' ? (
            <form onSubmit={handleSignIn} className="auth-form">
              <div className="field-group" style={{ marginBottom: 16 }}>
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
              <div className="field-group" style={{ marginBottom: 24 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <label className="field-label" style={{ marginBottom: 0 }}>Password</label>
                  <button
                    type="button"
                    className="demo-pill-btn"
                    onClick={fillDemo}
                    style={{
                      border: 'none',
                      background: '#eff6ff',
                      color: '#1d4ed8',
                      fontSize: 11,
                      fontWeight: 600,
                      padding: '2px 8px',
                      borderRadius: 4,
                      cursor: 'pointer'
                    }}
                  >
                    Use Demo Account
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
                    style={{
                      position: 'absolute', right: 12, top: '50%',
                      transform: 'translateY(-50%)',
                      background: 'none', border: 'none', cursor: 'pointer',
                      display: 'flex', alignItems: 'center'
                    }}
                  >
                    <Icon name="eye" size={16} color="#94a3b8" strokeWidth={1.8} />
                  </button>
                </div>
              </div>
              <button
                type="submit"
                className="btn btn-primary btn-full btn-lg"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <div className="loading-spinner" style={{ width: 18, height: 18, borderWidth: 2, marginRight: 8 }} />
                    Signing in…
                  </>
                ) : (
                  <>
                    Sign In
                    <Icon name="arrowRight" size={16} color="white" strokeWidth={2.2} />
                  </>
                )}
              </button>
            </form>
          ) : (
            <form onSubmit={handleSignUp} className="auth-form">
              <div className="field-group" style={{ marginBottom: 16 }}>
                <label className="field-label">FullName / Dealership Name</label>
                <input
                  className="field-input"
                  type="text"
                  value={signUpName}
                  onChange={e => setSignUpName(e.target.value)}
                  placeholder="e.g. Sharma Motors"
                  required
                />
              </div>
              <div className="field-group" style={{ marginBottom: 16 }}>
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
              <div className="field-group" style={{ marginBottom: 24 }}>
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
                    style={{
                      position: 'absolute', right: 12, top: '50%',
                      transform: 'translateY(-50%)',
                      background: 'none', border: 'none', cursor: 'pointer',
                      display: 'flex', alignItems: 'center'
                    }}
                  >
                    <Icon name="eye" size={16} color="#94a3b8" strokeWidth={1.8} />
                  </button>
                </div>
              </div>
              <button
                type="submit"
                className="btn btn-primary btn-full btn-lg"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <div className="loading-spinner" style={{ width: 18, height: 18, borderWidth: 2, marginRight: 8 }} />
                    Creating account…
                  </>
                ) : (
                  <>
                    Register & Sign In
                    <Icon name="arrowRight" size={16} color="white" strokeWidth={2.2} />
                  </>
                )}
              </button>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
