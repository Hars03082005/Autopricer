import { useState } from 'react';
import { useAuth } from '../context/AuthContext.jsx';
import Icon from '../components/Icon.jsx';

const DEMO_ACCOUNTS = [
  {
    label: 'Dealer Manager',
    email: 'dealer@PriceRef.ai',
    password: 'dealer123',
    role: 'Dealer Account',
    avatar: 'D',
  },
];

export default function AuthScreen() {
  const { login } = useAuth();
  const [selected, setSelected]   = useState(null);
  const [password, setPassword]   = useState('');
  const [showPass, setShowPass]   = useState(false);
  const [error, setError]         = useState('');
  const [loading, setLoading]     = useState(false);

  const handleSelect = (acc) => {
    setSelected(acc);
    setPassword(acc.password);
    setError('');
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    if (!selected) { setError('Please select an account'); return; }
    if (!password) { setError('Password required'); return; }
    setLoading(true);
    setError('');
    await new Promise(r => setTimeout(r, 600));
    const result = login({ email: selected.email, password });
    if (!result.ok) setError(result.error || 'Invalid credentials');
    setLoading(false);
  };

  return (
    <div className="auth-root">
      {/* Left panel — dark marketing panel */}
      <div className="auth-left">
        <div className="auth-left-brand">
          <div className="auth-left-logo">
            <Icon name="car" size={22} color="white" strokeWidth={2} />
          </div>
          <div>
            <div className="auth-left-title">Pricer<span>Point</span></div>
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
            { icon: 'robot', text: 'CatBoost ML engine trained on 213,820 real transactions' },
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

      {/* Right panel — login form */}
      <div className="auth-right">
        <div className="auth-form-wrap">
          <div className="auth-form-title">Welcome back</div>
          <div className="auth-form-sub">Sign in to your dealer account</div>

          {/* Account selector */}
          <div className="auth-accounts">
            {DEMO_ACCOUNTS.map((acc) => (
              <button
                key={acc.email}
                className={`auth-account-btn ${selected?.email === acc.email ? 'selected' : ''}`}
                onClick={() => handleSelect(acc)}
                type="button"
              >
                <div className="auth-account-info">
                  <div className="auth-account-avatar">{acc.avatar}</div>
                  <div>
                    <div className="auth-account-name">{acc.label}</div>
                    <div className="auth-account-meta">{acc.email}</div>
                  </div>
                </div>
                <div>
                  {selected?.email === acc.email && (
                    <Icon name="check" size={16} color="#f75d34" strokeWidth={2.5} />
                  )}
                </div>
              </button>
            ))}
          </div>

          {/* Password form */}
          <form onSubmit={handleLogin}>
            <div className="field-group auth-password-row">
              <label className="field-label">Password</label>
              <div style={{ position: 'relative' }}>
                <input
                  className="field-input"
                  type={showPass ? 'text' : 'password'}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="Enter password…"
                  autoComplete="current-password"
                  style={{ paddingRight: 44 }}
                />
                <button
                  type="button"
                  onClick={() => setShowPass(s => !s)}
                  style={{
                    position: 'absolute', right: 12, top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none', border: 'none', cursor: 'pointer',
                  }}
                >
                  <Icon name="eye" size={16} color="#94a3b8" strokeWidth={1.8} />
                </button>
              </div>
            </div>

            {error && (
              <div className="error-banner" style={{ marginBottom: 16 }}>
                <Icon name="warning" size={14} color="#dc2626" strokeWidth={2} />
                {error}
              </div>
            )}

            <button
              type="submit"
              className="btn btn-primary btn-full btn-lg"
              disabled={loading}
              style={{ marginBottom: 0 }}
            >
              {loading ? (
                <>
                  <div className="loading-spinner" style={{ width: 18, height: 18, borderWidth: 2, marginRight: 4 }} />
                  Signing in…
                </>
              ) : (
                <>
                  Sign in to PriceRef
                  <Icon name="arrowRight" size={16} color="white" strokeWidth={2.2} />
                </>
              )}
            </button>
          </form>

          <div className="auth-terms">
            Demo account · <strong>dealer@PriceRef.ai</strong> / dealer123
          </div>
        </div>
      </div>
    </div>
  );
}
