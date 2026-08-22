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
  
  const [error, setError]         = useState('');
  const [loading, setLoading]     = useState(false);

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
      {/* Left Brand Showcase (Desktop) */}
      <div className="auth-left">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div className="sidebar-logo" style={{ width: 34, height: 34 }}>
            <Icon name="car" size={18} color="white" strokeWidth={2} />
          </div>
          <div>
            <div style={{ fontSize: 18, fontWeight: 800, color: '#fff', letterSpacing: -0.4 }}>
              Price<span style={{ color: '#e85d26' }}>Ref</span>
            </div>
            <div style={{ fontSize: 10.5, color: '#9dafc2', fontWeight: 500, letterSpacing: 0.8, textTransform: 'uppercase' }}>
              Dealer Valuation OS
            </div>
          </div>
        </div>

        <div>
          <div style={{ fontSize: 32, fontWeight: 900, color: '#fff', letterSpacing: -1, lineHeight: 1.15, marginBottom: 16 }}>
            Precision vehicle pricing for modern automotive dealerships.
          </div>
          <div style={{ fontSize: 14, color: '#9dafc2', lineHeight: 1.6, maxWidth: 440 }}>
            Powered by comprehensive market transaction intelligence, real-time comparable benchmarking, and dealer acquisition margin waterfalls.
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {[
            'Live acquisition buy/sell price recommendations',
            'Full dealer reconditioning and holding cost breakdown',
            'Real-time market comparable transactions in your city',
            'Automated deal quality scoring and margin protection',
          ].map((text, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 13, color: '#c8d9e8' }}>
              <div style={{ width: 20, height: 20, borderRadius: '50%', background: 'rgba(232,93,38,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <Icon name="check" size={11} color="#e85d26" strokeWidth={2.5} />
              </div>
              <span>{text}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Right Login / Register Panel */}
      <div className="auth-panel">
        <div className="auth-form-wrap">
          <div className="auth-logo">
            <Icon name="car" size={20} color="white" strokeWidth={2} />
          </div>

          <div className="auth-heading">
            {activeTab === 'signin' ? 'Sign In to Dealer Terminal' : 'Create Dealer Account'}
          </div>
          <div className="auth-sub">
            {activeTab === 'signin'
              ? 'Access real-time vehicle valuations, pipeline margins, and deal analytics.'
              : 'Register your dealership account to begin evaluating inventory.'}
          </div>

          {/* Tabs */}
          <div className="auth-tabs">
            <button
              type="button"
              className={`auth-tab ${activeTab === 'signin' ? 'active' : ''}`}
              onClick={() => { setActiveTab('signin'); setError(''); }}
            >
              Sign In
            </button>
            <button
              type="button"
              className={`auth-tab ${activeTab === 'signup' ? 'active' : ''}`}
              onClick={() => { setActiveTab('signup'); setError(''); }}
            >
              Sign Up
            </button>
          </div>

          {error && (
            <div className="auth-error" style={{ marginBottom: 16 }}>
              {error}
            </div>
          )}

          {activeTab === 'signin' ? (
            <form onSubmit={handleSignIn} className="auth-form">
              <div className="form-group">
                <label className="form-label form-label-req">Email Address</label>
                <input
                  type="email"
                  className="form-input"
                  placeholder="dealer@dealership.com"
                  value={signInEmail}
                  onChange={(e) => setSignInEmail(e.target.value)}
                  autoComplete="email"
                  required
                />
              </div>

              <div className="form-group">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <label className="form-label form-label-req">Password</label>
                  <button
                    type="button"
                    onClick={fillDemo}
                    style={{ fontSize: 11.5, color: '#e85d26', fontWeight: 600 }}
                  >
                    Use Demo Account
                  </button>
                </div>
                <input
                  type="password"
                  className="form-input"
                  placeholder="••••••••"
                  value={signInPassword}
                  onChange={(e) => setSignInPassword(e.target.value)}
                  autoComplete="current-password"
                  required
                />
              </div>

              <button
                type="submit"
                className="btn btn-primary btn-lg"
                disabled={loading}
                style={{ width: '100%', justifyContent: 'center', marginTop: 6 }}
              >
                {loading ? 'Authenticating...' : 'Sign In to Terminal'}
              </button>

              <div className="auth-divider">
                <div className="auth-divider-line" />
                <span className="auth-divider-text">DEMO ACCESS</span>
                <div className="auth-divider-line" />
              </div>

              <button
                type="button"
                className="auth-demo-btn"
                onClick={fillDemo}
              >
                Auto-fill Dealer Credentials
              </button>
            </form>
          ) : (
            <form onSubmit={handleSignUp} className="auth-form">
              <div className="form-group">
                <label className="form-label form-label-req">Dealership / User Name</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="e.g. Apex Auto Wheels"
                  value={signUpName}
                  onChange={(e) => setSignUpName(e.target.value)}
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label form-label-req">Email Address</label>
                <input
                  type="email"
                  className="form-input"
                  placeholder="dealer@dealership.com"
                  value={signUpEmail}
                  onChange={(e) => setSignUpEmail(e.target.value)}
                  autoComplete="email"
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label form-label-req">Password</label>
                <input
                  type="password"
                  className="form-input"
                  placeholder="••••••••"
                  value={signUpPassword}
                  onChange={(e) => setSignUpPassword(e.target.value)}
                  autoComplete="new-password"
                  required
                />
              </div>

              <button
                type="submit"
                className="btn btn-primary btn-lg"
                disabled={loading}
                style={{ width: '100%', justifyContent: 'center', marginTop: 6 }}
              >
                {loading ? 'Creating Account...' : 'Register Dealer Account'}
              </button>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
