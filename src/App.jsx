import { useState } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext.jsx';
import { AppProvider, useApp }   from './context/AppContext.jsx';
import Icon          from './components/Icon.jsx';
import AuthScreen    from './screens/AuthScreen.jsx';
import HomeScreen     from './screens/HomeScreen.jsx';
import InputScreen    from './screens/InputScreen.jsx';
import ResultScreen   from './screens/ResultScreen.jsx';
import PricingScreen  from './screens/PricingScreen.jsx';
import DashboardScreen from './screens/DashboardScreen.jsx';
import AssistantScreen from './screens/AssistantScreen.jsx';
import EnhancedValuationScreen from './screens/EnhancedValuationScreen.jsx';
import EnhancedResultScreen from './screens/EnhancedResultScreen.jsx';
import './App.css';

const NAV_SECTIONS = [
  {
    title: 'Workspace',
    items: [
      { id: 'home',   label: 'Dashboard',     icon: 'home'  },
      { id: 'input',  label: 'New Valuation',  icon: 'car'   },
      { id: 'result', label: 'Valuations',    icon: 'bulb'  },
    ],
  },
  {
    title: 'Intelligence',
    items: [
      { id: 'dashboard',    label: 'Market Intel',     icon: 'chart'         },
      { id: 'pricing',      label: 'Deal Financials',  icon: 'coins'         },
    ],
  },
  {
    title: 'Tools',
    items: [
      { id: 'assistant', label: 'Deal Assistant', icon: 'brain' },
    ],
  },
];

const MOBILE_NAV = [
  { id: 'home',      label: 'Dashboard',  icon: 'home'  },
  { id: 'input',     label: 'Valuate',    icon: 'car'   },
  { id: 'result',    label: 'Report',     icon: 'bulb'  },
  { id: 'pricing',   label: 'Financials', icon: 'coins' },
  { id: 'dashboard', label: 'Intel',      icon: 'chart' },
];

function UserMenu({ isSidebar = false }) {
  const { currentUser, logout } = useAuth();
  const [open, setOpen] = useState(false);
  if (!currentUser) return null;

  if (isSidebar) {
    return (
      <div className="sb-user-menu-wrap">
        <button
          className="sb-user-btn"
          onClick={() => setOpen(o => !o)}
          aria-label="User account"
        >
          <span className="sb-user-avatar" style={{ background: '#e85d26' }}>
            {currentUser.avatar || currentUser.name?.[0]?.toUpperCase() || 'D'}
          </span>
          <div style={{ textAlign: 'left', minWidth: 0, flex: 1 }}>
            <div className="sb-user-name">{currentUser.name || 'Dealer User'}</div>
            <div className="sb-user-role">Dealer Account</div>
          </div>
          <Icon name="logout" size={14} color="#64748b" strokeWidth={1.8} />
        </button>

        {open && (
          <>
            <div className="user-menu-backdrop" onClick={() => setOpen(false)} />
            <div className="user-menu-dropdown" style={{ left: 0, bottom: 'calc(100% + 8px)', top: 'auto' }}>
              <div className="user-menu-profile">
                <div className="user-menu-avatar" style={{ background: '#e85d26' }}>
                  {currentUser.avatar || currentUser.name?.[0]?.toUpperCase() || 'D'}
                </div>
                <div className="user-menu-info">
                  <div className="user-menu-name">{currentUser.name}</div>
                  <div className="user-menu-email">{currentUser.email}</div>
                  <div className="user-menu-role" style={{ color: '#e85d26' }}>
                    Dealer Terminal
                  </div>
                </div>
              </div>
              <div className="user-menu-divider" />
              <button
                className="user-menu-item"
                onClick={() => { setOpen(false); logout(); }}
              >
                <Icon name="arrowLeft" size={14} color="#dc2626" strokeWidth={2} />
                <span style={{ color: '#dc2626' }}>Sign Out</span>
              </button>
            </div>
          </>
        )}
      </div>
    );
  }

  return (
    <div className="user-menu-wrap">
      <button
        className="user-avatar-btn"
        onClick={() => setOpen(o => !o)}
        aria-label="User menu"
      >
        <span className="user-avatar" style={{ background: '#e85d26' }}>
          {currentUser.avatar || currentUser.name?.[0]?.toUpperCase() || 'D'}
        </span>
        <span className="user-menu-name-inline">{currentUser.name}</span>
      </button>

      {open && (
        <>
          <div className="user-menu-backdrop" onClick={() => setOpen(false)} />
          <div className="user-menu-dropdown">
            <div className="user-menu-profile">
              <div className="user-menu-avatar" style={{ background: '#e85d26' }}>
                {currentUser.avatar || currentUser.name?.[0]?.toUpperCase() || 'D'}
              </div>
              <div className="user-menu-info">
                <div className="user-menu-name">{currentUser.name}</div>
                <div className="user-menu-email">{currentUser.email}</div>
                <div className="user-menu-role" style={{ color: '#e85d26' }}>
                  Dealer Account
                </div>
              </div>
            </div>
            <div className="user-menu-divider" />
            <button
              className="user-menu-item"
              onClick={() => { setOpen(false); logout(); }}
            >
              <Icon name="arrowLeft" size={14} color="#dc2626" strokeWidth={2} />
              <span style={{ color: '#dc2626' }}>Sign Out</span>
            </button>
          </div>
        </>
      )}
    </div>
  );
}

function AppShell() {
  const { activeScreen, setActiveScreen } = useApp();

  const SCREENS = {
    home:             <HomeScreen />,
    input:            <InputScreen />,
    'enhanced-input': <EnhancedValuationScreen />,
    'enhanced-result':<EnhancedResultScreen />,
    result:           <ResultScreen />,
    pricing:          <PricingScreen />,
    dashboard:        <DashboardScreen />,
    assistant:        <AssistantScreen />,
  };

  return (
    <div className="app-root">
      {/* Dark Sidebar */}
      <aside className="app-sidebar">
        <div className="sidebar-brand">
          <div className="sidebar-logo">
            <Icon name="car" size={16} color="white" strokeWidth={2} />
          </div>
          <div>
            <div className="sidebar-brand-name">PriceRef</div>
            <div className="sidebar-brand-sub">Dealer OS</div>
          </div>
        </div>

        <nav className="sidebar-nav">
          {NAV_SECTIONS.map(section => (
            <div key={section.title} className="sidebar-section">
              <div className="sidebar-section-label">{section.title}</div>
              {section.items.map(item => {
                const isActive = activeScreen === item.id || 
                  (item.id === 'result' && activeScreen === 'enhanced-result') ||
                  (item.id === 'input' && activeScreen === 'enhanced-input');
                return (
                  <button
                    key={item.id}
                    className={`sidebar-nav-btn ${isActive ? 'active' : ''}`}
                    onClick={() => setActiveScreen(item.id)}
                  >
                    <Icon
                      name={item.icon}
                      size={15}
                      color={isActive ? '#e85d26' : '#64748b'}
                      strokeWidth={isActive ? 2.2 : 1.7}
                    />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <UserMenu isSidebar={true} />
        </div>
      </aside>

      {/* Main App Body */}
      <div className="app-body">
        <header className="app-header">
          <div className="header-brand">
            <div className="header-logo-wrap">
              <Icon name="car" size={16} color="white" strokeWidth={2} />
            </div>
            <div>
              <div className="header-name">Price<span className="header-ai">Ref</span></div>
              <div className="header-tagline">Dealer Valuation Terminal</div>
            </div>
          </div>

          <div className="header-center">
            <div className="header-search">
              <Icon name="search" size={14} color="#94a3b8" strokeWidth={2} />
              <input
                type="text"
                placeholder="Search inventory, valuations, models..."
                readOnly
                aria-label="Search"
              />
            </div>
          </div>

          <div className="header-right">
            <button
              className="btn btn-primary btn-sm"
              onClick={() => setActiveScreen('input')}
            >
              <Icon name="car" size={13} color="white" strokeWidth={2} />
              <span>New Valuation</span>
            </button>
            <UserMenu />
          </div>
        </header>

        {/* Mobile/Tablet Tab Bar */}
        <nav className="top-nav">
          {MOBILE_NAV.map(tab => (
            <button
              key={tab.id}
              className={`top-nav-btn ${activeScreen === tab.id ? 'active' : ''}`}
              onClick={() => setActiveScreen(tab.id)}
            >
              <Icon
                name={tab.icon}
                size={18}
                color={activeScreen === tab.id ? '#e85d26' : '#94a3b8'}
                strokeWidth={activeScreen === tab.id ? 2.2 : 1.6}
              />
              <span className="tnav-label">{tab.label}</span>
            </button>
          ))}
        </nav>

        <main className="app-main">
          <div className="screen-wrapper" key={activeScreen}>
            {SCREENS[activeScreen] || <HomeScreen />}
          </div>
        </main>
      </div>
    </div>
  );
}

function Root() {
  const { currentUser, loading } = useAuth();

  if (loading) {
    return (
      <div className="splash-screen">
        <div className="splash-logo">
          <Icon name="car" size={26} color="white" strokeWidth={2} />
        </div>
        <div className="splash-name">Price<span style={{ color: '#e85d26' }}>Ref</span></div>
        <div className="splash-dots">
          <span /><span /><span />
        </div>
      </div>
    );
  }

  if (!currentUser) {
    return (
      <div className="app-root">
        <AuthScreen />
      </div>
    );
  }

  return (
    <AppProvider>
      <AppShell />
    </AppProvider>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Root />
    </AuthProvider>
  );
}
