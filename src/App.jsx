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
import ReverseCalculatorScreen from './screens/ReverseCalculatorScreen.jsx';
import './App.css';

const NAV_SECTIONS = [
  {
    title: 'Workspace',
    items: [
      { id: 'home',   label: 'Dashboard',     icon: 'home'  },
      { id: 'input',  label: 'New Valuation',  icon: 'car'   },
      { id: 'result', label: 'Result',         icon: 'bulb'  },
    ],
  },
  {
    title: 'Analysis',
    items: [
      { id: 'pricing',         label: 'Pricing Intel',     icon: 'coins'        },
      { id: 'dashboard',       label: 'Analytics',         icon: 'chart'        },
      { id: 'enhanced-input',  label: 'Enhanced',          icon: 'zap'          },
      { id: 'reverse-calc',    label: 'Reverse Calc',      icon: 'arrowLeftRight'},
    ],
  },
  {
    title: 'Tools',
    items: [
      { id: 'assistant', label: 'AI Assistant', icon: 'brain' },
    ],
  },
];

// Mobile bottom nav — 5 key items only
const MOBILE_NAV = [
  { id: 'home',      label: 'Home',     icon: 'home'   },
  { id: 'input',     label: 'Valuate',  icon: 'car'    },
  { id: 'result',    label: 'Result',   icon: 'bulb'   },
  { id: 'pricing',   label: 'Pricing',  icon: 'coins'  },
  { id: 'dashboard', label: 'Analytics',icon: 'chart'  },
];

// User Avatar Button + Dropdown
function UserMenu() {
  const { currentUser, logout } = useAuth();
  const [open, setOpen] = useState(false);
  if (!currentUser) return null;

  return (
    <div className="user-menu-wrap">
      <button
        className="user-avatar-btn"
        onClick={() => setOpen(o => !o)}
        aria-label="User menu"
      >
        <span className="user-avatar" style={{ background: '#f75d34' }}>
          {currentUser.avatar}
        </span>
      </button>

      {open && (
        <>
          <div className="user-menu-backdrop" onClick={() => setOpen(false)} />
          <div className="user-menu-dropdown">
            <div className="user-menu-profile">
              <div className="user-menu-avatar" style={{ background: '#f75d34' }}>
                {currentUser.avatar}
              </div>
              <div className="user-menu-info">
                <div className="user-menu-name">{currentUser.name}</div>
                <div className="user-menu-email">{currentUser.email}</div>
                <div className="user-menu-role" style={{ color: '#f75d34' }}>
                  <Icon name="store" size={11} color="#f75d34" strokeWidth={2} />
                  Dealer Account
                </div>
              </div>
            </div>
            <div className="user-menu-divider" />
            <button
              className="user-menu-item"
              onClick={() => { setOpen(false); logout(); }}
            >
              <Icon name="arrowLeft" size={15} color="#dc2626" strokeWidth={2} />
              <span style={{ color: '#dc2626' }}>Sign Out</span>
            </button>
          </div>
        </>
      )}
    </div>
  );
}

// Main App Shell
function AppShell() {
  const { activeScreen, setActiveScreen } = useApp();

  const SCREENS = {
    home:             <HomeScreen />,
    input:            <InputScreen />,
    'enhanced-input': <EnhancedValuationScreen />,
    'enhanced-result':<EnhancedResultScreen />,
    'reverse-calc':   <ReverseCalculatorScreen />,
    result:           <ResultScreen />,
    pricing:          <PricingScreen />,
    dashboard:        <DashboardScreen />,
    assistant:        <AssistantScreen />,
  };

  return (
    <div className="app-root">
      {/* Desktop Sidebar */}
      <aside className="app-sidebar">
        {/* Brand */}
        <div className="sidebar-brand">
          <div className="sidebar-logo">
            <Icon name="car" size={18} color="white" strokeWidth={2} />
          </div>
          <div>
            <div className="sidebar-brand-name">PriceRef</div>
            <div className="sidebar-brand-sub">Dealer OS</div>
          </div>
        </div>

        {/* Nav sections */}
        <nav className="sidebar-nav">
          {NAV_SECTIONS.map(section => (
            <div key={section.title} className="sidebar-section">
              <div className="sidebar-section-label">{section.title}</div>
              {section.items.map(item => (
                <button
                  key={item.id}
                  className={`sidebar-nav-btn ${activeScreen === item.id ? 'active' : ''}`}
                  onClick={() => setActiveScreen(item.id)}
                >
                  <Icon
                    name={item.icon}
                    size={16}
                    color={activeScreen === item.id ? '#f75d34' : '#94a3b8'}
                    strokeWidth={activeScreen === item.id ? 2.2 : 1.6}
                  />
                  <span>{item.label}</span>
                </button>
              ))}
            </div>
          ))}
        </nav>

        {/* User at bottom */}
        <div className="sidebar-footer">
          <UserMenu />
        </div>
      </aside>

      {/* Body */}
      <div className="app-body">
        {/* Header */}
        <header className="app-header">
          <div className="header-brand">
            <div className="header-logo-wrap">
              <Icon name="car" size={18} color="white" strokeWidth={2} />
            </div>
            <div>
              <div className="header-name">Price<span className="header-ai">Ref</span></div>
              <div className="header-tagline">Dealer Decision Engine</div>
            </div>
          </div>

          <div className="header-search">
            <Icon name="search" size={15} color="#94a3b8" strokeWidth={2} />
            <input
              type="text"
              placeholder="Search vehicles, evaluations…"
              readOnly
              aria-label="Search"
            />
          </div>

          <div className="header-right">
            <UserMenu />
          </div>
        </header>

        {/* Mobile bottom nav */}
        <nav className="top-nav">
          {MOBILE_NAV.map(tab => (
            <button
              key={tab.id}
              className={`top-nav-btn ${activeScreen === tab.id ? 'active' : ''}`}
              onClick={() => setActiveScreen(tab.id)}
            >
              <Icon
                name={tab.icon}
                size={19}
                color={activeScreen === tab.id ? '#f75d34' : '#94a3b8'}
                strokeWidth={activeScreen === tab.id ? 2.2 : 1.6}
              />
              <span className="tnav-label">{tab.label}</span>
            </button>
          ))}
        </nav>

        {/* Main content */}
        <main className="app-main">
          <div className="screen-wrapper" key={activeScreen}>
            {SCREENS[activeScreen] || <HomeScreen />}
          </div>
        </main>
      </div>
    </div>
  );
}

// Root: Auth gate
function Root() {
  const { currentUser, loading } = useAuth();

  if (loading) {
    return (
      <div className="splash-screen">
        <div className="splash-logo">
          <Icon name="car" size={28} color="white" strokeWidth={2} />
        </div>
        <div className="splash-name">Price<span style={{ color:'#f75d34', fontStyle:'italic' }}>Ref</span></div>
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
