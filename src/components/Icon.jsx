/**
 * PriceRef — SVG Icon Library
 * All icons are 2D line-art SVGs, consistent style:
 *   strokeWidth=1.8, strokeLinecap="round", strokeLinejoin="round"
 * Usage: <Icon name="car" size={20} color="#f75d34" />
 */

const ICONS = {
  // Navigation
  home: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M4 10.5L12 4l8 6.5" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M6 9.5V20h12V9.5" strokeLinejoin="round"/>
      <path d="M10 20v-5h4v5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  car: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M5 11l1.5-4.5A2 2 0 0 1 8.4 5h7.2a2 2 0 0 1 1.9 1.5L19 11" strokeLinecap="round" strokeLinejoin="round"/>
      <rect x="2" y="11" width="20" height="7" rx="2" strokeLinejoin="round"/>
      <circle cx="7" cy="18" r="2" fill="currentColor" stroke="none"/>
      <circle cx="17" cy="18" r="2" fill="currentColor" stroke="none"/>
      <path d="M2 14h20" strokeLinecap="round"/>
    </svg>
  ),
  bulb: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M9 18h6M10 21h4" strokeLinecap="round"/>
      <path d="M12 2a7 7 0 0 1 5 11.9c-.6.6-1 1.4-1 2.1H8c0-.7-.4-1.5-1-2.1A7 7 0 0 1 12 2z" strokeLinejoin="round"/>
    </svg>
  ),
  brain: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M9.5 2A3.5 3.5 0 0 0 6 5.5c0 .4.1.8.2 1.1A3.5 3.5 0 0 0 4 10a3.5 3.5 0 0 0 2 3.1V15a3 3 0 0 0 3 3h6a3 3 0 0 0 3-3v-1.9A3.5 3.5 0 0 0 20 10a3.5 3.5 0 0 0-2.2-3.4c.1-.3.2-.7.2-1.1A3.5 3.5 0 0 0 14.5 2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M12 2v16M9 8h6M9 12h6" strokeLinecap="round"/>
    </svg>
  ),
  coins: (
    <svg viewBox="0 0 24 24" fill="none">
      <circle cx="9" cy="14" r="6"/>
      <path d="M15.5 6.5A6 6 0 0 1 15 18" strokeLinecap="round"/>
      <path d="M18.5 4.5A6 6 0 0 1 18 16" strokeLinecap="round"/>
      <path d="M9 11v6M6.5 13.5l2.5-2.5 2.5 2.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  chart: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M3 20h18" strokeLinecap="round"/>
      <path d="M5 20V12M9 20V8M13 20v-6M17 20V4" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),

  // Roles
  store: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M3 9l1.5-5h15L21 9" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M3 9h18v1a3 3 0 0 1-6 0 3 3 0 0 1-6 0 3 3 0 0 1-6 0V9z" strokeLinejoin="round"/>
      <path d="M5 20h14V13H5z" strokeLinejoin="round"/>
      <rect x="9" y="14" width="6" height="6" rx="1"/>
    </svg>
  ),
  person: (
    <svg viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="7" r="4"/>
      <path d="M4 21c0-4 3.6-7 8-7s8 3 8 7" strokeLinecap="round"/>
    </svg>
  ),
  lock: (
    <svg viewBox="0 0 24 24" fill="none">
      <rect x="5" y="11" width="14" height="10" rx="2" strokeLinejoin="round"/>
      <path d="M8 11V7a4 4 0 0 1 8 0v4" strokeLinecap="round"/>
      <circle cx="12" cy="16" r="1.5" fill="currentColor" stroke="none"/>
    </svg>
  ),

  // Input Screen
  search: (
    <svg viewBox="0 0 24 24" fill="none">
      <circle cx="11" cy="11" r="7"/>
      <path d="M16.5 16.5L21 21" strokeLinecap="round"/>
    </svg>
  ),
  camera: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M9 3l-1.5 2H4a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-3.5L15 3H9z" strokeLinejoin="round"/>
      <circle cx="12" cy="12" r="4"/>
    </svg>
  ),
  upload: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M12 15V3M8 7l4-4 4 4" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M3 17v2a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-2" strokeLinecap="round"/>
    </svg>
  ),

  // Fuel types
  fuelPump: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M4 22V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v14" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M4 22h12M16 6l3 3v3a2 2 0 0 0 4 0V8l-4-4" strokeLinecap="round" strokeLinejoin="round"/>
      <rect x="7" y="6" width="6" height="4" rx="1"/>
    </svg>
  ),
  lightning: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z" strokeLinejoin="round"/>
    </svg>
  ),
  oilDrop: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M12 2C8 8 5 12 5 16a7 7 0 0 0 14 0c0-4-3-8-7-14z" strokeLinejoin="round"/>
      <path d="M9 17a3 3 0 0 0 3 3" strokeLinecap="round"/>
    </svg>
  ),
  leaf: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M20 4C15 4 9 5 5 10c-2 3-2 7 0 10l3-3M20 4c0 5-1 11-6 15M20 4L9 15" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  recycle: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M7 19H5a2 2 0 0 1-2-2V9M17 5h2a2 2 0 0 1 2 2v8" strokeLinecap="round"/>
      <path d="M8 5l4-4 4 4M12 1v9M16 19l-4 4-4-4M12 23v-9" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),

  // Result / General
  mapPin: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M12 2a7 7 0 0 1 7 7c0 5-7 13-7 13S5 14 5 9a7 7 0 0 1 7-7z" strokeLinejoin="round"/>
      <circle cx="12" cy="9" r="2.5"/>
    </svg>
  ),
  check: (
    <svg viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="9"/>
      <path d="M8 12l3 3 5-5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  warning: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M12 2L2 21h20L12 2z" strokeLinejoin="round"/>
      <path d="M12 9v4M12 17h.01" strokeLinecap="round"/>
    </svg>
  ),
  robot: (
    <svg viewBox="0 0 24 24" fill="none">
      <rect x="4" y="10" width="16" height="11" rx="2"/>
      <circle cx="9" cy="15" r="2"/>
      <circle cx="15" cy="15" r="2"/>
      <path d="M8 21v1M16 21v1" strokeLinecap="round"/>
      <rect x="9" y="4" width="6" height="4" rx="1"/>
      <path d="M12 4V2M9 10V8M15 10V8" strokeLinecap="round"/>
    </svg>
  ),
  star: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M12 2l3 7h7l-5.5 4 2 7L12 16l-6.5 4 2-7L2 9h7z" strokeLinejoin="round"/>
    </svg>
  ),
  refresh: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M21 12a9 9 0 1 1-2-5.7" strokeLinecap="round"/>
      <path d="M21 4v4h-4" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),

  // Analytics / Dashboard
  trendUp: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M3 17l5-5 4 4 9-9" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M15 7h5v5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  trendDown: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M3 7l5 5 4-4 9 9" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M15 17h5v-5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  map: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M9 3L3 6v15l6-3 6 3 6-3V3l-6 3-6-3z" strokeLinejoin="round"/>
      <path d="M9 3v15M15 6v15" strokeLinecap="round"/>
    </svg>
  ),
  clipboard: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M9 2h6a1 1 0 0 1 1 1v1H8V3a1 1 0 0 1 1-1z" strokeLinejoin="round"/>
      <path d="M6 4h12a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z" strokeLinejoin="round"/>
      <path d="M9 11h6M9 15h4" strokeLinecap="round"/>
    </svg>
  ),
  barChart: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M3 20h18" strokeLinecap="round"/>
      <path d="M7 20V10M12 20V4M17 20V14" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),

  // Info / misc
  bell: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M6 9a6 6 0 1 1 12 0c0 6 2 7 2 7H4s2-1 2-7" strokeLinejoin="round"/>
      <path d="M10 19a2 2 0 0 0 4 0" strokeLinecap="round"/>
    </svg>
  ),
  info: (
    <svg viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="9"/>
      <path d="M12 8v.5M12 11v5" strokeLinecap="round"/>
    </svg>
  ),
  arrowRight: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M5 12h14M13 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  arrowLeft: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M19 12H5M11 18l-6-6 6-6" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  chevronDown: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  sparkle: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M12 2v4M12 18v4M4.9 4.9l2.8 2.8M16.3 16.3l2.8 2.8M2 12h4M18 12h4M4.9 19.1l2.8-2.8M16.3 7.7l2.8-2.8" strokeLinecap="round"/>
      <circle cx="12" cy="12" r="3"/>
    </svg>
  ),
  swap: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M7 4l-4 4 4 4M17 20l4-4-4-4" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M3 8h18M3 16h18" strokeLinecap="round"/>
    </svg>
  ),
  shield: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M12 2l8 3v5c0 5-3.5 9.5-8 11C7.5 19.5 4 15 4 10V5l8-3z" strokeLinejoin="round"/>
      <path d="M9 12l2 2 4-4" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  gauge: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M12 22a9 9 0 1 0-9-9" strokeLinecap="round"/>
      <path d="M12 13l-4-4" strokeLinecap="round"/>
      <circle cx="12" cy="13" r="1.5" fill="currentColor" stroke="none"/>
    </svg>
  ),
  tag: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M12.5 2H4a2 2 0 0 0-2 2v8.5l9.3 9.3a2 2 0 0 0 2.8 0l7-7a2 2 0 0 0 0-2.8L12.5 2z" strokeLinejoin="round"/>
      <circle cx="7.5" cy="8.5" r="1.5" fill="currentColor" stroke="none"/>
    </svg>
  ),
  eye: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" strokeLinejoin="round"/>
      <circle cx="12" cy="12" r="3"/>
    </svg>
  ),
  zap: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z" strokeLinejoin="round"/>
    </svg>
  ),
  arrowLeftRight: (
    <svg viewBox="0 0 24 24" fill="none">
      <path d="M8 3L4 7l4 4M16 21l4-4-4-4" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M4 7h16M20 17H4" strokeLinecap="round"/>
    </svg>
  ),
};

/**
 * Icon component
 * @param {string} name – icon key from ICONS
 * @param {number} size – px size (default 20)
 * @param {string} color – CSS color (default "currentColor")
 * @param {string} className
 */
export default function Icon({ name, size = 20, color = 'currentColor', className = '', strokeWidth = 1.8 }) {
  const svg = ICONS[name];
  if (!svg) return null;

  return (
    <span
      className={`svg-icon ${className}`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: size,
        height: size,
        flexShrink: 0,
        color,
      }}
    >
      {/* Clone SVG with current props */}
      <svg
        viewBox="0 0 24 24"
        width={size}
        height={size}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {svg.props.children}
      </svg>
    </span>
  );
}
