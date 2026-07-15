import { useState } from 'react';
import { useApp } from '../context/AppContext.jsx';
import Icon from '../components/Icon.jsx';

/* ─── Helpers ──────────────────────────────────────────────── */
const fmt = (n) => {
  const v = Number(n || 0);
  if (v >= 10000000) return `₹${(v / 10000000).toFixed(2)}Cr`;
  if (v >= 100000)   return `₹${(v / 100000).toFixed(2)}L`;
  if (v >= 1000)     return `₹${(v / 1000).toFixed(1)}K`;
  return `₹${Math.round(v).toLocaleString('en-IN')}`;
};

const fmtFull = (n) => {
  const v = Number(n || 0);
  return `₹${Math.round(v).toLocaleString('en-IN')}`;
};

const pct = (n) => `${Number(n || 0).toFixed(1)}%`;

const ACTION_CFG = {
  BUY:       { color: '#16a34a', bg: '#f0fdf4', border: '#bbf7d0', label: 'BUY', sub: 'Good Deal' },
  NEGOTIATE: { color: '#ea580c', bg: '#fff7ed', border: '#ffedd5', label: 'NEGOTIATE', sub: 'Review Terms' },
  REJECT:    { color: '#dc2626', bg: '#fef2f2', border: '#fecaca', label: 'REJECT', sub: 'High Risk' },
  PASS:      { color: '#dc2626', bg: '#fef2f2', border: '#fecaca', label: 'PASS', sub: 'High Risk' },
};

const getAction = (a = '') =>
  ACTION_CFG[String(a).toUpperCase()] ||
  { color: '#475569', bg: '#f8fafc', border: '#e2e8f0', label: String(a).toUpperCase() || 'REVIEW', sub: 'Manual Check' };

/* ─── Loading State ────────────────────────────────────────── */
function LoadingState() {
  return (
    <div className="rs2-loading">
      <div className="rs2-loading-spinner" />
      <div className="rs2-loading-title">Analysing vehicle…</div>
      <div className="rs2-loading-steps">
        {['Routing to segment model', 'Running ML inference', 'Computing dealer margins', 'Building negotiation strategy'].map((s, i) => (
          <div key={i} className="rs2-loading-step" style={{ animationDelay: `${i * 0.35}s` }}>
            <Icon name="check" size={11} color="#2563eb" strokeWidth={2.5} /> {s}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Empty State ──────────────────────────────────────────── */
function EmptyState({ setActiveScreen }) {
  return (
    <div className="rs2-empty">
      <div className="rs2-empty-icon">
        <Icon name="car" size={36} color="#93c5fd" strokeWidth={1.5} />
      </div>
      <div className="rs2-empty-title">No valuation yet</div>
      <div className="rs2-empty-sub">Run a vehicle valuation to see ML-powered dealer recommendations</div>
      <button className="btn btn-primary" onClick={() => setActiveScreen('input')}>
        <Icon name="car" size={15} color="white" strokeWidth={2} /> Start Valuation
      </button>
    </div>
  );
}

/* ─── Car Placeholder SVG ──────────────────────────────────── */
function CarImage() {
  return (
    <div className="rs2-car-img-placeholder">
      <svg width="150" height="85" viewBox="0 0 150 85" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect width="150" height="85" rx="10" fill="#EFF6FF"/>
        <rect x="10" y="44" width="130" height="28" rx="6" fill="#BFDBFE"/>
        <path d="M28 44 L46 24 L104 24 L122 44" fill="#93C5FD" stroke="#60A5FA" strokeWidth="1.5"/>
        <circle cx="34" cy="70" r="10" fill="#1E40AF"/>
        <circle cx="34" cy="70" r="5.5" fill="#DBEAFE"/>
        <circle cx="116" cy="70" r="10" fill="#1E40AF"/>
        <circle cx="116" cy="70" r="5.5" fill="#DBEAFE"/>
        <rect x="50" y="28" width="22" height="14" rx="2.5" fill="#DBEAFE" stroke="#93C5FD" strokeWidth="1"/>
        <rect x="76" y="28" width="22" height="14" rx="2.5" fill="#DBEAFE" stroke="#93C5FD" strokeWidth="1"/>
        <rect x="10" y="52" width="15" height="7" rx="2.5" fill="#FEF08A"/>
        <rect x="125" y="52" width="15" height="7" rx="2.5" fill="#FCA5A5"/>
        <rect x="28" y="44" width="94" height="4" rx="2" fill="#60A5FA" opacity="0.4"/>
      </svg>
    </div>
  );
}

/* ─── Pricing Confidence Band card ───────────────────────────── */
function PricingBandCard({ min, max, color, icon, title, confidenceScore }) {
  return (
    <div className="rs2-card rs2-range-card" style={{ padding: '18px 24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ width: 36, height: 36, borderRadius: 8, background: color + '15', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Icon name={icon} size={18} color={color} strokeWidth={2.2} />
          </div>
          <div>
            <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.4px' }}>{title}</div>
            <div style={{ fontSize: '22px', fontWeight: 800, color: 'var(--text-1)', marginTop: 2 }}>
              {fmtFull(min)} – {fmtFull(max)}
            </div>
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-3)' }}>Confidence Interval</div>
          <div style={{ fontSize: '15px', fontWeight: 800, color, marginTop: 2 }}>
            {confidenceScore}% Confidence
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─── Negotiation Strategy ──────────────────────────────────── */
function NegotiationSection({ opening, ideal, walkAway }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="rs2-card" style={{ padding: '20px 24px' }}>
      <div className="rs2-neg-header" onClick={() => setExpanded(e => !e)}>
        <div className="rs2-section-title" style={{ fontSize: '16px', fontWeight: '700' }}>
          <Icon name="coins" size={17} color="#2563eb" strokeWidth={2} />
          Negotiation Strategy
          <Icon name="info" size={14} color="#64748b" className="rs2-section-hint" />
        </div>
        <button className="rs2-neg-toggle">
          {expanded ? 'Hide Details' : 'View Strategy Details'}
          <Icon name={expanded ? 'chevronUp' : 'chevronDown'} size={14} color="#2563eb" strokeWidth={2} />
        </button>
      </div>
      {expanded && (
        <div className="rs2-neg-body">
          <div className="rs2-neg-cards">
            <div className="rs2-neg-card rs2-neg-green">
              <div className="rs2-neg-card-label">Opening Offer</div>
              <div className="rs2-neg-card-value" style={{ color: '#16a34a' }}>{fmtFull(opening)}</div>
              <div className="rs2-neg-card-tip">Start your negotiation here</div>
            </div>
            <div className="rs2-neg-arrow">
              <Icon name="arrowRight" size={20} color="#cbd5e1" strokeWidth={2} />
            </div>
            <div className="rs2-neg-card rs2-neg-amber">
              <div className="rs2-neg-card-label">Ideal Price</div>
              <div className="rs2-neg-card-value" style={{ color: '#ea580c' }}>{fmtFull(ideal)}</div>
              <div className="rs2-neg-card-tip">Target price to aim for</div>
            </div>
            <div className="rs2-neg-arrow">
              <Icon name="arrowRight" size={20} color="#cbd5e1" strokeWidth={2} />
            </div>
            <div className="rs2-neg-card rs2-neg-red">
              <div className="rs2-neg-card-label">Walk Away Price</div>
              <div className="rs2-neg-card-value" style={{ color: '#dc2626' }}>{fmtFull(walkAway)}</div>
              <div className="rs2-neg-card-tip">Do not exceed this price</div>
            </div>
          </div>
          <div className="rs2-neg-note">
            <Icon name="info" size={14} color="#2563eb" strokeWidth={2} />
            These ranges are AI-powered recommendations based on market data, vehicle condition, and demand trends.
          </div>
        </div>
      )}
    </div>
  );
}

/* ─── Similar Cars Section ──────────────────────────────────── */
function SimilarCarsSection({ cars, predictedPrice, inputs, evaluations }) {
  const raw = (cars && cars.length > 0)
    ? cars
    : (evaluations || [])
        .filter(t =>
          t.brand === inputs.brand &&
          !(t.year === Number(inputs.year) &&
            t.model === inputs.model &&
            t.marketValue === predictedPrice)
        )
        .slice(0, 6);

  if (raw.length === 0) return null;

  const rows = raw.map(c => ({
    brand:        c.brand        || inputs.brand        || '',
    model:        c.model        || inputs.model        || '',
    year:         c.year         || inputs.year         || '',
    fuel:         c.fuel         || c.fuel_type         || inputs.fuel         || '',
    transmission: c.transmission || inputs.transmission || 'Manual',
    variant:      c.variant      || inputs.variant      || '',
    odometer:     Number(c.odometer || c.odometer_reading || c.kmDriven || 0),
    city:         c.city         || inputs.city         || '',
    marketValue:  Number(c.market_value || c.marketValue || c.predictedPrice || 0),
    condition:    c.condition    || 'Good',
    segment:      c.segment      || '',
  }));

  return (
    <div className="rs2-card" style={{ padding: '22px 24px' }}>
      <div className="rs2-similar-header">
        <div>
          <div className="rs2-section-title" style={{ fontSize: '16px', fontWeight: '700' }}>
            Similar Cars <span style={{ fontWeight: 'normal', color: '#64748b', fontSize: '14px', marginLeft: '4px' }}>(Based on listings in dataset)</span>
          </div>
        </div>
      </div>

      {/* Table header */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr',
        gap: 8,
        padding: '8px 10px',
        background: 'var(--surface-2)',
        borderRadius: 8,
        marginTop: 14,
        marginBottom: 2,
      }}>
        {['VEHICLE', 'FUEL / TRANS', 'ODOMETER', 'CITY', 'LISTED PRICE'].map(h => (
          <div key={h} style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', letterSpacing: '0.4px' }}>{h}</div>
        ))}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
        {rows.map((car, idx) => {
          const diff    = car.marketValue - predictedPrice;
          const diffPct = predictedPrice ? ((diff / predictedPrice) * 100).toFixed(1) : '0.0';
          const isPos   = diff >= 0;
          return (
            <div key={idx} style={{
              display: 'grid',
              gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr',
              gap: 8,
              padding: '11px 10px',
              borderBottom: '1px solid var(--border)',
              alignItems: 'center',
            }}>
              {/* Vehicle */}
              <div>
                <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-1)' }}>
                  {car.brand} {car.model}{car.variant && car.variant !== 'unknown' ? ` ${car.variant}` : ''}
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>
                  {car.year}{car.condition ? ` · ${car.condition}` : ''}{car.segment ? ` · ${car.segment.toUpperCase()}` : ''}
                </div>
              </div>
              {/* Fuel / Trans */}
              <div style={{ fontSize: 12, color: 'var(--text-2)' }}>
                {car.fuel || '—'}{car.transmission ? ` / ${car.transmission}` : ''}
              </div>
              {/* Odometer */}
              <div style={{ fontSize: 12, color: 'var(--text-2)' }}>
                {car.odometer > 0 ? `${(car.odometer / 1000).toFixed(0)}k km` : '—'}
              </div>
              {/* City */}
              <div style={{ fontSize: 12, color: 'var(--text-2)' }}>{car.city || '—'}</div>
              {/* Price + diff */}
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: 13, fontWeight: 800, color: 'var(--text-1)' }}>{fmtFull(car.marketValue)}</div>
                <span style={{
                  fontSize: 10.5, fontWeight: 700,
                  color: isPos ? '#16a34a' : '#dc2626',
                }}>{isPos ? '+' : ''}{diffPct}% vs pred</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ─── Main Component ────────────────────────────────────────── */
export default function ResultScreen() {
  const { valuationResult, inputs, isLoading, setActiveScreen, evaluations } = useApp();

  if (isLoading) return <LoadingState />;
  if (!valuationResult) return <EmptyState setActiveScreen={setActiveScreen} />;

  const {
    predictedPrice = 0,
    priceMin,
    priceMax,
    confidenceScore = 80,
    recommendedBuyPrice = 0,
    recommendedSellPrice = 0,
    openingOffer,
    maxOffer,
    targetOffer,
    expectedProfit = 0,
    expectedMarginPct = 0,
    action = 'BUY',
    segmentClass = 'economy',
    similarCars = [],
  } = valuationResult;

  const ac = getAction(action);
  const buyPrice  = Number(recommendedBuyPrice || predictedPrice * 0.82);
  const sellPrice = Number(recommendedSellPrice || predictedPrice * 1.08);
  const profit    = Number(expectedProfit || sellPrice - buyPrice);
  const roi       = buyPrice ? (profit / buyPrice) * 100 : 0;

  const opening  = Number(openingOffer || targetOffer || buyPrice * 0.95);
  const ideal    = Number(targetOffer  || buyPrice);
  const walkAway = Number(maxOffer     || buyPrice * 1.05);

  const minP = Number(priceMin || predictedPrice * 0.88);
  const maxP = Number(priceMax || predictedPrice * 1.12);

  const minBuy = Math.round(buyPrice * 0.95 / 500) * 500;
  const maxBuy = Math.round(buyPrice * 1.03 / 500) * 500;

  const km = Number(inputs.mileage || 0);

  return (
    <div className="rs2-root">

      {/* ── VEHICLE HEADER CARD ─────────────────────────── */}
      <div className="rs2-card rs2-hero-card" style={{ padding: '22px 26px' }}>
        <div className="rs2-hero-left">
          <CarImage />
          <div className="rs2-hero-info">
            <div className="rs2-vehicle-name">
              {inputs.brand} {inputs.model}
              {inputs.variant && inputs.variant !== 'unknown' && <span className="rs2-vehicle-var"> {inputs.variant}</span>}
            </div>
            {/* Car details in a compact grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '4px 12px', marginTop: 8 }}>
              {[
                { label: 'Year',  val: inputs.year },
                { label: 'Fuel',  val: inputs.fuel || inputs.fuel_type },
                { label: 'Trans', val: inputs.transmission },
                { label: 'Odometer', val: km > 0 ? `${km.toLocaleString('en-IN')} km` : '—' },
                { label: 'City',  val: inputs.city },
                { label: 'Owners', val: inputs.ownerCount ? `${inputs.ownerCount} Owner${inputs.ownerCount !== '1' ? 's' : ''}` : '—' },
              ].map(({ label, val }) => val ? (
                <div key={label}>
                  <div style={{ fontSize: 9.5, color: 'var(--text-3)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.3px' }}>{label}</div>
                  <div style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-1)', marginTop: 1 }}>{val}</div>
                </div>
              ) : null)}
            </div>
            <div className="rs2-hero-badges" style={{ marginTop: '10px' }}>
              <span className="rs2-badge rs2-badge-seg">{(segmentClass || 'economy').toUpperCase()}</span>
              <span className="rs2-badge rs2-badge-conf">ML Confidence: {confidenceScore}%</span>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'flex-end', marginLeft: 'auto' }}>
            <span className="rs2-decision-badge" style={{ background: ac.bg, color: ac.color, borderColor: ac.border, padding: '10px 18px', borderRadius: '12px' }}>
              <div style={{ fontSize: '18px', fontWeight: '800', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Icon name="check" size={16} color={ac.color} strokeWidth={3} /> {ac.label}
              </div>
              <span className="rs2-decision-sub" style={{ fontSize: '11px', color: ac.color, fontWeight: '600', marginTop: '2px' }}>{ac.sub}</span>
            </span>
          </div>
        </div>

        {/* Right Stats grid */}
        <div className="rs2-hero-stats">
          <div className="rs2-hero-stat">
            <div className="rs2-hero-stat-label">Market Selling Range</div>
            <div className="rs2-hero-stat-value rs2-blue" style={{ fontSize: '15px' }}>{fmt(minP)} – {fmt(maxP)}</div>
            <div className="rs2-hero-stat-sub">Based on similar listings</div>
          </div>
          <div className="rs2-hero-stat">
            <div className="rs2-hero-stat-label">Expected Sell Price</div>
            <div className="rs2-hero-stat-value rs2-blue" style={{ fontSize: '15px' }}>{fmt(sellPrice)}</div>
            <div className="rs2-hero-stat-sub">After reconditioning</div>
          </div>
          <div className="rs2-hero-stat">
            <div className="rs2-hero-stat-label">Recommended Buy Range</div>
            <div className="rs2-hero-stat-value rs2-orange" style={{ fontSize: '15px' }}>{fmt(minBuy)} – {fmt(maxBuy)}</div>
            <div className="rs2-hero-stat-sub">Ideal acquisition range</div>
          </div>
          <div className="rs2-hero-stat">
            <div className="rs2-hero-stat-label">Expected Profit</div>
            <div className="rs2-hero-stat-value" style={{ color: profit >= 0 ? '#16a34a' : '#dc2626', fontSize: '15px' }}>
              {fmt(profit)}
            </div>
            <div className="rs2-hero-stat-sub">ROI: {pct(roi)}</div>
          </div>
        </div>
      </div>

      {/* ── MARKET SELLING RANGE ─────────────────────────── */}
      <PricingBandCard
        title="Market Selling Range"
        icon="chart"
        color="#2563eb"
        min={minP}
        max={maxP}
        confidenceScore={confidenceScore}
      />

      {/* ── RECOMMENDED BUY RANGE ────────────────────────── */}
      <PricingBandCard
        title="Recommended Purchase Range"
        icon="coins"
        color="#ea580c"
        min={minBuy}
        max={maxBuy}
        confidenceScore={confidenceScore}
      />

      {/* ── NEGOTIATION COLLAPSIBLE ──────────────────────── */}
      <NegotiationSection opening={opening} ideal={ideal} walkAway={walkAway} />

      {/* ── SIMILAR CARS ──────────────────────────────────── */}
      <SimilarCarsSection
        cars={similarCars}
        predictedPrice={predictedPrice}
        inputs={inputs}
        evaluations={evaluations}
      />

      {/* ── BOTTOM ACTIONS ───────────────────────────────── */}
      <div className="rs2-actions" style={{ justifyContent: 'center', marginTop: '16px' }}>
        <button className="rs2-btn-primary" onClick={() => setActiveScreen('pricing')}>
          <Icon name="coins" size={15} color="white" strokeWidth={2} />
          Full Pricing Breakdown
          <Icon name="chevronDown" size={13} color="white" strokeWidth={2} />
        </button>
        <button className="rs2-btn-ghost" onClick={() => setActiveScreen('input')} style={{ background: '#ffffff', borderColor: '#e2e8f0', color: '#1e293b' }}>
          <Icon name="refresh" size={15} color="#475569" strokeWidth={2} />
          New Valuation
        </button>
      </div>

    </div>
  );
}
