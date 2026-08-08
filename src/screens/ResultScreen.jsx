import { useState, useCallback } from 'react';
import { runMLValuationWithVariant } from '../utils/apiValuation.js';
import { useApp } from '../context/AppContext.jsx';
import { exportEvaluationsToCSV } from '../utils/csvExporter.js';
import Icon from '../components/Icon.jsx';

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

const MAIN_VARIANTS = [
  { id: 'variant_1', label: 'Variant 1', mape: '6.16%', emoji: '🥇' },
  { id: 'variant_2', label: 'Variant 2', mape: '6.67%', emoji: '🥈' },
  { id: 'variant_3', label: 'Variant 3', mape: '6.50%', emoji: '🥉' },
];

function VariantSwitcher({ activeVariant, onSwitch, switching, switchError }) {
  return (
    <div style={{
      background: 'var(--surface-1)',
      border: '1px solid var(--border)',
      borderRadius: 12,
      padding: '14px 18px',
      marginBottom: 12,
    }}>
      <div>
        <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-3)', letterSpacing: '0.4px', marginBottom: 8 }}>
          ML MODEL ENGINE
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {MAIN_VARIANTS.map(v => {
            const active = activeVariant === v.id;
            return (
              <button
                key={v.id}
                onClick={() => onSwitch(v.id)}
                disabled={switching}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '7px 14px',
                  borderRadius: 8,
                  border: active ? '2px solid #2563eb' : '1.5px solid var(--border)',
                  background: active ? '#eff6ff' : 'var(--surface-2)',
                  color: active ? '#1d4ed8' : 'var(--text-2)',
                  fontWeight: active ? 700 : 500,
                  fontSize: 13,
                  cursor: switching ? 'not-allowed' : 'pointer',
                  opacity: switching ? 0.6 : 1,
                  transition: 'all 0.15s',
                }}
              >
                <span>{v.emoji}</span>
                <span>{v.label}</span>
                <span style={{ fontSize: 10, color: active ? '#3b82f6' : 'var(--text-3)', fontWeight: 600 }}>
                  {v.mape}
                </span>
                {active && (
                  <span style={{
                    fontSize: 9, background: '#2563eb', color: 'white',
                    borderRadius: 4, padding: '1px 5px', fontWeight: 700,
                  }}>ACTIVE</span>
                )}
              </button>
            );
          })}
          {switching && (
            <span style={{ fontSize: 12, color: '#64748b', alignSelf: 'center', marginLeft: 4 }}>
              Switching…
            </span>
          )}
        </div>
        {switchError && (
          <div style={{
            marginTop: 8,
            padding: '6px 10px',
            background: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: 6,
            fontSize: 12,
            color: '#dc2626',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}>
            <span>⚠️</span>
            <span>{switchError}</span>
          </div>
        )}
      </div>
    </div>
  );
}

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

function SimilarCarsSection({ cars, predictedPrice }) {
  
  const rows = (cars || []).filter(c => c && (c.source === 'dataset' || c.market_value > 0));

  if (rows.length === 0) return null;

  const displayRows = rows.map(c => ({
    brand:        c.brand        || '',
    model:        c.model        || '',
    year:         c.year         || '',
    fuel:         c.fuel         || c.fuel_type || '',
    transmission: c.transmission || 'Manual',
    variant:      c.variant      || '',
    odometer:     Number(c.odometer || c.odometer_reading || 0),
    city:         c.city         || 'Bangalore',
    marketValue:  Number(c.market_value || c.marketValue || 0),
    condition:    c.condition    || 'Good',
    segment:      c.segment      || '',
    ownerCount:   c.owner_count  || '',
    similarity:   Number(c.similarity || 0),
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

      {}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr 0.7fr',
        gap: 8,
        padding: '8px 10px',
        background: 'var(--surface-2)',
        borderRadius: 8,
        marginTop: 14,
        marginBottom: 2,
      }}>
        {['VEHICLE', 'FUEL / TRANS', 'ODOMETER', 'OWNERS', 'LISTED PRICE', 'MATCH'].map(h => (
          <div key={h} style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-3)', letterSpacing: '0.4px' }}>{h}</div>
        ))}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
        {displayRows.map((car, idx) => {
          const diff    = car.marketValue - predictedPrice;
          const diffPct = predictedPrice ? ((diff / predictedPrice) * 100).toFixed(1) : '0.0';
          const isPos   = diff >= 0;
          return (
            <div key={idx} style={{
              display: 'grid',
              gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr 0.7fr',
              gap: 8,
              padding: '11px 10px',
              borderBottom: '1px solid var(--border)',
              alignItems: 'center',
            }}>
              {}
              <div>
                <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-1)' }}>
                  {car.brand} {car.model}{car.variant && car.variant !== 'Unknown' && car.variant !== 'unknown' ? ` · ${car.variant}` : ''}
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>
                  {car.year}{car.condition ? ` · ${car.condition}` : ''}{car.segment ? ` · ${car.segment.toUpperCase()}` : ''}
                </div>
              </div>
              {}
              <div style={{ fontSize: 12, color: 'var(--text-2)' }}>
                {car.fuel || '—'}{car.transmission ? ` / ${car.transmission}` : ''}
              </div>
              {}
              <div style={{ fontSize: 12, color: 'var(--text-2)' }}>
                {car.odometer > 0 ? `${(car.odometer / 1000).toFixed(0)}k km` : '—'}
              </div>
              {}
              <div style={{ fontSize: 12, color: 'var(--text-2)' }}>
                {car.ownerCount ? `${car.ownerCount} owner${car.ownerCount > 1 ? 's' : ''}` : '—'}
              </div>
              {}
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: 13, fontWeight: 800, color: 'var(--text-1)' }}>{fmtFull(car.marketValue)}</div>
                <span style={{
                  fontSize: 10.5, fontWeight: 700,
                  color: isPos ? '#16a34a' : '#dc2626',
                }}>{isPos ? '+' : ''}{diffPct}% vs pred</span>
              </div>
              {}
              {car.similarity > 0 ? (
                <div style={{
                  textAlign: 'center',
                  fontSize: 11,
                  fontWeight: 700,
                  padding: '3px 6px',
                  borderRadius: 8,
                  background: car.similarity >= 85 ? '#dcfce7' : car.similarity >= 65 ? '#dbeafe' : '#fef9c3',
                  color:      car.similarity >= 85 ? '#15803d' : car.similarity >= 65 ? '#1d4ed8' : '#92400e',
                }}>
                  {car.similarity.toFixed(0)}%
                </div>
              ) : <div />}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function ResultScreen() {
  const { valuationResult, inputs, isLoading, setActiveScreen, evaluations } = useApp();

  const [activeVariant, setActiveVariant] = useState('variant_1');
  const [displayResult, setDisplayResult] = useState(null);
  const [switching, setSwitching] = useState(false);
  const [switchError, setSwitchError] = useState(null);
  const [s5Active, setS5Active] = useState(false);

  const result = displayResult || valuationResult;

  const handleVariantSwitch = useCallback(async (variantId) => {
    if (variantId === activeVariant && !s5Active) return;
    setSwitching(true);
    setSwitchError(null);
    setS5Active(false);
    try {
      const switched = await runMLValuationWithVariant(inputs, variantId);
      setDisplayResult(switched);
      setActiveVariant(variantId);
    } catch (e) {
      console.error('Variant switch failed:', e);
      // Extract a user-friendly message from the API error (FastAPI returns
      // {"detail": "..."} in the body; postJson wraps it as "ML API error 404: ...").
      const raw = e?.message || 'Variant switch failed. Please try again.';
      const match = raw.match(/\{.*"detail"\s*:\s*"([^"]+)"/);
      setSwitchError(match ? match[1] : raw);
    } finally {
      setSwitching(false);
    }
  }, [activeVariant, s5Active, inputs]);

  if (isLoading) return <LoadingState />;
  if (!result) return <EmptyState setActiveScreen={setActiveScreen} />;

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
    action = 'BUY',
    segmentClass = 'economy',
    similarCars = [],
    valuationConfidence = 'Low',
    marketSupport = 'Weak',
    comparablesUsed = 0,
    averageSimilarity = 0,
    expectedModelError = 6.3,
  } = result;

  const confColor = {
    'Very High': { bg: '#dcfce7', color: '#15803d', border: '#86efac' },
    'High':      { bg: '#dbeafe', color: '#1d4ed8', border: '#93c5fd' },
    'Medium':    { bg: '#fef9c3', color: '#92400e', border: '#fde047' },
    'Low':       { bg: '#ffedd5', color: '#c2410c', border: '#fdba74' },
    'Very Low':  { bg: '#fee2e2', color: '#b91c1c', border: '#fca5a5' },
  }[valuationConfidence] || { bg: '#f1f5f9', color: '#475569', border: '#cbd5e1' };

  const ac = getAction(action);
  const buyPrice  = Number(recommendedBuyPrice || predictedPrice * 0.82);
  
  const rawSellPrice = Number(recommendedSellPrice || 0);
  const sellPrice = rawSellPrice > buyPrice ? rawSellPrice : Math.round(buyPrice * 1.10 / 500) * 500;
  const profit    = Number(expectedProfit || sellPrice - buyPrice);
  const roi       = buyPrice ? (profit / buyPrice) * 100 : 0;

  const opening  = Number(openingOffer || targetOffer || buyPrice * 0.95);
  const ideal    = Number(targetOffer  || buyPrice);
  const walkAway = Number(maxOffer     || buyPrice * 1.05);

  const minP = Number(priceMin || predictedPrice * 0.9372);
  const maxP = Number(priceMax || predictedPrice * 1.0628);

  const minBuy = Math.round((opening || buyPrice * 0.95) / 500) * 500;
  const maxBuy = Math.round((walkAway || buyPrice * 1.03) / 500) * 500;

  const rangeSub = comparablesUsed > 0
    ? `${comparablesUsed} comps · ${averageSimilarity.toFixed(1)}% avg match · ${marketSupport} support`
    : `ML model uncertainty band · ±${expectedModelError.toFixed(1)}% MAPE`;

  const km = Number(inputs.mileage || 0);

  return (
    <div className="rs2-root">

      {}
      <VariantSwitcher
        activeVariant={activeVariant}
        onSwitch={handleVariantSwitch}
        switching={switching}
        switchError={switchError}
      />

      {}
      {s5Active && (
        <div style={{
          background: '#f5f3ff',
          border: '1.5px solid #c4b5fd',
          borderRadius: 10,
          padding: '10px 16px',
          marginBottom: 12,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          fontSize: 13,
          fontWeight: 600,
          color: '#6d28d9',
        }}>
          <span style={{ fontSize: 16 }}>✨</span>
          <span>
            S5 Quality Shop Model — Pricing from specialized quality dealer dataset (age ≤ 7 yrs).
            These prices reflect premium quality shop benchmarks and may be higher than the general market.
          </span>
        </div>
      )}

      {}
      <div className="rs2-card rs2-hero-card">
        {}
        <div className="rs2-hero-top">
          <div className="rs2-hero-top-left">
            <CarImage />
            <div className="rs2-hero-info-new">
              <div className="rs2-vehicle-name">
                {inputs.brand} {inputs.model}
                {inputs.variant && inputs.variant !== 'unknown' && <span className="rs2-vehicle-var"> {inputs.variant}</span>}
              </div>
              <div className="rs2-spec-row">
                {[
                  inputs.year,
                  inputs.fuel || inputs.fuel_type,
                  inputs.transmission,
                  km > 0 ? `${km.toLocaleString('en-IN')} km` : null,
                  inputs.city,
                  inputs.ownerCount ? `${inputs.ownerCount} Owner${inputs.ownerCount !== '1' ? 's' : ''}` : null
                ].filter(Boolean).map((val, idx) => (
                  <span key={idx} className="rs2-spec-chip">{val}</span>
                ))}
              </div>
              <div className="rs2-hero-badges" style={{ marginTop: '8px' }}>
                <span className="rs2-badge rs2-badge-seg">{(segmentClass || 'economy').toUpperCase()}</span>
                <span className="rs2-badge rs2-badge-conf">ML Confidence: {confidenceScore}%</span>
                <span
                  className="rs2-badge"
                  style={{
                    background: confColor.bg,
                    color: confColor.color,
                    border: `1px solid ${confColor.border}`,
                    fontWeight: 700,
                    fontSize: '10px',
                    letterSpacing: '0.03em',
                  }}
                >
                  {valuationConfidence} Confidence
                </span>
                <span className="rs2-badge" style={{
                  background: '#eff6ff', color: '#1d4ed8',
                  border: '1px solid #bfdbfe', fontWeight: 700, fontSize: '10px',
                }}>
                  🥇 Variant 1 (Default)
                </span>
              </div>
            </div>
          </div>
          <div className="rs2-hero-top-right">
            <span className="rs2-decision-badge-new" style={{ background: ac.bg, color: ac.color, borderColor: ac.border }}>
              <div className="rs2-decision-title-new">
                <Icon name="check" size={16} color={ac.color} strokeWidth={3} /> {ac.label}
              </div>
              <span className="rs2-decision-sub-new">{ac.sub}</span>
            </span>
          </div>
        </div>

        {}
        <div className="rs2-hero-stats-new">
          <div className="rs2-hero-stat-new-item">
            <div className="rs2-hero-stat-label">Market Selling Range</div>
            <div className="rs2-hero-stat-value rs2-blue">{fmt(minP)} – {fmt(maxP)}</div>
            <div className="rs2-hero-stat-sub">{rangeSub}</div>
            {comparablesUsed > 0 && (
              <div style={{
                marginTop: 4,
                display: 'inline-flex',
                alignItems: 'center',
                gap: 4,
                fontSize: 10,
                fontWeight: 600,
                padding: '2px 7px',
                borderRadius: 10,
                background: confColor.bg,
                color: confColor.color,
                border: `1px solid ${confColor.border}`,
              }}>
                {marketSupport} Market Support
              </div>
            )}
          </div>
          <div className="rs2-hero-stat-new-item">
            <div className="rs2-hero-stat-label">Expected Sell Price</div>
            <div className="rs2-hero-stat-value rs2-blue">{fmt(sellPrice)}</div>
            <div className="rs2-hero-stat-sub">After reconditioning</div>
          </div>
          <div className="rs2-hero-stat-new-item">
            <div className="rs2-hero-stat-label">Recommended Buy Range</div>
            <div className="rs2-hero-stat-value rs2-orange">{fmt(minBuy)} – {fmt(maxBuy)}</div>
            <div className="rs2-hero-stat-sub">Ideal acquisition range</div>
          </div>
          <div className="rs2-hero-stat-new-item">
            <div className="rs2-hero-stat-label">Expected Profit</div>
            <div className="rs2-hero-stat-value" style={{ color: profit >= 0 ? '#16a34a' : '#dc2626' }}>
              {fmt(profit)}
            </div>
            <div className="rs2-hero-stat-sub">ROI: {pct(roi)}</div>
          </div>
        </div>
      </div>

      {}
      <PricingBandCard
        title="Market Selling Range"
        icon="chart"
        color="#2563eb"
        min={minP}
        max={maxP}
        confidenceScore={confidenceScore}
      />

      {}
      <PricingBandCard
        title="Recommended Purchase Range"
        icon="coins"
        color="#ea580c"
        min={minBuy}
        max={maxBuy}
        confidenceScore={confidenceScore}
      />

      {}
      <NegotiationSection opening={opening} ideal={ideal} walkAway={walkAway} />

      {}
      <SimilarCarsSection
        cars={similarCars}
        predictedPrice={predictedPrice}
      />

      {}
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
        {evaluations.length > 0 && (
          <button
            className="rs2-btn-ghost"
            onClick={() => exportEvaluationsToCSV(evaluations, 'vehicle_evaluations.csv')}
            title="Export all evaluation history to CSV"
            style={{ background: '#f8fafc', borderColor: '#e2e8f0', color: '#475569', display: 'inline-flex', alignItems: 'center', gap: 6 }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Export CSV
          </button>
        )}
      </div>

    </div>
  );
}

