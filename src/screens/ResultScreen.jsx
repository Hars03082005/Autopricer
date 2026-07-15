import { useState } from 'react';
import { useApp } from '../context/AppContext.jsx';
import Icon from '../components/Icon.jsx';

/* ─── helpers ──────────────────────────────────────────────── */
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
  BUY:       { color: '#16a34a', bg: '#f0fdf4', border: '#86efac', label: 'BUY', sub: 'Good Deal' },
  NEGOTIATE: { color: '#d97706', bg: '#fffbeb', border: '#fcd34d', label: 'NEGOTIATE', sub: 'Review Terms' },
  REJECT:    { color: '#dc2626', bg: '#fef2f2', border: '#fca5a5', label: 'REJECT', sub: 'High Risk' },
  PASS:      { color: '#dc2626', bg: '#fef2f2', border: '#fca5a5', label: 'PASS', sub: 'High Risk' },
};
const getAction = (a = '') =>
  ACTION_CFG[String(a).toUpperCase()] ||
  { color: '#64748b', bg: '#f8fafc', border: '#e2e8f0', label: String(a) || 'REVIEW', sub: 'Manual Check' };

/* ─── Loading ─────────────────────────────────────────────── */
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

/* ─── Empty ───────────────────────────────────────────────── */
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

/* ─── Bold Price Range Display (no bar) ─────────────────────── */
/* Mimics the ₹2,18,899 - ₹2,47,703 style from reference */
function PriceRangeDisplay({ min, max, label, sublabel, color, confScore }) {
  return (
    <div className="rs2-prd-wrap">
      <div className="rs2-prd-header">
        <span className="rs2-prd-label">{label}</span>
        {sublabel && <span className="rs2-prd-sub">{sublabel}</span>}
      </div>
      <div className="rs2-prd-numbers" style={{ color }}>
        <span className="rs2-prd-min">{fmtFull(min)}</span>
        <span className="rs2-prd-dash">—</span>
        <span className="rs2-prd-max">{fmtFull(max)}</span>
      </div>
      {confScore !== undefined && (
        <div className="rs2-prd-conf">
          <span className="rs2-prd-conf-dot" style={{ background: color }} />
          ML Confidence: <strong>{confScore}%</strong>
        </div>
      )}
    </div>
  );
}

/* ─── Car SVG Image ────────────────────────────────────────── */
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

/* ─── Similar Cars Section — Detailed Table ────────────────── */
function SimilarCarsSection({ cars, predictedPrice, inputs, evaluations }) {
  // Fallback to local evaluation history if API returns no similar cars
  const raw = (cars && cars.length > 0)
    ? cars
    : (evaluations || [])
        .filter(t =>
          t.brand === inputs.brand &&
          !(t.year === Number(inputs.year) &&
            t.model === inputs.model &&
            t.marketValue === predictedPrice)
        )
        .slice(0, 5);

  if (raw.length === 0) return null;

  // Normalise so both API objects and history objects work
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
    <div className="rs2-card rs2-similar-table-card">
      <div className="rs2-similar-header">
        <div>
          <div className="rs2-section-title">
            <Icon name="search" size={17} color="#1e40af" strokeWidth={2.2} />
            Similar Cars used for prediction
          </div>
          <div className="rs2-similar-sub">
            Nearest neighbours from the dataset used to estimate this price
          </div>
        </div>
        <button className="rs2-view-all">View All</button>
      </div>

      <div className="rs2-similar-table-wrap">
        <table className="rs2-similar-table">
          <thead>
            <tr>
              <th>Vehicle Details</th>
              <th>Specifications</th>
              <th>Odometer</th>
              <th>Dataset Value</th>
              <th>vs Predicted</th>
              <th>Similarity</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((car, idx) => {
              const isFirst  = idx === 0;
              const diff     = car.marketValue - predictedPrice;
              const diffPct  = predictedPrice ? ((diff / predictedPrice) * 100).toFixed(1) : '0.0';
              const isPos    = diff >= 0;
              return (
                <tr key={idx} className={isFirst ? 'rs2-tr-featured' : ''}>
                  <td>
                    <div className="rs2-table-name">
                      {car.brand} {car.model}
                      {car.variant && car.variant !== '' && (
                        <span className="rs2-table-var">{String(car.variant).toUpperCase()}</span>
                      )}
                    </div>
                    <div className="rs2-table-meta">
                      {car.city && <span>{car.city}</span>}
                      {car.condition && <span> · {car.condition} Condition</span>}
                      {car.segment && <span> · {car.segment}</span>}
                    </div>
                  </td>
                  <td>
                    <div className="rs2-table-specs-primary">
                      {car.year} · {car.fuel}
                    </div>
                    <div className="rs2-table-specs-secondary">
                      {car.transmission}
                    </div>
                  </td>
                  <td>
                    <div className="rs2-table-specs-primary">
                      {car.odometer > 0
                        ? `${Number(car.odometer).toLocaleString('en-IN')} km`
                        : '—'}
                    </div>
                  </td>
                  <td className="rs2-table-price">
                    {fmtFull(car.marketValue)}
                  </td>
                  <td>
                    <span className={`rs2-table-diff ${isPos ? 'rs2-diff-pos' : 'rs2-diff-neg'}`}>
                      {isPos ? '+' : ''}{diffPct}%
                    </span>
                  </td>
                  <td>
                    <span className={`rs2-table-sim-badge${isFirst ? ' featured' : ''}`}>
                      {isFirst ? 'Most Similar' : idx === 1 ? 'High Match' : 'Good Match'}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ─── Negotiation Collapsible ──────────────────────────────── */
function NegotiationSection({ opening, ideal, walkAway }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="rs2-card">
      <div className="rs2-neg-header" onClick={() => setExpanded(e => !e)}>
        <div className="rs2-section-title">
          <Icon name="coins" size={16} color="#1e40af" strokeWidth={2} />
          Negotiation Strategy
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
              <div className="rs2-neg-card-label">OPENING OFFER</div>
              <div className="rs2-neg-card-value" style={{ color: '#16a34a' }}>{fmt(opening)}</div>
              <div className="rs2-neg-card-tip">Start your negotiation here</div>
            </div>
            <div className="rs2-neg-arrow">→</div>
            <div className="rs2-neg-card rs2-neg-amber">
              <div className="rs2-neg-card-label">IDEAL PRICE</div>
              <div className="rs2-neg-card-value" style={{ color: '#d97706' }}>{fmt(ideal)}</div>
              <div className="rs2-neg-card-tip">Target price to aim for</div>
            </div>
            <div className="rs2-neg-arrow">→</div>
            <div className="rs2-neg-card rs2-neg-red">
              <div className="rs2-neg-card-label">WALK AWAY PRICE</div>
              <div className="rs2-neg-card-value" style={{ color: '#dc2626' }}>{fmt(walkAway)}</div>
              <div className="rs2-neg-card-tip">Do not exceed this price</div>
            </div>
          </div>
          <div className="rs2-neg-note">
            <Icon name="bulb" size={13} color="#2563eb" strokeWidth={2} />
            These ranges are AI-powered recommendations based on market data, vehicle condition, and demand trends.
          </div>
        </div>
      )}
    </div>
  );
}

/* ─── Main ──────────────────────────────────────────────────── */
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
    warnings = [],
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
      <div className="rs2-card rs2-hero-card">
        <div className="rs2-hero-left">
          <CarImage />
          <div className="rs2-hero-info">
            <div className="rs2-vehicle-name">
              {inputs.brand} {inputs.model}
              {inputs.variant && <span className="rs2-vehicle-var"> {inputs.variant}</span>}
            </div>
            <div className="rs2-vehicle-meta">
              {inputs.year} · {inputs.fuel || inputs.fuel_type} · {inputs.transmission} · {(km / 1000).toFixed(0)}k km · {inputs.city}
            </div>
            <div className="rs2-hero-badges">
              <span className="rs2-badge rs2-badge-seg">{(segmentClass || 'economy').toUpperCase()}</span>
              {inputs.inspected && <span className="rs2-badge rs2-badge-ok">✓ Inspected</span>}
              <span className="rs2-badge rs2-badge-conf">ML Confidence: {confidenceScore}%</span>
              <span className="rs2-decision-badge" style={{ background: ac.bg, color: ac.color, borderColor: ac.border }}>
                ✓ {ac.label}
                <span className="rs2-decision-sub">{ac.sub}</span>
              </span>
            </div>
          </div>
        </div>

        {/* Right Stats grid */}
        <div className="rs2-hero-stats">
          <div className="rs2-hero-stat">
            <div className="rs2-hero-stat-label">Market Selling Range</div>
            <div className="rs2-hero-stat-value rs2-blue">{fmt(minP)} – {fmt(maxP)}</div>
            <div className="rs2-hero-stat-sub">Based on similar listings</div>
          </div>
          <div className="rs2-hero-stat">
            <div className="rs2-hero-stat-label">Expected Selling Price</div>
            <div className="rs2-hero-stat-value rs2-blue">{fmt(sellPrice)}</div>
            <div className="rs2-hero-stat-sub">After reconditioning</div>
          </div>
          <div className="rs2-hero-stat">
            <div className="rs2-hero-stat-label">Recommended Buy Range</div>
            <div className="rs2-hero-stat-value rs2-orange">{fmt(minBuy)} – {fmt(maxBuy)}</div>
            <div className="rs2-hero-stat-sub">Ideal acquisition range</div>
          </div>
          <div className="rs2-hero-stat">
            <div className="rs2-hero-stat-label">Expected Dealer Profit</div>
            <div className="rs2-hero-stat-value" style={{ color: profit >= 0 ? '#16a34a' : '#dc2626' }}>
              {fmt(profit)}
            </div>
            <div className="rs2-hero-stat-sub">ROI: {pct(roi)}</div>
          </div>
        </div>
      </div>

      {/* ── MARKET SELLING RANGE ─────────────────────────── */}
      <div className="rs2-card rs2-range-card">
        <div className="rs2-range-card-title">
          <Icon name="chart" size={17} color="#1e40af" strokeWidth={2} />
          Market Selling Range
          <span className="rs2-range-card-sub">Based on similar car listings in the dataset</span>
        </div>
        <PriceRangeDisplay
          min={minP}
          max={maxP}
          label="Sell Price Range"
          sublabel={null}
          color="#1d4ed8"
          confScore={confidenceScore}
        />
      </div>

      {/* ── RECOMMENDED BUY RANGE ────────────────────────── */}
      <div className="rs2-card rs2-range-card">
        <div className="rs2-range-card-title">
          <Icon name="coins" size={17} color="#1e40af" strokeWidth={2} />
          Recommended Purchase Range
          <span className="rs2-range-card-sub">Ideal acquisition range for maximum dealer profit</span>
        </div>
        <PriceRangeDisplay
          min={minBuy}
          max={maxBuy}
          label="Buy Price Range"
          sublabel={null}
          color="#ea580c"
          confScore={confidenceScore}
        />
      </div>

      {/* ── NEGOTIATION COLLAPSIBLE ──────────────────────── */}
      <NegotiationSection opening={opening} ideal={ideal} walkAway={walkAway} />

      {/* ── SIMILAR CARS ──────────────────────────────────── */}
      <SimilarCarsSection
        cars={similarCars}
        predictedPrice={predictedPrice}
        inputs={inputs}
        evaluations={evaluations}
      />

      {/* ── WARNINGS ─────────────────────────────────────── */}
      {warnings.length > 0 && (
        <div className="rs2-warnings">
          {warnings.slice(0, 2).map((w, i) => (
            <div key={i} className="rs2-warning-row">
              <Icon name="warning" size={13} color="#d97706" strokeWidth={2} /> {w}
            </div>
          ))}
        </div>
      )}

      {/* ── BOTTOM ACTIONS ───────────────────────────────── */}
      <div className="rs2-actions">
        <button className="rs2-btn-primary" onClick={() => setActiveScreen('pricing')}>
          <Icon name="coins" size={15} color="white" strokeWidth={2} />
          Full Pricing Breakdown
          <Icon name="chevronDown" size={13} color="white" strokeWidth={2} />
        </button>
        <button className="rs2-btn-ghost" onClick={() => setActiveScreen('input')}>
          <Icon name="refresh" size={15} color="#1e293b" strokeWidth={2} />
          New Valuation
        </button>
      </div>

    </div>
  );
}
