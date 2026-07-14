import { useApp } from '../context/AppContext.jsx';
import Icon from '../components/Icon.jsx';

/* ─── helpers ────────────────────────────────────────────── */
const fmtL = (n) => {
  if (!n || isNaN(n)) return '₹0';
  const v = Number(n);
  if (v >= 10000000) return `₹${(v/10000000).toFixed(2)}Cr`;
  if (v >= 100000)   return `₹${(v/100000).toFixed(2)}L`;
  if (v >= 1000)     return `₹${Math.round(v/1000)}K`;
  return `₹${Math.round(v).toLocaleString()}`;
};

const fmtLarge = (n) => {
  const v = Number(n || 0);
  if (v >= 100000) return (v/100000).toFixed(2);
  return '0.00';
};

const getActionInfo = (action = '') => {
  const a = String(action).toUpperCase();
  if (a === 'BUY') return { cls: 'buy', label: '✓ BUY', color: '#16a34a' };
  if (a === 'NEGOTIATE') return { cls: 'negotiate', label: '⟳ NEGOTIATE', color: '#d97706' };
  if (a === 'REJECT' || a === 'PASS') return { cls: 'reject', label: '✕ PASS', color: '#dc2626' };
  return { cls: 'review', label: '⊘ REVIEW', color: '#94a3b8' };
};

const getRiskClass = (score) => {
  if (score <= 35) return 'low';
  if (score <= 65) return 'mid';
  return 'high';
};

const getRiskLabel = (score) => {
  if (score <= 35) return 'Low';
  if (score <= 65) return 'Medium';
  return 'High';
};

function ConfidencePill({ score }) {
  const cls = score >= 75 ? 'high' : score >= 50 ? 'medium' : 'low';
  const label = score >= 75 ? 'High Confidence' : score >= 50 ? 'Medium' : 'Low Confidence';
  return (
    <span className={`conf-pill ${cls}`}>
      <Icon name="shield" size={11} strokeWidth={2.2}
        color={cls === 'high' ? '#16a34a' : cls === 'medium' ? '#d97706' : '#dc2626'} />
      {score}% · {label}
    </span>
  );
}

function WaterfallItem({ label, value, pct, color = 'gray', deduct = false }) {
  return (
    <div className="waterfall-row">
      <div className="waterfall-label">{label}</div>
      <div className="waterfall-bar-track">
        <div
          className={`waterfall-bar-fill ${color}`}
          style={{ width: `${Math.min(100, Math.max(2, pct))}%` }}
        />
      </div>
      <div className="waterfall-val" style={{ color: deduct ? '#dc2626' : 'var(--text-1)' }}>
        {deduct ? '−' : ''}{fmtL(Math.abs(value))}
      </div>
    </div>
  );
}

function RiskItem({ label, score, sub }) {
  const cls = getRiskClass(score);
  return (
    <div className="risk-item">
      <div>
        <div className="risk-item-label">{label}</div>
        {sub && <div className="risk-item-sub">{sub}</div>}
      </div>
      <div className={`risk-score-badge ${cls}`}>
        {score}
      </div>
    </div>
  );
}

/* ─── Loading state ──────────────────────────────────────── */
function ResultLoading() {
  const steps = [
    'Routing to segment model…',
    'Running CatBoost inference…',
    'Computing dealer margins…',
    'Building risk profile…',
  ];
  return (
    <div className="screen">
      <div className="loading-screen">
        <div className="loading-spinner" />
        <div className="loading-label">Analysing vehicle with ML…</div>
        <div className="loading-steps-list">
          {steps.map((s, i) => (
            <div key={i} className="loading-step-item" style={{ animationDelay: `${i * 0.4}s` }}>
              <Icon name="check" size={13} color="#16a34a" strokeWidth={2.5} />
              {s}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ─── Empty state ────────────────────────────────────────── */
function ResultEmpty({ setActiveScreen }) {
  return (
    <div className="screen">
      <div className="empty-screen">
        <div className="home-empty-icon">
          <Icon name="bulb" size={28} color="#94a3b8" strokeWidth={1.8} />
        </div>
        <div className="empty-title">No result yet</div>
        <div className="empty-sub">
          Run a vehicle valuation to see ML-powered buy/sell recommendations and ROI analysis.
        </div>
        <button className="btn btn-primary btn-lg" onClick={() => setActiveScreen('input')}>
          <Icon name="car" size={16} color="white" strokeWidth={2} />
          Start Valuation
        </button>
      </div>
    </div>
  );
}

/* ─── Main Result Screen ─────────────────────────────────── */
export default function ResultScreen() {
  const { valuationResult, inputs, isLoading, setActiveScreen } = useApp();

  if (isLoading) return <ResultLoading />;
  if (!valuationResult) return <ResultEmpty setActiveScreen={setActiveScreen} />;

  const {
    predictedPrice,
    priceMin, priceMax,
    confidenceScore = 80,
    conditionScore  = 75,
    recommendedBuyPrice,
    openingOffer,
    maxOffer,
    recommendedSellPrice,
    expectedProfit  = 0,
    expectedMarginPct = 0,
    action          = 'MANUAL REVIEW',
    positiveFactors = [],
    negativeFactors = [],
    warnings        = [],
    riskScore       = 50,
    riskLevel       = 'Medium',
    segmentClass    = 'economy',
    holdingCost     = 0,
    riskBuffer      = 0,
    repairBuffer    = 0,
  } = valuationResult;

  const actionInfo     = getActionInfo(action);
  const marketValueNum = Number(predictedPrice || 0);
  const buyPrice       = Number(recommendedBuyPrice || marketValueNum * 0.82);
  const sellPrice      = Number(recommendedSellPrice || marketValueNum * 1.08);
  const profit         = Number(expectedProfit || sellPrice - buyPrice);
  const marginPct      = Number(expectedMarginPct || ((profit / buyPrice) * 100).toFixed(1));
  const opening        = Number(openingOffer || buyPrice * 0.97);
  const walkAway       = Number(maxOffer || buyPrice);

  // Risk items computed from available data
  const ageYears       = new Date().getFullYear() - Number(inputs.year || 2020);
  const mileageNum     = Number(inputs.mileage || 0);
  const ownerCount     = Number(inputs.ownerCount || 1);

  const riskItems = [
    {
      label: 'Mechanical',
      score: Math.max(5, Math.round(100 - (conditionScore || 75))),
      sub: `Condition: ${inputs.condition || 'Good'}`,
    },
    {
      label: 'Market Demand',
      score: Math.round(riskScore * 0.6),
      sub: `${inputs.city || 'Local'} market`,
    },
    {
      label: 'Ownership',
      score: Math.min(90, ownerCount * 20),
      sub: `${ownerCount} previous owner${ownerCount > 1 ? 's' : ''}`,
    },
    {
      label: 'Depreciation',
      score: Math.min(85, ageYears * 9),
      sub: `${ageYears} years old`,
    },
    {
      label: 'Mileage',
      score: Math.min(90, Math.round(mileageNum / 2000)),
      sub: `${(mileageNum/1000).toFixed(0)}k km driven`,
    },
    {
      label: 'Overall Risk',
      score: riskScore,
      sub: riskLevel,
    },
  ];

  // Waterfall denominator = buyPrice
  const wfBase = buyPrice;
  const wfPct  = (v) => (Math.abs(v) / wfBase) * 100;

  return (
    <div className="screen">
      {/* ── Hero Card ─────────────────────────────────────── */}
      <div className="result-hero-card">
        <div className="result-hero-top">
          <div>
            <div className="result-vehicle-name">
              {inputs.brand} {inputs.model}
              {inputs.variant && <span style={{ opacity:0.65 }}> {inputs.variant}</span>}
            </div>
            <div className="result-vehicle-spec">
              {inputs.year} · {inputs.fuel} · {inputs.transmission} ·{' '}
              {(mileageNum/1000).toFixed(0)}k km · {inputs.city}
            </div>
            <div style={{ display:'flex', gap:8, alignItems:'center', marginTop:10, flexWrap:'wrap' }}>
              <span className={`segment-badge ${segmentClass}`}>
                {segmentClass?.toUpperCase()}
              </span>
              {inputs.inspected && (
                <span className="badge badge-success" style={{ fontSize:10 }}>
                  ✓ Inspected
                </span>
              )}
            </div>
          </div>

          <div className="result-market-value" style={{ textAlign:'right' }}>
            <div className="result-market-label">ML Market Value</div>
            <div className="result-market-price">
              <span className="currency">₹</span>
              {fmtLarge(predictedPrice)}L
            </div>
            <div className="result-ci-row" style={{ justifyContent:'flex-end' }}>
              <span className="result-ci-label">Range</span>
              <span className="result-ci-range">
                {fmtL(priceMin)} – {fmtL(priceMax)}
              </span>
            </div>
          </div>
        </div>

        <div className="result-hero-bottom">
          <ConfidencePill score={confidenceScore} />
          <div style={{ display:'flex', gap:8, flexWrap:'wrap' }}>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => setActiveScreen('pricing')}
              style={{ background:'rgba(255,255,255,0.1)', border:'1px solid rgba(255,255,255,0.15)', color:'rgba(255,255,255,0.8)' }}
            >
              Pricing Detail
            </button>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => setActiveScreen('explain')}
              style={{ background:'rgba(255,255,255,0.1)', border:'1px solid rgba(255,255,255,0.15)', color:'rgba(255,255,255,0.8)' }}
            >
              AI Explain
            </button>
          </div>
        </div>
      </div>

      {/* ── Decision Row ──────────────────────────────────── */}
      <div className="decision-row">
        <div className="decision-action-card">
          <div className="decision-label">Dealer Recommendation</div>
          <div className={`action-badge ${actionInfo.cls}`} style={{ fontSize:16, padding:'10px 20px' }}>
            {actionInfo.label}
          </div>
          <div style={{ fontSize:11, color:'var(--text-3)' }}>
            {action === 'BUY' ? 'High value deal' :
             action === 'NEGOTIATE' ? 'Negotiate price down' :
             action === 'REJECT' ? 'Not profitable' : 'Review manually'}
          </div>
        </div>

        <div className="decision-action-card">
          <div className="decision-label">Risk Profile</div>
          <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:4 }}>
            <div
              style={{
                width:60, height:60, borderRadius:'50%',
                background: riskScore <= 35 ? '#f0fdf4' : riskScore <= 65 ? '#fffbeb' : '#fef2f2',
                border: `3px solid ${riskScore <= 35 ? '#16a34a' : riskScore <= 65 ? '#d97706' : '#dc2626'}`,
                display:'flex', alignItems:'center', justifyContent:'center',
                fontSize:20, fontWeight:800,
                color: riskScore <= 35 ? '#16a34a' : riskScore <= 65 ? '#d97706' : '#dc2626',
              }}
            >
              {riskScore}
            </div>
            <span className={`badge badge-${riskScore<=35?'success':riskScore<=65?'warning':'danger'}`}>
              {getRiskLabel(riskScore)} Risk
            </span>
          </div>
        </div>

        <div className="decision-action-card" style={{ display: window.innerWidth >= 768 ? 'flex' : 'none' }}>
          <div className="decision-label">Net Dealer Profit</div>
          <div style={{ fontSize:28, fontWeight:800, color: profit > 0 ? '#16a34a' : '#dc2626', letterSpacing:'-0.8px' }}>
            {fmtL(profit)}
          </div>
          <div style={{ fontSize:11, color:'var(--text-3)' }}>
            {marginPct}% margin · target achieved
          </div>
        </div>
      </div>

      {/* ── ROI Summary Grid ──────────────────────────────── */}
      <div className="roi-grid" style={{ marginBottom:16 }}>
        <div className="roi-item">
          <div className="roi-item-label">Buy Price</div>
          <div className="roi-item-value">{fmtL(buyPrice)}</div>
        </div>
        <div className="roi-item">
          <div className="roi-item-label">Sell Price</div>
          <div className="roi-item-value">{fmtL(sellPrice)}</div>
        </div>
        <div className="roi-item">
          <div className="roi-item-label">Net Profit</div>
          <div className={`roi-item-value ${profit > 0 ? 'green' : 'red'}`}>{fmtL(profit)}</div>
        </div>
      </div>

      {/* ── Acquisition Strategy ─────────────────────────── */}
      <div className="acq-timeline card">
        <div className="card-label" style={{ marginBottom:0 }}>Negotiation Strategy</div>
        <div style={{ fontSize:11, color:'var(--text-3)', marginBottom:16 }}>
          Three-point offer framework for negotiating acquisition price
        </div>
        <div className="acq-timeline-points">
          <div className="acq-timeline-line" />
          <div className="acq-point">
            <div className="acq-dot opening" />
            <div className="acq-point-label">Opening</div>
            <div className="acq-point-price">{fmtL(opening)}</div>
          </div>
          <div className="acq-point">
            <div className="acq-dot ideal" />
            <div className="acq-point-label">Ideal</div>
            <div className="acq-point-price">{fmtL(buyPrice)}</div>
          </div>
          <div className="acq-point">
            <div className="acq-dot walkaway" />
            <div className="acq-point-label">Walk Away</div>
            <div className="acq-point-price">{fmtL(walkAway)}</div>
          </div>
        </div>
      </div>

      {/* ── Cost Waterfall ────────────────────────────────── */}
      <div className="card">
        <div className="card-header">
          <div>
            <div style={{ fontSize:14, fontWeight:700, color:'var(--text-1)' }}>Cost Waterfall</div>
            <div style={{ fontSize:11, color:'var(--text-3)' }}>How dealer margin is calculated</div>
          </div>
        </div>
         <div className="waterfall">
          {valuationResult.waterfall && valuationResult.waterfall.length > 0 ? (
            valuationResult.waterfall.map((item, idx) => {
              if (idx === valuationResult.waterfall.length - 1) return null; // Skip recommended buy price row at end of list
              const isDeduct = item.sign === '-';
              return (
                <WaterfallItem
                  key={idx}
                  label={item.label}
                  value={item.value}
                  pct={idx === 0 ? 100 : wfPct(item.value)}
                  color={idx === 0 ? 'blue' : isDeduct ? 'red' : 'gray'}
                  deduct={isDeduct}
                />
              );
            })
          ) : (
            <>
              <WaterfallItem label="ML Market Value"  value={marketValueNum} pct={100}           color="blue"  />
              {repairBuffer > 0 && <WaterfallItem label="Recon / Repair"   value={repairBuffer}  pct={wfPct(repairBuffer)} color="red"   deduct />}
              {riskBuffer > 0   && <WaterfallItem label="Risk Buffer"      value={riskBuffer}    pct={wfPct(riskBuffer)}   color="red"   deduct />}
              {holdingCost > 0  && <WaterfallItem label="Holding Cost"     value={holdingCost}   pct={wfPct(holdingCost)}  color="orange" deduct />}
              <WaterfallItem label="Target Margin"     value={profit}        pct={wfPct(profit)}   color="gray"  deduct />
            </>
          )}
        </div>
        <div
          style={{
            display:'flex', justifyContent:'space-between', alignItems:'center',
            borderTop:'2px solid var(--border)', paddingTop:14, marginTop:8,
          }}
        >
          <div style={{ fontSize:14, fontWeight:700, color:'var(--text-1)' }}>Recommended Buy Price</div>
          <div style={{ fontSize:22, fontWeight:800, color:'var(--accent)', letterSpacing:'-0.5px' }}>
            {fmtL(buyPrice)}
          </div>
        </div>
      </div>

      {/* ── Risk Grid ─────────────────────────────────────── */}
      <div className="card">
        <div className="card-header">
          <div style={{ fontSize:14, fontWeight:700, color:'var(--text-1)' }}>Risk Assessment</div>
          <span className={`badge badge-${riskScore<=35?'success':riskScore<=65?'warning':'danger'}`}>
            {getRiskLabel(riskScore)} · {riskScore}/100
          </span>
        </div>
        <div className="risk-grid">
          {riskItems.map(item => (
            <RiskItem key={item.label} {...item} />
          ))}
        </div>
      </div>

      {/* ── Key Factors ───────────────────────────────────── */}
      {(positiveFactors.length > 0 || negativeFactors.length > 0) && (
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12 }}>
          {positiveFactors.length > 0 && (
            <div className="card card-success">
              <div style={{ fontSize:12, fontWeight:700, color:'#16a34a', textTransform:'uppercase', letterSpacing:'0.4px', marginBottom:10 }}>
                ✓ Positive Factors
              </div>
              <div className="factor-list">
                {positiveFactors.slice(0, 4).map((f, i) => (
                  <div key={i} className="factor-item positive">
                    <div className="factor-dot" />
                    {f}
                  </div>
                ))}
              </div>
            </div>
          )}
          {negativeFactors.length > 0 && (
            <div className="card card-danger">
              <div style={{ fontSize:12, fontWeight:700, color:'#dc2626', textTransform:'uppercase', letterSpacing:'0.4px', marginBottom:10 }}>
                ✗ Risk Factors
              </div>
              <div className="factor-list">
                {negativeFactors.slice(0, 4).map((f, i) => (
                  <div key={i} className="factor-item negative">
                    <div className="factor-dot" />
                    {f}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Warnings ──────────────────────────────────────── */}
      {warnings.length > 0 && (
        <div className="card" style={{ borderLeft:'3px solid var(--warning)', background:'var(--warning-light)' }}>
          <div style={{ fontSize:12, fontWeight:700, color:'var(--warning)', textTransform:'uppercase', letterSpacing:'0.4px', marginBottom:10 }}>
            ⚠ Warnings
          </div>
          {warnings.map((w, i) => (
            <div key={i} style={{ fontSize:13, color:'#92400e', marginBottom:6, display:'flex', alignItems:'flex-start', gap:8 }}>
              <span style={{ marginTop:2 }}>•</span> {w}
            </div>
          ))}
        </div>
      )}

      {/* ── CTA row ───────────────────────────────────────── */}
      <div style={{ display:'flex', gap:10, marginTop:8, flexWrap:'wrap' }}>
        <button className="btn btn-primary" onClick={() => setActiveScreen('pricing')}>
          <Icon name="coins" size={15} color="white" strokeWidth={2} />
          Full Pricing Breakdown
        </button>
        <button className="btn btn-secondary" onClick={() => setActiveScreen('explain')}>
          <Icon name="brain" size={15} color="#475569" strokeWidth={2} />
          AI Explanation
        </button>
        <button className="btn btn-secondary" onClick={() => setActiveScreen('input')}>
          <Icon name="refresh" size={15} color="#475569" strokeWidth={2} />
          New Valuation
        </button>
      </div>
    </div>
  );
}
