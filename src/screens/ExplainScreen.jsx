import { useApp } from '../context/AppContext.jsx';
import Icon from '../components/Icon.jsx';

const fmtL = (n) => {
  const v = Number(n || 0);
  if (v >= 100000) return `₹${(v/100000).toFixed(2)}L`;
  return `₹${Math.round(v).toLocaleString()}`;
};

// Segment color per segment class
const SEGMENT_COLOR = {
  economy: '#2563eb',
  premium: '#d97706',
  luxury:  '#7c3aed',
};

function FeatureBar({ feature, value, contribution, label, predictedPrice }) {
  const positive = contribution >= 0;
  // Calculate percentage bar width relative to predicted price, min 3% max 100%
  const pct = Math.min(100, Math.max(3, (Math.abs(contribution) / (predictedPrice || 1000000)) * 100));

  return (
    <div className="shap-item" style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 0', borderBottom: '1px solid var(--border-light)', animationDelay: `${Math.random() * 0.2}s` }}>
      <div style={{ flex: '1 1 200px', minWidth: 160 }}>
        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-1)' }}>{feature}</div>
        <div style={{ fontSize: '12px', color: 'var(--text-2)', marginTop: 2 }}>{label}</div>
      </div>
      <div style={{ flex: '0 0 80px', fontSize: '12.5px', color: 'var(--text-3)', fontWeight: 500, textAlign: 'right' }}>
        {value}
      </div>
      <div className="shap-bar-area" style={{ flex: '1 1 120px', minWidth: 80, display: 'flex', alignItems: 'center' }}>
        <div
          className={`shap-bar ${positive ? 'pos' : 'neg'}`}
          style={{ width: `${pct}%`, height: 8, borderRadius: 4, transition: 'width 0.8s ease' }}
        />
      </div>
      <span className={`shap-bar-label ${positive ? 'pos' : 'neg'}`} style={{ flex: '0 0 85px', textAlign: 'right', fontWeight: 700, fontSize: '13px' }}>
        {positive ? '+' : '−'}{fmtL(Math.abs(contribution))}
      </span>
    </div>
  );
}

function ConfGauge({ score }) {
  // Draw a simple half-circle arc gauge
  const r = 44;
  const cx = 50, cy = 50;
  const startAngle = Math.PI;
  const endAngle   = Math.PI * 2;
  const sweepAngle = ((score/100) * Math.PI);
  const x1 = cx + r * Math.cos(startAngle);
  const y1 = cy + r * Math.sin(startAngle);
  const x2 = cx + r * Math.cos(startAngle + sweepAngle);
  const y2 = cy + r * Math.sin(startAngle + sweepAngle);
  const color = score >= 75 ? '#16a34a' : score >= 50 ? '#d97706' : '#dc2626';
  return (
    <div className="gauge-wrap">
      <svg width={100} height={58} viewBox="0 0 100 58">
        <path d={`M ${cx-r} ${cy} A ${r} ${r} 0 0 1 ${cx+r} ${cy}`}
              fill="none" stroke="var(--surface-3)" strokeWidth={8} strokeLinecap="round" />
        {score > 0 && (
          <path d={`M ${x1} ${y1} A ${r} ${r} 0 ${sweepAngle > Math.PI/2 ? 1 : 0} 1 ${x2} ${y2}`}
                fill="none" stroke={color} strokeWidth={8} strokeLinecap="round" />
        )}
        <text x={50} y={50} textAnchor="middle" fontSize={16} fontWeight={800} fill={color}>
          {score}%
        </text>
      </svg>
      <div style={{ fontSize:11, fontWeight:600, color:'var(--text-3)', marginTop:-4 }}>
        Confidence Score
      </div>
    </div>
  );
}

export default function ExplainScreen() {
  const { valuationResult, inputs, setActiveScreen } = useApp();

  if (!valuationResult) {
    return (
      <div className="screen">
        <div className="empty-screen">
          <div className="home-empty-icon">
            <Icon name="brain" size={28} color="#94a3b8" strokeWidth={1.8} />
          </div>
          <div className="empty-title">No valuation to explain</div>
          <div className="empty-sub">
            Run a valuation first. AI Explain will show you exactly which factors
            drove the ML price prediction.
          </div>
          <button className="btn btn-primary btn-lg" onClick={() => setActiveScreen('input')}>
            <Icon name="car" size={16} color="white" strokeWidth={2} />
            Start Valuation
          </button>
        </div>
      </div>
    );
  }

  const {
    predictedPrice,
    confidenceScore = 80,
    segmentClass   = 'economy',
    positiveFactors = [],
    negativeFactors = [],
    routingNote    = '',
    shap = [],
  } = valuationResult;

  const mileageNum = Number(inputs.mileage || 0);
  const ageYears   = new Date().getFullYear() - Number(inputs.year || 2020);

  // Use dynamic shap list from backend payload
  const shapFeatures = shap;

  // Summary sentence
  const summaryParts = [];
  if (ageYears <= 3) summaryParts.push('relatively new vehicle');
  if (mileageNum < 40000) summaryParts.push('low odometer reading');
  if (inputs.ownerCount === '1') summaryParts.push('first-owner');
  if (inputs.condition === 'Excellent') summaryParts.push('excellent condition');
  if (inputs.transmission === 'Automatic') summaryParts.push('automatic transmission premium');

  const summary = summaryParts.length > 0
    ? `This ${inputs.brand} ${inputs.model} benefits from ${summaryParts.slice(0,-1).join(', ')}${summaryParts.length > 1 ? ' and ' : ''}${summaryParts.slice(-1)[0]}, contributing to its ML-estimated market value of ${fmtL(predictedPrice)}.`
    : `Based on the vehicle inputs, the model estimated a market value of ${fmtL(predictedPrice)} using the ${segmentClass} segment model with ${confidenceScore}% confidence.`;

  return (
    <div className="screen">
      <div className="page-header">
        <div>
          <div className="page-title">AI Explanation</div>
          <div className="page-subtitle">
            {inputs.year} {inputs.brand} {inputs.model} · {(segmentClass||'economy').toUpperCase()} segment
          </div>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={() => setActiveScreen('result')}>
          ← Result
        </button>
      </div>

      {/* Confidence + segment */}
      <div style={{ display:'flex', gap:12, marginBottom:20, flexWrap:'wrap' }}>
        <div className="card" style={{ flex:'1', minWidth:180 }}>
          <ConfGauge score={confidenceScore} />
        </div>
        <div className="card" style={{ flex:'2', minWidth:220 }}>
          <div className="card-label" style={{ marginBottom:10 }}>Model Routing</div>
          <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:12 }}>
            <span
              className="segment-badge"
              style={{
                background: SEGMENT_COLOR[segmentClass] + '20',
                color: SEGMENT_COLOR[segmentClass],
                border: `1px solid ${SEGMENT_COLOR[segmentClass]}40`,
                padding:'5px 14px', fontSize:12,
              }}
            >
              {(segmentClass||'economy').toUpperCase()} MODEL
            </span>
          </div>
          {routingNote && (
            <div style={{ fontSize:12.5, color:'var(--text-2)', lineHeight:1.6 }}>
              {routingNote}
            </div>
          )}
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:8, marginTop:12 }}>
            <div>
              <div className="label-xs">ML Estimate</div>
              <div style={{ fontSize:18, fontWeight:800, color:'var(--text-1)', marginTop:3 }}>
                {fmtL(predictedPrice)}
              </div>
            </div>
            <div>
              <div className="label-xs">Confidence</div>
              <div style={{ fontSize:18, fontWeight:800, color: confidenceScore >= 75 ? '#16a34a' : '#d97706', marginTop:3 }}>
                {confidenceScore}%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* SHAP-style Feature Contributions */}
      <div className="card">
        <div className="shap-header">
          <div>
            <div style={{ fontSize:14, fontWeight:700, color:'var(--text-1)' }}>Feature Impact Analysis</div>
            <div style={{ fontSize:11, color:'var(--text-3)' }}>
              How each input factor influenced the ML prediction
            </div>
          </div>
          <div className="shap-legend-row">
            <div className="shap-legend-item">
              <div className="shap-dot-g" />Positive
            </div>
            <div className="shap-legend-item">
              <div className="shap-dot-r" />Negative
            </div>
          </div>
        </div>
        <div className="shap-list">
          {shapFeatures.map((f, i) => (
            <FeatureBar key={i} {...f} predictedPrice={predictedPrice} />
          ))}
        </div>
      </div>

      {/* AI Summary */}
      <div className="card" style={{ borderLeft:'3px solid var(--info)', background:'var(--info-light)' }}>
        <div style={{ display:'flex', gap:14, alignItems:'flex-start' }}>
          <div style={{ width:36, height:36, background:'var(--info-bg)', borderRadius:'var(--r-md)', display:'flex', alignItems:'center', justifyContent:'center', flexShrink:0 }}>
            <Icon name="brain" size={18} color={`var(--info)`} strokeWidth={2} />
          </div>
          <div>
            <div style={{ fontSize:12, fontWeight:700, color:'var(--info)', textTransform:'uppercase', letterSpacing:'0.4px', marginBottom:6 }}>
              ML Model Summary
            </div>
            <div style={{ fontSize:13.5, color:'var(--text-2)', lineHeight:1.65 }}>
              {summary}
            </div>
          </div>
        </div>
      </div>

      {/* Factor columns */}
      {(positiveFactors.length > 0 || negativeFactors.length > 0) && (
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12 }}>
          {positiveFactors.length > 0 && (
            <div className="card">
              <div style={{ fontSize:12, fontWeight:700, color:'#16a34a', marginBottom:10 }}>✓ Value Drivers</div>
              <div className="cf-list">
                {positiveFactors.map((f, i) => (
                  <div key={i} className="cf-item cf-pos">
                    <div className="cf-item-icon-wrap">
                      <Icon name="check" size={13} color="#16a34a" strokeWidth={2.5} />
                    </div>
                    <div className="cf-item-text">{f}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {negativeFactors.length > 0 && (
            <div className="card">
              <div style={{ fontSize:12, fontWeight:700, color:'#dc2626', marginBottom:10 }}>✗ Value Detractors</div>
              <div className="cf-list">
                {negativeFactors.map((f, i) => (
                  <div key={i} className="cf-item cf-neg">
                    <div className="cf-item-icon-wrap">
                      <Icon name="warning" size={13} color="#dc2626" strokeWidth={2.5} />
                    </div>
                    <div className="cf-item-text">{f}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      <div style={{ display:'flex', gap:10, marginTop:8 }}>
        <button className="btn btn-primary" onClick={() => setActiveScreen('pricing')}>
          <Icon name="coins" size={15} color="white" strokeWidth={2} />
          Full Pricing Breakdown
        </button>
        <button className="btn btn-secondary" onClick={() => setActiveScreen('result')}>
          ← Back to Result
        </button>
      </div>
    </div>
  );
}
