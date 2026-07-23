import { useMemo } from 'react';
import { useApp } from '../context/AppContext.jsx';
import Icon from '../components/Icon.jsx';

const fmtL = (n) => {
  const v = Number(n || 0);
  if (v >= 10000000) return `₹${(v/10000000).toFixed(2)}Cr`;
  if (v >= 100000)   return `₹${(v/100000).toFixed(2)}L`;
  if (v >= 1000)     return `₹${Math.round(v/1000)}K`;
  return `₹${Math.round(v).toLocaleString()}`;
};

/* ─── Realistic Dealer Cost Model ─────────────────────────────── */
function computeRealisticCosts(marketValue, inputs = {}) {
  const mv = Number(marketValue || 0);
  const mileage = Number(inputs.mileage || 0);
  const condition = inputs.condition || 'Good';
  const ownerCount = Number(inputs.ownerCount || 1);
  const repairBuffer = Number(inputs.repairBuffer || 0);

  // Base recon from condition
  const reconMap = { Excellent: 8000, Good: 18000, Average: 35000, Poor: 65000 };
  const recon = repairBuffer > 0 ? repairBuffer : (reconMap[condition] || 18000);

  // Detailing + cleaning
  const detailing = mv > 2000000 ? 8000 : mv > 800000 ? 5500 : 3500;

  // RC Transfer
  const rcTransfer = 3500;

  // Holding cost: 2.5% p.a. × (45 days / 365)
  const holdingCost = Math.round(mv * 0.025 * (45 / 365));

  // Interest on acquisition: 9% p.a. × 45 days
  // (buy price ≈ 82% of market value)
  const buyApprox = mv * 0.82;
  const interestCost = Math.round(buyApprox * 0.09 * (45 / 365));

  // Insurance gap (1 month)
  const insurance = Math.round(mv * 0.01 * (1/12));

  // Sales commission (1.5% of sell price ≈ 1.5% × 1.08 × mv)
  const salesCommission = Math.round(mv * 1.08 * 0.015);

  // Negotiation buffer (2% of buy price)
  const negotiationBuffer = Math.round(buyApprox * 0.02);

  // Unexpected repairs (based on mileage + age)
  const unexpectedRepair = mileage > 80000 ? 15000 : mileage > 50000 ? 10000 : 7000;

  // Old-owner premium markup (if multi-owner, harder to sell)
  const ownerPenalty = ownerCount > 2 ? 5000 : 0;

  const totalCosts = recon + detailing + rcTransfer + holdingCost + interestCost + insurance + salesCommission + negotiationBuffer + unexpectedRepair + ownerPenalty;

  const sellPrice   = Math.round(mv * 1.06); // retail listing at 6% premium
  const idealBuy    = Math.round(sellPrice - totalCosts - (mv * 0.07)); // target ₹30-70k profit
  const netProfit   = sellPrice - idealBuy - totalCosts;
  const roiPct      = idealBuy > 0 ? ((netProfit / idealBuy) * 100).toFixed(1) : '0';

  // Warn if margin is too high (buy price can be increased to win deal)
  const isProfitHealthy = netProfit >= 20000 && netProfit <= 120000;
  const profitAlert = netProfit > 120000
    ? `Margin ₹${Math.round(netProfit/1000)}K is very high — consider offering ₹${fmtL(netProfit - 80000)} more to win the deal.`
    : netProfit < 20000
    ? 'Margin is very thin. Negotiate harder or reject this deal.'
    : null;

  return {
    recon, detailing, rcTransfer, holdingCost, interestCost, insurance,
    salesCommission, negotiationBuffer, unexpectedRepair, ownerPenalty,
    totalCosts, sellPrice, idealBuy, netProfit, roiPct,
    isProfitHealthy, profitAlert,
    openingOffer: Math.round(idealBuy * 0.96),
    walkAway: Math.round(idealBuy * 1.03),
  };
}

function CostRow({ icon, label, amount, isDeduct = true, highlight = false }) {
  return (
    <div className="cost-row">
      <div className="cost-row-label">
        <div className="cost-row-icon">
          <Icon name={icon} size={13} color="#94a3b8" strokeWidth={1.8} />
        </div>
        {label}
      </div>
      <div
        className={`cost-row-amount ${isDeduct ? 'negative' : ''}`}
        style={ highlight ? { color:'var(--accent)', fontSize:16 } : {} }
      >
        {isDeduct ? '−' : ''}{fmtL(Math.abs(amount))}
      </div>
    </div>
  );
}

function FeatureBar({ feature, value, contribution, label, predictedPrice }) {
  const positive = contribution >= 0;
  // Calculate percentage bar width relative to predicted price, min 3% max 100%
  const pct = Math.min(100, Math.max(3, (Math.abs(contribution) / (predictedPrice || 1000000)) * 100));

  return (
    <div className="shap-item" style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 0', borderBottom: '1px solid var(--border-light)' }}>
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

export default function PricingScreen() {
  const { valuationResult, inputs, setActiveScreen, evaluations } = useApp();

  if (!valuationResult) {
    return (
      <div className="screen">
        <div className="empty-screen">
          <div className="home-empty-icon">
            <Icon name="coins" size={28} color="#94a3b8" strokeWidth={1.8} />
          </div>
          <div className="empty-title">No pricing data yet</div>
          <div className="empty-sub">
            Run a valuation first to see the full dealer cost breakdown and realistic profit analysis.
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
    priceMin,
    priceMax,
    recommendedBuyPrice,
    recommendedSellPrice,
    expectedProfit = 0,
    expectedMarginPct = 0,
    recon_cost = 18000,
    holding_cost = 5000,
    doc_cost = 4500,
    risk_buffer = 3000,
    target_profit = 35000,
    opening_offer,
    max_offer,
    target_offer,
    quoteMessage,
    action,
    similarCars = [],
    positiveFactors = [],
    negativeFactors = [],
    shap = [],
  } = valuationResult;

  const finalBuyPrice  = recommendedBuyPrice || 0;
  // Sell price must always be above buy price — guard against backend inversion
  const rawFinalSell   = recommendedSellPrice || 0;
  const finalSellPrice = rawFinalSell > finalBuyPrice ? rawFinalSell : Math.round(finalBuyPrice * 1.10 / 500) * 500;
  const finalProfit    = finalSellPrice > finalBuyPrice
    ? expectedProfit || Math.round(finalSellPrice - finalBuyPrice - recon_cost - holding_cost - doc_cost)
    : expectedProfit;
  const finalROI       = expectedMarginPct;

  const totalOperatingCosts = recon_cost + holding_cost + doc_cost + risk_buffer;

  // Only show similar cars from the real dataset — never use evaluations history as fallback
  const comparables = (similarCars || []).filter(c => c && (c.source === 'dataset' || c.market_value > 0));


  const actionLabel = String(action||'').toUpperCase();
  const profitColor = finalProfit > 50000 ? '#16a34a' : finalProfit > 25000 ? '#d97706' : '#dc2626';

  return (
    <div className="screen">
      <div className="page-header">
        <div>
          <div className="page-title">Deal Financials</div>
          <div className="page-subtitle">
            {inputs.year} {inputs.brand} {inputs.model} · Acquisition cost view
          </div>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={() => setActiveScreen('result')}>
          ← Result
        </button>
      </div>


      {/* Headline ROI cards */}
      <div className="kpi-grid" style={{ marginBottom:16 }}>
        <div className="kpi-tile">
          <div className="kpi-tile-header">
            <div className="kpi-tile-label">ML Market Value</div>
            <div className="kpi-icon" style={{ background:'#dbeafe' }}>
              <Icon name="trendUp" size={14} color="#2563eb" strokeWidth={2} />
            </div>
          </div>
          <div className="kpi-tile-value">{fmtL(predictedPrice)}</div>
          <div className="kpi-tile-sub">{fmtL(priceMin)} – {fmtL(priceMax)}</div>
        </div>
        <div className="kpi-tile">
          <div className="kpi-tile-header">
            <div className="kpi-tile-label">Recommended Buy Price</div>
            <div className="kpi-icon" style={{ background:'#fff4f0' }}>
              <Icon name="car" size={14} color="#f75d34" strokeWidth={2} />
            </div>
          </div>
          <div className="kpi-tile-value" style={{ color:'var(--accent)' }}>{fmtL(finalBuyPrice)}</div>
          <div className="kpi-tile-sub">After all costs</div>
        </div>
        <div className="kpi-tile">
          <div className="kpi-tile-header">
            <div className="kpi-tile-label">Target Retail Price</div>
            <div className="kpi-icon" style={{ background:'#dcfce7' }}>
              <Icon name="coins" size={14} color="#16a34a" strokeWidth={2} />
            </div>
          </div>
          <div className="kpi-tile-value">{fmtL(finalSellPrice)}</div>
          <div className="kpi-tile-sub">Retail listing target</div>
        </div>
        <div className="kpi-tile">
          <div className="kpi-tile-header">
            <div className="kpi-tile-label">Net Dealer Profit</div>
            <div className="kpi-icon" style={{ background: finalProfit > 30000 ? '#dcfce7' : '#fef2f2' }}>
              <Icon name="lightning" size={14} color={profitColor} strokeWidth={2} />
            </div>
          </div>
          <div className="kpi-tile-value" style={{ color: profitColor }}>{fmtL(finalProfit)}</div>
          <div className="kpi-tile-sub">{finalROI}% ROI · {actionLabel}</div>
        </div>
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'1fr', gap:16 }}>
        {/* Full Cost Breakdown */}
        <div className="card">
          <div style={{ fontSize:14, fontWeight:700, color:'var(--text-1)', marginBottom:4 }}>
            Acquisition Cost Breakdown
          </div>
          <div style={{ fontSize:11, color:'var(--text-3)', marginBottom:16 }}>
            Cost factors deducted from market value to determine buy offer
          </div>

          {/* Start with market value */}
          <div className="cost-row" style={{ paddingTop:0 }}>
            <div className="cost-row-label">
              <div className="cost-row-icon">
                <Icon name="trendUp" size={13} color="#2563eb" strokeWidth={1.8} />
              </div>
              <strong>ML Market Value</strong>
            </div>
            <div className="cost-row-amount" style={{ color:'var(--info)', fontSize:15 }}>
              {fmtL(predictedPrice)}
            </div>
          </div>

          <div style={{ height:1, background:'var(--border)', margin:'6px 0' }} />

          <CostRow icon="tool"    label="Reconditioning & Repairs"        amount={recon_cost} />
          <CostRow icon="clock"   label="Holding Cost (30 days)"          amount={holding_cost} />
          <CostRow icon="document" label="RC Transfer & Documentation"    amount={doc_cost} />
          <CostRow icon="shield"  label="Risk & Repair Buffer"            amount={risk_buffer} />
          <CostRow icon="coins"   label="Target Dealer Margin"            amount={target_profit} />

          <div style={{ height:2, background:'var(--border)', margin:'8px 0' }} />

          <div className="cost-total-row">
            <div className="cost-total-label">Total Deductions</div>
            <div className="cost-total-amount" style={{ color:'var(--danger)' }}>
              {fmtL(totalOperatingCosts + target_profit)}
            </div>
          </div>

          {/* Profit line */}
          <div style={{ background: finalProfit > 20000 ? 'var(--success-light)' : 'var(--danger-light)', borderRadius:'var(--r-md)', padding:'14px 16px', marginTop:12, display:'flex', justifyContent:'space-between', alignItems:'center' }}>
            <div>
              <div style={{ fontSize:11, fontWeight:700, color: profitColor, textTransform:'uppercase', letterSpacing:'0.4px', marginBottom:3 }}>
                Expected Net Profit
              </div>
              <div style={{ fontSize:12, color:'var(--text-3)' }}>
                Sell {fmtL(finalSellPrice)} − Buy {fmtL(finalBuyPrice)} − Recon {fmtL(recon_cost)} − Holding {fmtL(holding_cost)}
              </div>
            </div>
            <div style={{ textAlign:'right' }}>
              <div style={{ fontSize:24, fontWeight:800, color: profitColor, letterSpacing:'-0.5px' }}>
                {fmtL(finalProfit)}
              </div>
              <div style={{ fontSize:12, color:'var(--text-3)' }}>{finalROI}% ROI</div>
            </div>
          </div>
        </div>


      </div>

      <div style={{ display:'flex', gap:10, marginTop:16 }}>
        <button className="btn btn-secondary" onClick={() => setActiveScreen('result')}>
          ← Back to Result
        </button>
        <button className="btn btn-secondary" onClick={() => setActiveScreen('input')}>
          New Valuation
        </button>
      </div>
    </div>
  );
}
