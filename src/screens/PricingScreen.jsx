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
    similarCars = []
  } = valuationResult;

  const finalBuyPrice  = recommendedBuyPrice;
  const finalSellPrice = recommendedSellPrice;
  const finalProfit    = expectedProfit;
  const finalROI       = expectedMarginPct;

  const totalOperatingCosts = recon_cost + holding_cost + doc_cost + risk_buffer;

  // Use similarCars from backend. If empty, fallback to local evaluations history
  const comparables = similarCars.length > 0
    ? similarCars
    : evaluations
        .filter(t => t.brand === inputs.brand &&
          !(t.year === Number(inputs.year) && t.model === inputs.model && t.marketValue === predictedPrice))
        .slice(0, 5);

  const actionLabel = String(action||'').toUpperCase();
  const profitColor = finalProfit > 50000 ? '#16a34a' : finalProfit > 25000 ? '#d97706' : '#dc2626';

  return (
    <div className="screen">
      <div className="page-header">
        <div>
          <div className="page-title">Pricing Intelligence</div>
          <div className="page-subtitle">
            {inputs.year} {inputs.brand} {inputs.model} · Dealer acquisition view
          </div>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={() => setActiveScreen('result')}>
          ← Result
        </button>
      </div>

      {/* Dealer role notice */}
      <div className="role-notice">
        <Icon name="store" size={14} color="#f75d34" strokeWidth={1.8} />
        Realistic dealer cost model · Target margins dynamically calibrated
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

          <CostRow icon="tool"    label="Expected Reconditioning Cost"   amount={recon_cost} />
          <CostRow icon="clock"   label="Expected Holding Cost (30 days)" amount={holding_cost} />
          <CostRow icon="document" label="RC Transfer & Documentation"   amount={doc_cost} />
          <CostRow icon="shield"  label="Dynamic Risk Buffer"            amount={risk_buffer} />
          <CostRow icon="coins"   label="Segment-capped Dealer Profit"   amount={target_profit} />

          <div style={{ height:2, background:'var(--border)', margin:'8px 0' }} />

          <div className="cost-total-row">
            <div className="cost-total-label">Total Cost & Profit Deductions</div>
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

        {/* Negotiation Strategy */}
        <div className="card">
          <div style={{ fontSize:14, fontWeight:700, color:'var(--text-1)', marginBottom:4 }}>Negotiation Strategy</div>
          <div style={{ fontSize:11, color:'var(--text-3)', marginBottom:0 }}>
            Three-point offer framework based on city demand & condition
          </div>
          <div className="negotiation-trio">
            <div className="neg-point opening">
              <div className="neg-point-label">Opening Offer</div>
              <div className="neg-point-price">{fmtL(opening_offer)}</div>
              <div style={{ fontSize:10, color:'var(--text-3)', marginTop:4 }}>Start negotiation</div>
            </div>
            <div className="neg-point ideal">
              <div className="neg-point-label">Ideal Offer</div>
              <div className="neg-point-price">{fmtL(target_offer || finalBuyPrice)}</div>
              <div style={{ fontSize:10, color:'var(--text-3)', marginTop:4 }}>Target buy outcome</div>
            </div>
            <div className="neg-point walkaway">
              <div className="neg-point-label">Maximum Offer</div>
              <div className="neg-point-price">{fmtL(max_offer)}</div>
              <div style={{ fontSize:10, color:'var(--text-3)', marginTop:4 }}>Walk away limit</div>
            </div>
          </div>
        </div>

        {/* Seller script */}
        {quoteMessage && (
          <div className="seller-script-card">
            <div className="seller-script-label">Seller Script</div>
            <div className="seller-script-text">"{quoteMessage}"</div>
          </div>
        )}

        {/* Confidence interval */}
        <div className="card">
          <div style={{ fontSize:14, fontWeight:700, color:'var(--text-1)', marginBottom:14 }}>Price Confidence Band</div>
          <div className="ci-row">
            <div className="ci-box low">
              <div className="ci-box-label">Floor</div>
              <div className="ci-box-val">{fmtL(priceMin)}</div>
            </div>
            <Icon name="arrowRight" size={16} color="#cbd5e1" strokeWidth={2} />
            <div className="ci-box mid">
              <div className="ci-box-label">ML Estimate</div>
              <div className="ci-box-val orange">{fmtL(predictedPrice)}</div>
            </div>
            <Icon name="arrowRight" size={16} color="#cbd5e1" strokeWidth={2} />
            <div className="ci-box high">
              <div className="ci-box-label">Ceiling</div>
              <div className="ci-box-val">{fmtL(priceMax)}</div>
            </div>
          </div>
        </div>

        {/* Comparables */}
        <div className="card">
          <div style={{ fontSize:14, fontWeight:700, color:'var(--text-1)', marginBottom:12 }}>
            Comparable Market Vehicles
          </div>
          {comparables.length === 0 ? (
            <div style={{ textAlign:'center', padding:'20px 0', color:'var(--text-3)', fontSize:13 }}>
              No similar vehicles available for comparables.
            </div>
          ) : (
            <div className="comp-list">
              {comparables.map((tx, idx) => (
                <div key={idx} className="comp-item">
                  <div className="comp-item-info">
                    <div className="comp-item-name">{tx.year} {tx.brand} {tx.model}</div>
                    <div className="comp-item-spec">
                      <Icon name="mapPin" size={10} color="#94a3b8" strokeWidth={2} />
                      {((tx.odometer || tx.kmDriven || 0)/1000).toFixed(0)}k km · {tx.city}
                    </div>
                    <span className="comp-cond cond-good" style={{ background: '#f1f5f9', color: '#475569' }}>
                      {tx.condition || 'Good'}
                    </span>
                  </div>
                  <div className="comp-item-price">{fmtL(tx.market_value || tx.marketValue || tx.predictedPrice)}</div>
                </div>
              ))}
            </div>
          )}
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
