import { useApp } from '../context/AppContext.jsx';
import Icon from '../components/Icon.jsx';

const fmtL = (n) => {
  const v = Number(n || 0);
  if (v >= 10000000) return `₹${(v/10000000).toFixed(2)}Cr`;
  if (v >= 100000)   return `₹${(v/100000).toFixed(2)}L`;
  if (v >= 1000)     return `₹${Math.round(v/1000)}K`;
  return `₹${Math.round(v).toLocaleString()}`;
};

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
  const { valuationResult, inputs, setActiveScreen } = useApp();

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
    predictedPrice = 0,
    priceMin = 0,
    priceMax = 0,
    recommendedBuyPrice,
    recommendedSellPrice,
    expectedProfit = 0,
    expectedMarginPct = 0,
    recon_cost = 18000,
    holding_cost = 5000,
    doc_cost = 4500,
    risk_buffer = 3000,
    target_profit = 35000,
    action,
  } = valuationResult;

  const finalBuyPrice  = recommendedBuyPrice || 0;
  
  const rawFinalSell   = recommendedSellPrice || 0;
  const finalSellPrice = rawFinalSell > finalBuyPrice ? rawFinalSell : Math.round(finalBuyPrice * 1.10 / 500) * 500;
  const finalProfit    = finalSellPrice > finalBuyPrice
    ? expectedProfit || Math.round(finalSellPrice - finalBuyPrice - recon_cost - holding_cost - doc_cost)
    : expectedProfit;
  const finalROI       = expectedMarginPct;

  const totalOperatingCosts = recon_cost + holding_cost + doc_cost + risk_buffer;

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

      {}
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
        {}
        <div className="card">
          <div style={{ fontSize:14, fontWeight:700, color:'var(--text-1)', marginBottom:4 }}>
            Acquisition Cost Breakdown
          </div>
          <div style={{ fontSize:11, color:'var(--text-3)', marginBottom:16 }}>
            Cost factors deducted from market value to determine buy offer
          </div>

          {}
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

          {}
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
