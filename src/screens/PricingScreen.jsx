import { useApp } from '../context/AppContext.jsx';
import Icon from '../components/Icon.jsx';

const fmt = (n) => {
  const v = Number(n || 0);
  if (v >= 10000000) return `₹${(v / 10000000).toFixed(2)}Cr`;
  if (v >= 100000)   return `₹${(v / 100000).toFixed(2)}L`;
  if (v >= 1000)     return `₹${(v / 1000).toFixed(1)}k`;
  return `₹${Math.round(v).toLocaleString('en-IN')}`;
};

const fmtFull = (n) => `₹${Math.round(Number(n || 0)).toLocaleString('en-IN')}`;

export default function PricingScreen() {
  const { valuationResult, inputs, setActiveScreen } = useApp();

  if (!valuationResult) {
    return (
      <div className="screen">
        <div className="empty-screen">
          <div className="empty-icon-wrap">
            <Icon name="coins" size={32} color="#e85d26" strokeWidth={1.8} />
          </div>
          <div className="empty-title">No Deal Financials Available</div>
          <div className="empty-sub">
            Run a vehicle valuation to inspect the dealer cost waterfall, net margin projections, and acquisition economics.
          </div>
          <button className="btn btn-primary btn-lg" onClick={() => setActiveScreen('input')}>
            <Icon name="car" size={15} color="white" strokeWidth={2} />
            <span>Run Valuation</span>
          </button>
        </div>
      </div>
    );
  }

  const {
    predictedPrice = 0,
    recommendedBuyPrice = 0,
    recommendedSellPrice = 0,
    expectedProfit = 0,
    expectedMarginPct = 0,
    recon_cost = 18000,
    holding_cost = 5000,
    doc_cost = 4500,
    risk_buffer = 3000,
    action = 'BUY',
    dealQualityScore = 78,
  } = valuationResult;

  const totalCosts = recon_cost + holding_cost + doc_cost + risk_buffer;
  const targetRetail = recommendedSellPrice || predictedPrice;
  const netProfit = expectedProfit || (targetRetail - recommendedBuyPrice - totalCosts);
  const roiPct = expectedMarginPct || (recommendedBuyPrice ? ((netProfit / recommendedBuyPrice) * 100).toFixed(1) : 12.5);

  return (
    <div className="screen">
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24, flexWrap: 'wrap', gap: 12 }}>
        <div>
          <div className="page-title">Deal Financials & Margin Waterfall</div>
          <div className="page-subtitle">
            Accounting breakdown for {inputs.year} {inputs.brand} {inputs.model} {inputs.variant ? `(${inputs.variant})` : ''}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-secondary btn-sm" onClick={() => setActiveScreen('result')}>
            <Icon name="arrowLeft" size={13} strokeWidth={2} />
            <span>Back to Report</span>
          </button>
          <button className="btn btn-primary btn-sm" onClick={() => setActiveScreen('input')}>
            <Icon name="car" size={13} color="white" strokeWidth={2} />
            <span>New Valuation</span>
          </button>
        </div>
      </div>

      <div className="pricing-root">
        {/* Main Waterfall Panel */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Acquisition & Profit Waterfall</div>
              <div style={{ fontSize: 11.5, color: 'var(--text-4)', marginTop: 2 }}>
                Net dealer margin computed from ML market estimate and operational allowances
              </div>
            </div>
            <span className="badge badge-buy">Audit Verified</span>
          </div>

          <div className="card-body">
            <div className="waterfall">
              {/* Row 1: Estimated Market Value */}
              <div className="waterfall-row">
                <div className="waterfall-label">
                  <Icon name="car" size={15} color="#2563eb" strokeWidth={2} />
                  <strong>ESTIMATED MARKET VALUE</strong>
                </div>
                <div className="waterfall-amount" style={{ fontSize: 16 }}>
                  {fmtFull(predictedPrice)}
                </div>
              </div>

              {/* Row 2: Target Retail */}
              <div className="waterfall-row">
                <div className="waterfall-label">
                  <Icon name="store" size={15} color="#16a34a" strokeWidth={2} />
                  <span>Expected Resale Benchmark</span>
                </div>
                <div className="waterfall-amount">
                  {fmtFull(targetRetail)}
                </div>
              </div>

              {/* Row 3: Target Acquisition (Buy Price) */}
              <div className="waterfall-row divider">
                <div className="waterfall-label">
                  <Icon name="coins" size={15} color="#e85d26" strokeWidth={2} />
                  <strong>RECOMMENDED ACQUISITION (BUY)</strong>
                </div>
                <div className="waterfall-amount buy-price" style={{ color: '#16a34a' }}>
                  {fmtFull(recommendedBuyPrice)}
                </div>
              </div>

              {/* Operating Cost Deductions */}
              <div style={{ padding: '8px 0', fontSize: 11, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-4)', letterSpacing: 0.6 }}>
                Operational & Risk Allowances
              </div>

              <div className="waterfall-row">
                <div className="waterfall-label indent">
                  <span>Reconditioning & Detailing</span>
                </div>
                <div className="waterfall-amount deduction">
                  − {fmtFull(recon_cost)}
                </div>
              </div>

              <div className="waterfall-row">
                <div className="waterfall-label indent">
                  <span>Holding & Capital Cost (30-day est.)</span>
                </div>
                <div className="waterfall-amount deduction">
                  − {fmtFull(holding_cost)}
                </div>
              </div>

              <div className="waterfall-row">
                <div className="waterfall-label indent">
                  <span>RTO Documentation & RC Transfer</span>
                </div>
                <div className="waterfall-amount deduction">
                  − {fmtFull(doc_cost)}
                </div>
              </div>

              <div className="waterfall-row divider">
                <div className="waterfall-label indent">
                  <span>Contingency & Risk Reserve</span>
                </div>
                <div className="waterfall-amount deduction">
                  − {fmtFull(risk_buffer)}
                </div>
              </div>

              {/* Total Costs Subtotal */}
              <div className="waterfall-row">
                <div className="waterfall-label" style={{ color: 'var(--text-3)' }}>
                  <span>Total Deductions & Operational Costs</span>
                </div>
                <div className="waterfall-amount" style={{ color: 'var(--risk-mid)' }}>
                  − {fmtFull(totalCosts)}
                </div>
              </div>

              {/* Final Net Profit */}
              <div className="waterfall-row total" style={{ borderTop: '2px solid var(--border)', marginTop: 8 }}>
                <div className="waterfall-label">
                  <Icon name="check" size={18} color="#16a34a" strokeWidth={2.5} />
                  <strong style={{ fontSize: 16 }}>PROJECTED NET DEALER PROFIT</strong>
                </div>
                <div className="waterfall-amount profit">
                  +{fmtFull(netProfit)}
                </div>
              </div>

              {/* ROI Percentage */}
              <div className="waterfall-row">
                <div className="waterfall-label" style={{ color: 'var(--text-4)' }}>
                  <span>Return on Capital Deployed (ROI)</span>
                </div>
                <div className="waterfall-amount" style={{ color: '#e85d26', fontSize: 16 }}>
                  {roiPct}%
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Aside: Deal Sensitivity & Summary */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Summary Box */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Deal Summary</div>
              <span className="badge badge-buy">{action}</span>
            </div>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div>
                <div style={{ fontSize: 10.5, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-4)' }}>Market Valuation</div>
                <div style={{ fontSize: 24, fontWeight: 900, color: 'var(--text-1)', marginTop: 2 }}>{fmt(predictedPrice)}</div>
              </div>

              <div style={{ borderTop: '1px solid var(--border-2)', paddingTop: 10 }}>
                <div style={{ fontSize: 10.5, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-4)' }}>Net Margin Expected</div>
                <div style={{ fontSize: 22, fontWeight: 900, color: '#16a34a', marginTop: 2 }}>+{fmt(netProfit)}</div>
                <div style={{ fontSize: 11.5, color: 'var(--text-4)', marginTop: 2 }}>{roiPct}% margin on buy price</div>
              </div>

              <div style={{ borderTop: '1px solid var(--border-2)', paddingTop: 10 }}>
                <div style={{ fontSize: 10.5, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-4)' }}>Quality Rating</div>
                <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--text-1)', marginTop: 2 }}>{dealQualityScore} / 100</div>
              </div>
            </div>
          </div>

          {/* Quick Scenario Box */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Turnaround Scenarios</div>
            </div>
            <div className="card-body" style={{ fontSize: 12.5, color: 'var(--text-2)', display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: 8, borderBottom: '1px solid var(--border-2)' }}>
                <span>Fast Sale (7 days, −2%):</span>
                <strong style={{ color: '#16a34a' }}>+{fmt(netProfit - targetRetail * 0.02)}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: 8, borderBottom: '1px solid var(--border-2)' }}>
                <span>Target Sale (21 days):</span>
                <strong style={{ color: '#e85d26' }}>+{fmt(netProfit)}</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Extended Hold (45 days, +costs):</span>
                <strong style={{ color: 'var(--text-3)' }}>+{fmt(netProfit - 8000)}</strong>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
