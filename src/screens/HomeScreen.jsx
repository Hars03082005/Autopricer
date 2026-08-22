import { useMemo } from 'react';
import { useApp } from '../context/AppContext.jsx';
import { useAuth } from '../context/AuthContext.jsx';
import { exportEvaluationsToCSV } from '../utils/csvExporter.js';
import Icon from '../components/Icon.jsx';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  ResponsiveContainer, Tooltip, Cell,
} from 'recharts';

const fmtL = (n) => {
  const v = Number(n || 0);
  if (v >= 10000000) return `₹${(v / 10000000).toFixed(2)}Cr`;
  if (v >= 100000)   return `₹${(v / 100000).toFixed(2)}L`;
  if (v >= 1000)     return `₹${(v / 1000).toFixed(0)}k`;
  return `₹${Math.round(v).toLocaleString('en-IN')}`;
};

const fmtDate = (iso) => {
  if (!iso) return 'Today';
  try {
    const d = new Date(iso);
    return d.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' });
  } catch {
    return 'Recent';
  }
};

function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tip">
      <div className="chart-tip-label">{label}</div>
      {payload.map((p, i) => (
        <div key={i}>{p.name}: <strong>₹{p.value}L</strong></div>
      ))}
    </div>
  );
}

function ActionBadge({ action }) {
  const act = String(action || 'REVIEW').toUpperCase();
  if (act === 'BUY') {
    return <span className="badge badge-buy"><Icon name="check" size={10} color="#15803d" strokeWidth={2.5} /> BUY</span>;
  }
  if (act === 'NEGOTIATE' || act === 'INSPECT' || act === 'BUY AFTER INSPECTION') {
    return <span className="badge badge-caution">INSPECT</span>;
  }
  return <span className="badge badge-risk">PASS</span>;
}

export default function HomeScreen() {
  const { setActiveScreen, evaluations, viewEvaluation, editEvaluation, deleteEvaluation } = useApp();
  const { currentUser } = useAuth();

  const greeting = useMemo(() => {
    const h = new Date().getHours();
    if (h < 12) return 'Good morning';
    if (h < 17) return 'Good afternoon';
    return 'Good evening';
  }, []);

  const data = useMemo(() => {
    const records = [...evaluations];
    const totalProfit = records.reduce((s, v) => s + Number(v.expectedProfit || 0), 0);
    const pipelineValue = records.reduce((s, v) => s + Number(v.marketValue || 0), 0);
    const buyOpportunities = records.filter(v => (v.action === 'BUY' || (v.dealQualityScore || 0) >= 65));
    const activeCount = buyOpportunities.length;

    const brandAgg = {};
    records.forEach(v => {
      if (!v.brand || !v.marketValue) return;
      brandAgg[v.brand] ||= { brand: v.brand, total: 0, count: 0 };
      brandAgg[v.brand].total += Number(v.marketValue || 0);
      brandAgg[v.brand].count += 1;
    });
    const marketPulse = Object.values(brandAgg)
      .map(b => ({ brand: b.brand, avgValueL: +(b.total / b.count / 100000).toFixed(2) }))
      .sort((a, b) => b.avgValueL - a.avgValueL)
      .slice(0, 6);

    return {
      records,
      totalProfit,
      pipelineValue,
      activeCount,
      marketPulse,
    };
  }, [evaluations]);

  return (
    <div className="screen">
      {/* Dashboard Top Greeting & Action Header */}
      <div className="dash-greeting">
        <div>
          <div className="page-title">{greeting}, {currentUser?.name?.split(' ')[0] || 'Dealer'}</div>
          <div className="page-subtitle">PriceRef Valuation Terminal · Bengaluru Market Context</div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {evaluations.length > 0 && (
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => exportEvaluationsToCSV(evaluations)}
              title="Export all evaluations to CSV"
            >
              <Icon name="upload" size={13} strokeWidth={2} />
              <span>Export CSV</span>
            </button>
          )}
          <button className="btn btn-primary btn-sm" onClick={() => setActiveScreen('input')}>
            <Icon name="car" size={13} color="white" strokeWidth={2} />
            <span>New Valuation</span>
          </button>
        </div>
      </div>

      {/* 4 Pipeline Stat Tiles */}
      <div className="pipeline-grid">
        <div className="pipeline-tile">
          <div className="pipeline-tile-label">Total Evaluations</div>
          <div className="pipeline-tile-value">{evaluations.length}</div>
          <div className="pipeline-tile-sub">Vehicles processed</div>
        </div>
        <div className="pipeline-tile">
          <div className="pipeline-tile-label">Active Opportunities</div>
          <div className="pipeline-tile-value" style={{ color: '#16a34a' }}>{data.activeCount}</div>
          <div className="pipeline-tile-sub">High deal quality (Score &ge; 65)</div>
        </div>
        <div className="pipeline-tile">
          <div className="pipeline-tile-label">Potential Profit</div>
          <div className="pipeline-tile-value" style={{ color: '#e85d26' }}>{fmtL(data.totalProfit)}</div>
          <div className="pipeline-tile-sub">Projected dealer margin</div>
        </div>
        <div className="pipeline-tile">
          <div className="pipeline-tile-label">Pipeline Value</div>
          <div className="pipeline-tile-value">{fmtL(data.pipelineValue)}</div>
          <div className="pipeline-tile-sub">Total inventory market value</div>
        </div>
      </div>

      {/* Main Valuation Pipeline Table */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header">
          <div>
            <div className="card-title">Valuation Pipeline</div>
            <div style={{ fontSize: 11.5, color: 'var(--text-4)', marginTop: 2 }}>
              Comprehensive inventory valuation records with direct acquisition actions
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {evaluations.length > 0 && (
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => exportEvaluationsToCSV(evaluations)}
              >
                <Icon name="upload" size={12} strokeWidth={2} />
                <span>Export CSV</span>
              </button>
            )}
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => setActiveScreen('input')}
            >
              + Evaluate Car
            </button>
          </div>
        </div>

        {evaluations.length === 0 ? (
          <div className="empty-screen" style={{ padding: '40px 20px' }}>
            <div className="empty-icon-wrap">
              <Icon name="car" size={26} color="#e85d26" strokeWidth={1.8} />
            </div>
            <div className="empty-title">No evaluations in pipeline</div>
            <div className="empty-sub">
              Start by running a valuation. Your pipeline will populate with real-time acquisition pricing, profit margins, and deal quality scores.
            </div>
            <button className="btn btn-primary btn-md" onClick={() => setActiveScreen('input')}>
              <Icon name="car" size={14} color="white" strokeWidth={2} />
              <span>Start First Valuation</span>
            </button>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="pr-table">
              <thead>
                <tr>
                  <th style={{ width: 80 }}>STATUS</th>
                  <th>VEHICLE</th>
                  <th>YEAR</th>
                  <th>VARIANT</th>
                  <th style={{ textAlign: 'right' }}>MARKET VALUE</th>
                  <th style={{ textAlign: 'right' }}>BUY RANGE</th>
                  <th style={{ textAlign: 'right' }}>EXPECTED PROFIT</th>
                  <th style={{ textAlign: 'center' }}>DECISION</th>
                  <th style={{ textAlign: 'center' }}>DATE</th>
                  <th style={{ textAlign: 'center', minWidth: 150 }}>ACTIONS</th>
                </tr>
              </thead>
              <tbody>
                {evaluations.map((v, i) => {
                  const buyLow = v.opening_offer || Math.round(((v.buyPrice || v.recommendedBuyPrice || 0) * 0.95) / 500) * 500;
                  const buyHigh = v.max_offer || Math.round(((v.buyPrice || v.recommendedBuyPrice || 0) * 1.03) / 500) * 500;

                  return (
                    <tr key={v.id || i}>
                      <td>
                        <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-4)', letterSpacing: 0.5, textTransform: 'uppercase' }}>
                          COMPLETED
                        </span>
                      </td>
                      <td>
                        <div style={{ fontWeight: 700, color: 'var(--text-1)' }}>{v.brand} {v.model}</div>
                        <div style={{ fontSize: 11, color: 'var(--text-4)' }}>
                          {Number(v.odometer || v.mileage || 0).toLocaleString('en-IN')} km · {v.fuel || 'Petrol'}
                        </div>
                      </td>
                      <td style={{ fontWeight: 600, color: 'var(--text-2)' }}>{v.year}</td>
                      <td>
                        <div style={{ fontSize: 12, color: 'var(--text-2)', maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {v.variant || 'Standard'}
                        </div>
                      </td>
                      <td className="num" style={{ fontWeight: 800 }}>{fmtL(v.marketValue)}</td>
                      <td className="num" style={{ color: '#15803d', fontWeight: 600 }}>
                        {fmtL(buyLow)} – {fmtL(buyHigh)}
                      </td>
                      <td className="num" style={{ color: '#e85d26', fontWeight: 700 }}>
                        +{fmtL(v.expectedProfit)}
                        <span style={{ fontSize: 10.5, color: 'var(--text-4)', marginLeft: 3 }}>
                          ({Number(v.marginPct || 10).toFixed(0)}%)
                        </span>
                      </td>
                      <td style={{ textAlign: 'center' }}>
                        <ActionBadge action={v.action} />
                      </td>
                      <td style={{ textAlign: 'center', fontSize: 11.5, color: 'var(--text-4)' }}>
                        {fmtDate(v.createdAt)}
                      </td>
                      <td style={{ textAlign: 'center' }}>
                        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                          <button
                            className="btn btn-secondary btn-sm"
                            style={{ padding: '4px 9px', fontSize: 11.5, fontWeight: 700 }}
                            onClick={() => viewEvaluation(v)}
                            title="View complete valuation report"
                          >
                            VIEW
                          </button>
                          <button
                            className="btn btn-ghost btn-sm"
                            style={{ padding: '4px 8px', fontSize: 11.5, color: 'var(--text-3)' }}
                            onClick={() => editEvaluation(v)}
                            title="Edit valuation inputs"
                          >
                            EDIT
                          </button>
                          <button
                            className="btn btn-ghost btn-sm"
                            style={{ padding: '4px 6px', fontSize: 12, color: '#dc2626' }}
                            onClick={() => {
                              if (confirm(`Remove ${v.year} ${v.brand} ${v.model} from valuation history?`)) {
                                deleteEvaluation(v.id);
                              }
                            }}
                            title="Remove evaluation"
                          >
                            ✕
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Bottom Grid: Brand Average Resale Value & Decision Funnel */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* Brand Resale Pulse */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Brand Average Resale Value</div>
            <span style={{ fontSize: 11, color: 'var(--text-4)' }}>Bengaluru Market Intelligence</span>
          </div>
          <div className="card-body" style={{ height: 250, minHeight: 250, width: '100%', position: 'relative' }}>
            {data.marketPulse.length > 0 ? (
              <ResponsiveContainer width="100%" height={220} minHeight={200}>
                <BarChart data={data.marketPulse} margin={{ top: 10, right: 10, left: -10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                  <XAxis dataKey="brand" tick={{ fontSize: 11, fill: 'var(--text-4)' }} interval={0} />
                  <YAxis tick={{ fontSize: 11, fill: 'var(--text-4)' }} tickFormatter={v => `₹${v}L`} />
                  <Tooltip content={<ChartTip />} />
                  <Bar dataKey="avgValueL" name="Avg Value" radius={[4, 4, 0, 0]}>
                    {data.marketPulse.map((_, idx) => (
                      <Cell key={idx} fill={idx === 0 ? '#e85d26' : '#1e2d3d'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-screen" style={{ minHeight: 180 }}>
                <div style={{ fontSize: 12, color: 'var(--text-4)' }}>Run valuations to populate brand benchmark chart</div>
              </div>
            )}
          </div>
        </div>

        {/* Acquisition Call Breakdown */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Pipeline Opportunity Health</div>
            <span style={{ fontSize: 11, color: 'var(--text-4)' }}>Decision Distribution</span>
          </div>
          <div className="card-body">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, textAlign: 'center', marginBottom: 14 }}>
              <div style={{ padding: 12, background: '#f0fdf4', border: '1px solid #86efac', borderRadius: 'var(--r-md)' }}>
                <div style={{ fontSize: 10.5, fontWeight: 800, textTransform: 'uppercase', color: '#15803d' }}>BUY</div>
                <div style={{ fontSize: 24, fontWeight: 900, color: '#15803d', marginTop: 2 }}>
                  {evaluations.filter(e => e.action === 'BUY').length}
                </div>
                <div style={{ fontSize: 10.5, color: '#16a34a' }}>Target Deals</div>
              </div>
              <div style={{ padding: 12, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 'var(--r-md)' }}>
                <div style={{ fontSize: 10.5, fontWeight: 800, textTransform: 'uppercase', color: '#b45309' }}>INSPECT</div>
                <div style={{ fontSize: 24, fontWeight: 900, color: '#b45309', marginTop: 2 }}>
                  {evaluations.filter(e => ['NEGOTIATE', 'INSPECT', 'BUY AFTER INSPECTION'].includes(e.action)).length}
                </div>
                <div style={{ fontSize: 10.5, color: '#d97706' }}>Negotiate / Inspect</div>
              </div>
              <div style={{ padding: 12, background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 'var(--r-md)' }}>
                <div style={{ fontSize: 10.5, fontWeight: 800, textTransform: 'uppercase', color: '#b91c1c' }}>PASS</div>
                <div style={{ fontSize: 24, fontWeight: 900, color: '#b91c1c', marginTop: 2 }}>
                  {evaluations.filter(e => e.action === 'PASS' || e.action === 'MANUAL REVIEW').length}
                </div>
                <div style={{ fontSize: 10.5, color: '#dc2626' }}>Thin Margin / Risk</div>
              </div>
            </div>

            <div style={{ padding: 12, background: 'var(--surface-2)', borderRadius: 'var(--r-md)', border: '1px solid var(--border-2)', fontSize: 12, color: 'var(--text-3)' }}>
              <div style={{ fontWeight: 700, color: 'var(--text-1)', marginBottom: 2 }}>Acquisition Intelligence</div>
              <div>Deals marked as BUY have sufficient margin buffer (&ge;10%) after accounting for holding and standard reconditioning allowances.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
