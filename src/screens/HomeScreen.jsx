import { useMemo } from 'react';
import { useApp } from '../context/AppContext.jsx';
import { useAuth } from '../context/AuthContext.jsx';
import { formatINR } from '../utils/mockData.js';
import Icon from '../components/Icon.jsx';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  ResponsiveContainer, Tooltip, Cell,
} from 'recharts';

const BRAND_COLORS = [
  '#f75d34','#2563eb','#16a34a','#d97706','#7c3aed',
  '#0891b2','#be185d','#059669','#9333ea','#c2410c',
];

function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tip">
      <div className="chart-tip-label">{label}</div>
      {payload.map((p, i) => (
        <div key={i}>{p.name}: <strong>{p.value}L</strong></div>
      ))}
    </div>
  );
}

function EmptyHome({ setActiveScreen }) {
  return (
    <div className="home-empty">
      <div className="home-empty-icon">
        <Icon name="car" size={28} color="#f75d34" strokeWidth={1.8} />
      </div>
      <div className="home-empty-title">No evaluations yet</div>
      <div className="home-empty-desc">
        Run your first ML valuation to populate this dashboard with real data,
        KPIs, and dealer insights.
      </div>
      <button className="btn btn-primary btn-lg" onClick={() => setActiveScreen('input')}>
        <Icon name="car" size={16} color="white" strokeWidth={2} />
        Start First Valuation
      </button>
    </div>
  );
}

export default function HomeScreen() {
  const { setActiveScreen, evaluations } = useApp();
  const { currentUser } = useAuth();

  const greeting = useMemo(() => {
    const h = new Date().getHours();
    if (h < 12) return 'Good morning';
    if (h < 17) return 'Good afternoon';
    return 'Good evening';
  }, []);

  const data = useMemo(() => {
    const records = [...evaluations];
    const projectedProfit = records.reduce((s, v) => s + Number(v.expectedProfit || 0), 0);
    const pipeline       = records.reduce((s, v) => s + Number(v.marketValue || 0), 0);
    const buyCount       = records.filter(v => v.action === 'BUY').length;
    const avgProfit      = records.length ? Math.round(projectedProfit / records.length) : 0;

    const topOpportunities = [...records]
      .filter(v => v.marketValue > 0)
      .sort((a, b) => (b.dealQualityScore || 0) - (a.dealQualityScore || 0))
      .slice(0, 6);

    const brandAgg = {};
    records.forEach(v => {
      if (!v.brand || !v.marketValue) return;
      brandAgg[v.brand] ||= { brand: v.brand, total: 0, count: 0 };
      brandAgg[v.brand].total += Number(v.marketValue || 0);
      brandAgg[v.brand].count += 1;
    });
    const marketPulse = Object.values(brandAgg)
      .map(b => ({ brand: b.brand, avgResaleL: +(b.total / b.count / 100000).toFixed(1) }))
      .sort((a, b) => b.avgResaleL - a.avgResaleL)
      .slice(0, 8);

    const riskCount = records.filter(v => Number(v.riskScore || 0) >= 65).length;
    const recent = records.slice(0, 8);

    return {
      kpis: {
        evaluations: records.length,
        buy: buyCount,
        avgProfit,
        pipelineL: +(pipeline / 100000).toFixed(1),
      },
      topOpportunities,
      marketPulse,
      riskCount,
      recent,
    };
  }, [evaluations]);

  const getActionClass = (action = '') => {
    const a = String(action).toUpperCase();
    if (a === 'BUY') return 'buy';
    if (a === 'NEGOTIATE') return 'negotiate';
    if (a === 'REJECT') return 'reject';
    return 'review';
  };

  const fmtL = (n) => {
    if (!n) return '₹0';
    if (n >= 100000) return `₹${(n / 100000).toFixed(2)}L`;
    return formatINR(n);
  };

  return (
    <div className="screen screen-wide home-screen">
      {/* Greeting */}
      <div className="home-greeting page-header">
        <div>
          <div className="home-greeting-name">
            {greeting}, {currentUser?.name?.split(' ')[0] || 'Dealer'} 👋
          </div>
          <div className="home-greeting-sub">
            {evaluations.length === 0
              ? 'Run your first valuation to get started.'
              : `${data.kpis.evaluations} evaluations · ${data.kpis.buy} BUY signals · ₹${data.kpis.pipelineL}L pipeline`
            }
          </div>
        </div>
        <div className="page-header-actions home-cta-row">
          <button className="btn btn-primary" onClick={() => setActiveScreen('input')}>
            <Icon name="car" size={15} color="white" strokeWidth={2} />
            New Valuation
          </button>
          <button className="btn btn-secondary" onClick={() => setActiveScreen('dashboard')}>
            <Icon name="chart" size={15} color="#475569" strokeWidth={2} />
            Analytics
          </button>
        </div>
      </div>

      {/* KPI tiles */}
      <div className="kpi-grid">
        <div className="kpi-tile">
          <div className="kpi-tile-header">
            <div className="kpi-tile-label">Total Evaluations</div>
            <div className="kpi-icon" style={{ background: '#dbeafe' }}>
              <Icon name="clipboard" size={14} color="#2563eb" strokeWidth={2} />
            </div>
          </div>
          <div className="kpi-tile-value">{data.kpis.evaluations}</div>
          <div className="kpi-tile-sub">ML-powered valuations</div>
        </div>

        <div className="kpi-tile">
          <div className="kpi-tile-header">
            <div className="kpi-tile-label">BUY Signals</div>
            <div className="kpi-icon" style={{ background: '#dcfce7' }}>
              <Icon name="shield" size={14} color="#16a34a" strokeWidth={2} />
            </div>
          </div>
          <div className="kpi-tile-value" style={{ color: '#16a34a' }}>{data.kpis.buy}</div>
          <div className="kpi-tile-sub">Recommended acquisitions</div>
        </div>

        <div className="kpi-tile">
          <div className="kpi-tile-header">
            <div className="kpi-tile-label">Avg. Net Profit</div>
            <div className="kpi-icon" style={{ background: '#fff4f0' }}>
              <Icon name="coins" size={14} color="#f75d34" strokeWidth={2} />
            </div>
          </div>
          <div className="kpi-tile-value" style={{ color: data.kpis.avgProfit > 0 ? '#16a34a' : '#0f172a' }}>
            {fmtL(data.kpis.avgProfit)}
          </div>
          <div className="kpi-tile-sub">Per evaluated vehicle</div>
        </div>

        <div className="kpi-tile">
          <div className="kpi-tile-header">
            <div className="kpi-tile-label">Pipeline Value</div>
            <div className="kpi-icon" style={{ background: '#f3e8ff' }}>
              <Icon name="lightning" size={14} color="#7c3aed" strokeWidth={2} />
            </div>
          </div>
          <div className="kpi-tile-value">₹{data.kpis.pipelineL}L</div>
          <div className="kpi-tile-sub">Total market values</div>
        </div>
      </div>

      {evaluations.length === 0 ? (
        <div className="card">
          <EmptyHome setActiveScreen={setActiveScreen} />
        </div>
      ) : (
        <div className="home-main-grid">
          {/* Left column: Recent evaluations table */}
          <div className="card card-analytics" style={{ padding: '0', overflow: 'hidden' }}>
            <div style={{ padding: '18px 20px 14px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-1)' }}>Recent Evaluations</div>
                <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>Your latest ML valuations</div>
              </div>
              <button className="btn btn-ghost btn-sm" onClick={() => setActiveScreen('dashboard')}>
                View all
              </button>
            </div>
            <div className="home-eval-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Vehicle</th>
                    <th>Market Value</th>
                    <th>Action</th>
                    <th>Deal Score</th>
                  </tr>
                </thead>
                <tbody>
                  {data.recent.map(v => (
                    <tr key={v.id}>
                      <td>
                        <div className="vehicle-name">{v.brand} {v.model}</div>
                        <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>
                          {v.year} · {v.fuel} · {(Number(v.kmDriven || 0)/1000).toFixed(0)}k km
                        </div>
                      </td>
                      <td className="price-cell">{fmtL(v.marketValue)}</td>
                      <td>
                        <span className={`home-action-pill ${getActionClass(v.action)}`}>
                          {v.action}
                        </span>
                      </td>
                      <td style={{ fontWeight: 700, color: 'var(--text-1)' }}>
                        {v.dealQualityScore || '-'}/100
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Right column: Top opportunities + chart */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Top opportunities */}
            <div className="card card-profit">
              <div className="card-header">
                <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-1)' }}>Top Opportunities</div>
                <button className="btn btn-ghost btn-sm" onClick={() => setActiveScreen('input')}>+ Evaluate</button>
              </div>
              <div className="home-opportunity-list">
                {data.topOpportunities.slice(0, 5).map(v => (
                  <div key={v.id} className="home-opp-card">
                    <div>
                      <div className="home-opp-vehicle">{v.brand} {v.model} {v.year}</div>
                      <div className="home-opp-meta">
                        {v.city} · {v.fuel} · Deal {v.dealQualityScore || 0}/100
                      </div>
                    </div>
                    <div className="home-opp-right">
                      <div className="home-opp-price">{fmtL(v.expectedProfit)}</div>
                      <span className={`home-action-pill ${getActionClass(v.action)}`}>
                        {v.action}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Market pulse chart */}
            {data.marketPulse.length > 0 && (
              <div className="card card-prediction">
                <div className="card-header">
                  <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-1)' }}>Brand Values (Avg ₹L)</div>
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={data.marketPulse} layout="vertical" margin={{ top: 0, right: 16, left: 8, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
                    <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                    <YAxis dataKey="brand" type="category" tick={{ fill: '#475569', fontSize: 11 }} width={60} />
                    <Tooltip content={<ChartTip />} />
                    <Bar dataKey="avgResaleL" name="Avg Market Value" radius={[0, 4, 4, 0]}>
                      {data.marketPulse.map((_, i) => (
                        <Cell key={i} fill={BRAND_COLORS[i % BRAND_COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
