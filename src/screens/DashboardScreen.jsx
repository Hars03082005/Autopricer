import { useState, useMemo } from 'react';
import { useApp } from '../context/AppContext.jsx';
import { formatINR } from '../utils/mockData.js';
import { exportEvaluationsToCSV } from '../utils/csvExporter.js';
import Icon from '../components/Icon.jsx';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ScatterChart, Scatter, AreaChart, Area, Cell,
} from 'recharts';

const fmtL = (n) => {
  const v = Number(n || 0);
  if (v >= 10000000) return `₹${(v / 10000000).toFixed(2)}Cr`;
  if (v >= 100000)   return `₹${(v / 100000).toFixed(2)}L`;
  if (v >= 1000)     return `₹${(v / 1000).toFixed(0)}k`;
  return `₹${Math.round(v).toLocaleString('en-IN')}`;
};

const Tip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tip">
      <div className="chart-tip-label">{label || payload[0]?.payload?.brand || payload[0]?.payload?.name || payload[0]?.payload?.vehicle}</div>
      {payload.map((p, i) => (
        <div key={i}>
          {p.name}: <strong>{typeof p.value === 'number' && /price|profit|value/i.test(p.name) ? fmtL(p.value) : p.value}</strong>
        </div>
      ))}
    </div>
  );
};

export default function DashboardScreen() {
  const { evaluations, setActiveScreen, clearEvaluations } = useApp();
  const [timeFilter, setTimeFilter] = useState('ALL');
  const [brandFilter, setBrandFilter] = useState('All');

  const filtered = useMemo(() => {
    return evaluations.filter(v => {
      if (brandFilter !== 'All' && v.brand !== brandFilter) return false;
      return true;
    });
  }, [evaluations, brandFilter]);

  const brands = useMemo(() => {
    return ['All', ...Array.from(new Set(evaluations.map(v => v.brand).filter(Boolean))).sort()];
  }, [evaluations]);

  const metrics = useMemo(() => {
    const count = filtered.length;
    const totalPipeline = filtered.reduce((s, v) => s + Number(v.marketValue || 0), 0);
    const avgValue = count ? Math.round(totalPipeline / count) : 0;
    const totalProfit = filtered.reduce((s, v) => s + Number(v.expectedProfit || 0), 0);
    const avgProfit = count ? Math.round(totalProfit / count) : 0;
    const buyCount = filtered.filter(v => (v.action === 'BUY' || (v.dealQualityScore || 0) >= 65)).length;
    const buyRate = count ? Math.round((buyCount / count) * 100) : 0;

    // Brand performance
    const brandMap = {};
    filtered.forEach(v => {
      if (!v.brand) return;
      brandMap[v.brand] ||= { brand: v.brand, count: 0, profit: 0, avgValue: 0 };
      brandMap[v.brand].count += 1;
      brandMap[v.brand].profit += Number(v.expectedProfit || 0);
      brandMap[v.brand].avgValue += Number(v.marketValue || 0);
    });
    const brandChartData = Object.values(brandMap)
      .map(b => ({
        brand: b.brand,
        evals: b.count,
        avgProfitL: +(b.profit / b.count / 100000).toFixed(2),
        avgValL: +(b.avgValue / b.count / 100000).toFixed(2),
      }))
      .sort((a, b) => b.evals - a.evals)
      .slice(0, 7);

    // Funnel counts
    const buyTotal = filtered.filter(v => v.action === 'BUY').length;
    const inspectTotal = filtered.filter(v => v.action === 'NEGOTIATE' || v.action === 'INSPECT' || v.action === 'BUY AFTER INSPECTION').length;
    const passTotal = filtered.filter(v => v.action === 'REJECT' || v.action === 'PASS').length;

    // Price distribution bins
    const priceBins = [
      { name: '< ₹5L', count: 0 },
      { name: '₹5L–10L', count: 0 },
      { name: '₹10L–20L', count: 0 },
      { name: '₹20L–35L', count: 0 },
      { name: '> ₹35L', count: 0 },
    ];
    filtered.forEach(v => {
      const val = Number(v.marketValue || 0);
      if (val < 500000) priceBins[0].count += 1;
      else if (val <= 1000000) priceBins[1].count += 1;
      else if (val <= 2000000) priceBins[2].count += 1;
      else if (val <= 3500000) priceBins[3].count += 1;
      else priceBins[4].count += 1;
    });

    return {
      count,
      totalPipeline,
      avgValue,
      totalProfit,
      avgProfit,
      buyCount,
      buyRate,
      brandChartData,
      buyTotal,
      inspectTotal,
      passTotal,
      priceBins,
    };
  }, [filtered]);

  if (!evaluations.length) {
    return (
      <div className="screen">
        <div className="empty-screen">
          <div className="empty-icon-wrap">
            <Icon name="chart" size={32} color="#e85d26" strokeWidth={1.8} />
          </div>
          <div className="empty-title">Market Intelligence Empty</div>
          <div className="empty-sub">
            Real-time dealership intelligence builds up as you evaluate vehicles. Run ML valuations to populate charts, margins, and funnel analytics.
          </div>
          <button className="btn btn-primary btn-lg" onClick={() => setActiveScreen('input')}>
            <Icon name="car" size={15} color="white" strokeWidth={2} />
            <span>Start First Valuation</span>
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="screen">
      {/* Header & Controls */}
      <div className="analytics-header">
        <div>
          <div className="page-title">Market & Acquisition Intelligence</div>
          <div className="page-subtitle">Aggregate metrics, margin distributions, and brand liquidity across evaluated inventory.</div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          {/* Brand select */}
          <select
            className="form-select"
            value={brandFilter}
            onChange={(e) => setBrandFilter(e.target.value)}
            style={{ width: 140, padding: '6px 10px', fontSize: 12.5 }}
          >
            {brands.map(b => (
              <option key={b} value={b}>{b === 'All' ? 'All Brands' : b}</option>
            ))}
          </select>

          {/* Time Filter buttons */}
          <div className="time-filter-group">
            {['7D', '30D', '90D', 'ALL'].map(t => (
              <button
                key={t}
                className={`time-filter-btn ${timeFilter === t ? 'active' : ''}`}
                onClick={() => setTimeFilter(t)}
              >
                {t}
              </button>
            ))}
          </div>

          <button
            className="btn btn-secondary btn-sm"
            onClick={() => exportEvaluationsToCSV(filtered)}
          >
            <Icon name="upload" size={13} strokeWidth={2} />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* 6 Top Metric Tiles */}
      <div className="metric-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', marginBottom: 20 }}>
        <div className="metric-tile">
          <div className="metric-label">Evaluations</div>
          <div className="metric-value">{metrics.count}</div>
          <div className="metric-sub">Total pipeline assets</div>
        </div>

        <div className="metric-tile">
          <div className="metric-label">Qualified Deals</div>
          <div className="metric-value" style={{ color: '#16a34a' }}>{metrics.buyCount}</div>
          <div className="metric-sub">Score &ge; 65 / BUY action</div>
        </div>

        <div className="metric-tile">
          <div className="metric-label">Average Market Value</div>
          <div className="metric-value">{fmtL(metrics.avgValue)}</div>
          <div className="metric-sub">Per vehicle average</div>
        </div>

        <div className="metric-tile">
          <div className="metric-label">Average Profit</div>
          <div className="metric-value" style={{ color: '#e85d26' }}>+{fmtL(metrics.avgProfit)}</div>
          <div className="metric-sub">Projected net per vehicle</div>
        </div>

        <div className="metric-tile">
          <div className="metric-label">Buy Rate</div>
          <div className="metric-value" style={{ color: '#16a34a' }}>{metrics.buyRate}%</div>
          <div className="metric-sub">Conversion of pipeline</div>
        </div>

        <div className="metric-tile">
          <div className="metric-label">Total Pipeline Value</div>
          <div className="metric-value">{fmtL(metrics.totalPipeline)}</div>
          <div className="metric-sub">Aggregate gross value</div>
        </div>
      </div>

      {/* Opportunity Funnel Card */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header">
          <div className="card-title">Opportunity Funnel</div>
          <span style={{ fontSize: 11.5, color: 'var(--text-4)' }}>Decision Distribution</span>
        </div>
        <div className="card-body">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14, textAlign: 'center' }}>
            <div style={{ padding: 14, background: '#f0fdf4', border: '1px solid #86efac', borderRadius: 'var(--r-md)' }}>
              <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', color: '#15803d' }}>BUY (Target Deals)</div>
              <div style={{ fontSize: 28, fontWeight: 900, color: '#15803d', marginTop: 4 }}>{metrics.buyTotal}</div>
              <div style={{ fontSize: 11, color: '#16a34a', marginTop: 2 }}>
                {metrics.count ? Math.round((metrics.buyTotal / metrics.count) * 100) : 0}% of evaluations
              </div>
            </div>

            <div style={{ padding: 14, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 'var(--r-md)' }}>
              <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', color: '#b45309' }}>INSPECT / NEGOTIATE</div>
              <div style={{ fontSize: 28, fontWeight: 900, color: '#b45309', marginTop: 4 }}>{metrics.inspectTotal}</div>
              <div style={{ fontSize: 11, color: '#d97706', marginTop: 2 }}>
                {metrics.count ? Math.round((metrics.inspectTotal / metrics.count) * 100) : 0}% of evaluations
              </div>
            </div>

            <div style={{ padding: 14, background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 'var(--r-md)' }}>
              <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', color: '#b91c1c' }}>PASS (Thin Margin / High Risk)</div>
              <div style={{ fontSize: 28, fontWeight: 900, color: '#b91c1c', marginTop: 4 }}>{metrics.passTotal}</div>
              <div style={{ fontSize: 11, color: '#dc2626', marginTop: 2 }}>
                {metrics.count ? Math.round((metrics.passTotal / metrics.count) * 100) : 0}% of evaluations
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 2-Chart Grid */}
      <div className="analytics-grid">
        {/* Brand Performance */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Brand Pipeline Volume</div>
            <span style={{ fontSize: 11, color: 'var(--text-4)' }}>Evaluation count by make</span>
          </div>
          <div className="card-body" style={{ height: 250, minHeight: 250, width: '100%', position: 'relative' }}>
            {metrics.brandChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={220} minHeight={200}>
                <BarChart data={metrics.brandChartData} margin={{ top: 10, right: 10, left: -15, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                  <XAxis dataKey="brand" tick={{ fontSize: 11, fill: 'var(--text-4)' }} interval={0} />
                  <YAxis tick={{ fontSize: 11, fill: 'var(--text-4)' }} allowDecimals={false} />
                  <Tooltip content={<Tip />} />
                  <Bar dataKey="evals" name="Evaluations" fill="#1e2d3d" radius={[4, 4, 0, 0]}>
                    {metrics.brandChartData.map((_, idx) => (
                      <Cell key={idx} fill={idx === 0 ? '#e85d26' : '#1e2d3d'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-screen" style={{ minHeight: 180 }}>
                <span style={{ fontSize: 12, color: 'var(--text-4)' }}>No brand data available</span>
              </div>
            )}
          </div>
        </div>

        {/* Price Distribution */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Valuation Price Distribution</div>
            <span style={{ fontSize: 11, color: 'var(--text-4)' }}>Inventory brackets</span>
          </div>
          <div className="card-body" style={{ height: 250, minHeight: 250, width: '100%', position: 'relative' }}>
            <ResponsiveContainer width="100%" height={220} minHeight={200}>
              <AreaChart data={metrics.priceBins} margin={{ top: 10, right: 10, left: -15, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'var(--text-4)' }} />
                <YAxis tick={{ fontSize: 11, fill: 'var(--text-4)' }} allowDecimals={false} />
                <Tooltip content={<Tip />} />
                <Area type="monotone" dataKey="count" name="Vehicles" stroke="#e85d26" strokeWidth={2} fill="#fdf0ea" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
