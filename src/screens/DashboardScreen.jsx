import { useState, useMemo } from 'react';
import { useApp } from '../context/AppContext.jsx';
import { formatINR } from '../utils/mockData.js';
import { exportEvaluationsToCSV } from '../utils/csvExporter.js';
import Icon from '../components/Icon.jsx';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ScatterChart, Scatter, AreaChart, Area, Cell,
} from 'recharts';
const TABS   = ['Overview', 'Brands', 'Profit', 'Trends'];
const COLORS  = ['#f75d34','#2563eb','#16a34a','#d97706','#7c3aed','#0891b2','#be185d','#059669','#9333ea','#c2410c'];
const ACTION_COLORS = { BUY:'#16a34a', NEGOTIATE:'#d97706', REJECT:'#dc2626', 'MANUAL REVIEW':'#94a3b8' };
const fmtL = (n) => {
  const v = Number(n||0);
  if (v >= 100000) return `₹${(v/100000).toFixed(2)}L`;
  return formatINR(v);
};
function price_ok(price, range) {
  if (range === 'Under ₹5L')   return price < 500000;
  if (range === '₹5L–₹10L')   return price >= 500000  && price <= 1000000;
  if (range === '₹10L–₹30L')  return price >= 1000000 && price <= 3000000;
  if (range === 'Above ₹30L') return price > 3000000;
  return true;
}
const Tip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tip">
      <div className="chart-tip-label">{label || payload[0]?.payload?.brand || payload[0]?.payload?.city || payload[0]?.payload?.vehicle}</div>
      {payload.map((p,i) => (
        <div key={i}>{p.name}: <strong>{typeof p.value==='number' && /price|profit|value/i.test(p.name) ? fmtL(p.value) : p.value}</strong></div>
      ))}
    </div>
  );
};
function EmptyAnalytics({ setActiveScreen }) {
  return (
    <div className="empty-screen">
      <div className="home-empty-icon">
        <Icon name="chart" size={28} color="#94a3b8" strokeWidth={1.8} />
      </div>
      <div className="empty-title">No analytics yet</div>
      <div className="empty-sub">
        Analytics are powered by real evaluations. Run ML valuations to populate charts with live data.
      </div>
      <button className="btn btn-primary btn-lg" onClick={() => setActiveScreen('input')}>
        <Icon name="car" size={16} color="white" strokeWidth={2} />
        Run ML Evaluation
      </button>
    </div>
  );
}
export default function DashboardScreen() {
  const { evaluations, dashFilters, setDashFilters, setActiveScreen, clearEvaluations } = useApp();
  const [activeTab, setActiveTab]       = useState('Overview');
  const [confirmClear, setConfirmClear] = useState(false);
  const upd = (k, v) => setDashFilters(p => ({ ...p, [k]: v }));
  const brands    = useMemo(() => ['All',...Array.from(new Set(evaluations.map(v=>v.brand).filter(Boolean))).sort()], [evaluations]);
  const cities    = useMemo(() => ['All',...Array.from(new Set(evaluations.map(v=>v.city).filter(Boolean))).sort()], [evaluations]);
  const priceRanges = ['All','Under ₹5L','₹5L–₹10L','₹10L–₹30L','Above ₹30L'];
  const filtered = useMemo(() => evaluations.filter(v => {
    if (dashFilters.brand !== 'All' && v.brand !== dashFilters.brand) return false;
    if (dashFilters.city  !== 'All' && v.city  !== dashFilters.city)  return false;
    if (!price_ok(Number(v.marketValue||0), dashFilters.priceRange))  return false;
    return true;
  }), [evaluations, dashFilters]);
  const metrics = useMemo(() => {
    const count = filtered.length;
    const avgPrice  = count ? Math.round(filtered.reduce((s,v) => s + Number(v.marketValue||0), 0) / count) : 0;
    const avgMargin = count ? (filtered.reduce((s,v) => s + Number(v.marginPct||0), 0) / count).toFixed(1) : '0';
    const avgProfit = count ? Math.round(filtered.reduce((s,v) => s + Number(v.expectedProfit||0), 0) / count) : 0;
    const buyCount  = filtered.filter(v=>v.action==='BUY').length;
    const convRate  = count ? Math.round((buyCount/count)*100) : 0;
    // 1. Most Profitable Brand
    const brandProfitMap = {};
    filtered.forEach(v => {
      if (!v.brand || v.brand === 'Unknown') return;
      brandProfitMap[v.brand] = (brandProfitMap[v.brand] || 0) + Number(v.expectedProfit || 0);
    });
    let mostProfitableBrand = 'N/A';
    let maxBrandProfit = 0;
    Object.entries(brandProfitMap).forEach(([brand, profit]) => {
      if (profit > maxBrandProfit) {
        maxBrandProfit = profit;
        mostProfitableBrand = brand;
      }
    });
    // 2. Fastest Selling Segment
    const segmentLiquidityMap = {};
    const segmentCountMap = {};
    filtered.forEach(v => {
      const seg = v.segmentClass || 'economy';
      const liq = Number(v.resaleLiquidityScore || 50);
      segmentLiquidityMap[seg] = (segmentLiquidityMap[seg] || 0) + liq;
      segmentCountMap[seg] = (segmentCountMap[seg] || 0) + 1;
    });
    let fastestSellingSegment = 'N/A';
    let maxAvgLiquidity = -Infinity;
    Object.entries(segmentLiquidityMap).forEach(([seg, totalLiq]) => {
      const avgLiq = totalLiq / segmentCountMap[seg];
      if (avgLiq > maxAvgLiquidity) {
        maxAvgLiquidity = avgLiq;
        fastestSellingSegment = seg.charAt(0).toUpperCase() + seg.slice(1);
      }
    });
    // 3. High Risk Vehicles Count
    const highRiskCount = filtered.filter(v => Number(v.riskScore || 0) > 60).length;
    // 4. Average Confidence
    const avgConfidence = count ? Math.round(filtered.reduce((s,v) => s + Number(v.confidenceScore||0), 0) / count) : 0;
    const monthlyPipeline = filtered.reduce((s,v) => s + Number(v.buyPrice||0), 0);
    // Brand performance
    const brandMap = {};
    filtered.forEach(v => {
      if (!v.brand) return;
      brandMap[v.brand] ||= { brand:v.brand, count:0, totalVal:0, profit:0 };
      brandMap[v.brand].count++;
      brandMap[v.brand].totalVal += Number(v.marketValue||0);
      brandMap[v.brand].profit   += Number(v.expectedProfit||0);
    });
    const brandPerf = Object.values(brandMap)
      .map(b => ({ brand:b.brand, count:b.count, avgVal:Math.round(b.totalVal/b.count), avgProfit:Math.round(b.profit/b.count) }))
      .sort((a,b) => b.avgVal-a.avgVal)
      .slice(0,10);
    // City profitability
    const cityMap = {};
    filtered.forEach(v => {
      if (!v.city) return;
      cityMap[v.city] ||= { city:v.city, count:0, profit:0 };
      cityMap[v.city].count++;
      cityMap[v.city].profit += Number(v.expectedProfit||0);
    });
    const cityProf = Object.values(cityMap).map(c => ({...c, avgProfit:Math.round(c.profit/c.count)})).sort((a,b)=>b.avgProfit-a.avgProfit).slice(0,8);
    // Action distribution
    const actionDist = ['BUY','NEGOTIATE','REJECT','MANUAL REVIEW'].map(a => ({
      action: a.replace(' REVIEW',''), count: filtered.filter(v=>v.action===a).length,
    })).filter(a=>a.count>0);
    // Scatter: mileage vs price
    const scatter = filtered
      .filter(v => v.marketValue>0 && v.kmDriven>0)
      .map(v => ({ x:Math.round(Number(v.kmDriven||0)/1000), y:Math.round(Number(v.marketValue||0)/100000*10)/10, action:v.action }));
    // Profit histogram
    const profitBuckets = [
      { range:'<₹0',     min:-Infinity, max:0         },
      { range:'₹0-25K',  min:0,         max:25000     },
      { range:'₹25-50K', min:25000,     max:50000     },
      { range:'₹50-80K', min:50000,     max:80000     },
      { range:'₹80K+',   min:80000,     max:Infinity  },
    ].map(b => ({ range:b.range, count:filtered.filter(v=>{ const p=Number(v.expectedProfit||0); return p>b.min && p<=b.max; }).length }));
    return {
      count,
      avgPrice,
      avgMargin,
      avgProfit,
      buyCount,
      convRate,
      mostProfitableBrand,
      maxBrandProfit,
      fastestSellingSegment,
      highRiskCount,
      avgConfidence,
      monthlyPipeline,
      brandPerf,
      cityProf,
      actionDist,
      scatter,
      profitBuckets
    };
  }, [filtered]);
  if (evaluations.length === 0) {
    return (
      <div className="screen">
        <EmptyAnalytics setActiveScreen={setActiveScreen} />
      </div>
    );
  }
  return (
    <div className="screen screen-wide">
      <div className="page-header">
        <div>
          <div className="page-title">Analytics</div>
          <div className="page-subtitle">
            {filtered.length} evaluations · Live data from your ML valuations
          </div>
        </div>
        <div className="page-header-actions">
          <button className="btn btn-secondary btn-sm" onClick={() => setActiveScreen('input')}>
            + Evaluate
          </button>
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => exportEvaluationsToCSV(filtered.length > 0 ? filtered : evaluations)}
            title="Download all evaluation data as CSV"
            style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Export CSV
          </button>
          {evaluations.length > 0 && !confirmClear && (
            <button
              className="btn btn-danger btn-sm"
              onClick={() => setConfirmClear(true)}
            >
              Clear History
            </button>
          )}
          {confirmClear && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 13, color: 'var(--text-2)', fontWeight: 500 }}>Clear all data?</span>
              <button
                className="btn btn-danger btn-sm"
                onClick={() => { clearEvaluations(); setConfirmClear(false); }}
              >
                Yes, clear
              </button>
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => setConfirmClear(false)}
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>
      <div className="analytics-filters">
        <select className="filter-select" value={dashFilters.brand} onChange={e => upd('brand',e.target.value)}>
          {brands.map(b => <option key={b}>{b}</option>)}
        </select>
        <select className="filter-select" value={dashFilters.city} onChange={e => upd('city',e.target.value)}>
          {cities.map(c => <option key={c}>{c}</option>)}
        </select>
        <select className="filter-select" value={dashFilters.priceRange} onChange={e => upd('priceRange',e.target.value)}>
          {priceRanges.map(r => <option key={r}>{r}</option>)}
        </select>
      </div>
      <div className="kpi-grid">
        <div className="kpi-tile">
          <div className="kpi-tile-header">
            <div className="kpi-tile-label">Evaluations</div>
            <div className="kpi-icon" style={{ background:'#dbeafe' }}>
              <Icon name="clipboard" size={14} color="#2563eb" strokeWidth={2} />
            </div>
          </div>
          <div className="kpi-tile-value">{metrics.count}</div>
          <div className="kpi-tile-sub">{metrics.buyCount} BUY signals</div>
        </div>
        <div className="kpi-tile">
          <div className="kpi-tile-header">
            <div className="kpi-tile-label">Avg Market Value</div>
            <div className="kpi-icon" style={{ background:'#fff4f0' }}>
              <Icon name="trendUp" size={14} color="#f75d34" strokeWidth={2} />
            </div>
          </div>
          <div className="kpi-tile-value">{fmtL(metrics.avgPrice)}</div>
          <div className="kpi-tile-sub">Per valuation</div>
        </div>
        <div className="kpi-tile">
          <div className="kpi-tile-header">
            <div className="kpi-tile-label">Avg Profit</div>
            <div className="kpi-icon" style={{ background:'#dcfce7' }}>
              <Icon name="coins" size={14} color="#16a34a" strokeWidth={2} />
            </div>
          </div>
          <div className="kpi-tile-value" style={{ color:'#16a34a' }}>{fmtL(metrics.avgProfit)}</div>
          <div className="kpi-tile-sub">{metrics.avgMargin}% avg margin</div>
        </div>
        <div className="kpi-tile">
          <div className="kpi-tile-header">
            <div className="kpi-tile-label">BUY Rate</div>
            <div className="kpi-icon" style={{ background:'#f3e8ff' }}>
              <Icon name="shield" size={14} color="#7c3aed" strokeWidth={2} />
            </div>
          </div>
          <div className="kpi-tile-value">{metrics.convRate}%</div>
          <div className="kpi-tile-sub">Conversion signal rate</div>
        </div>
        <div className="kpi-tile">
          <div className="kpi-tile-header">
            <div className="kpi-tile-label">Most Profitable Brand</div>
            <div className="kpi-icon" style={{ background:'#fef3c7' }}>
              <Icon name="store" size={14} color="#d97706" strokeWidth={2} />
            </div>
          </div>
          <div className="kpi-tile-value" style={{ fontSize: '18px', padding: '3px 0' }}>{metrics.mostProfitableBrand}</div>
          <div className="kpi-tile-sub">Max profit cumulative</div>
        </div>
        <div className="kpi-tile">
          <div className="kpi-tile-header">
            <div className="kpi-tile-label">Fastest Segment</div>
            <div className="kpi-icon" style={{ background:'#ecfdf5' }}>
              <Icon name="lightning" size={14} color="#059669" strokeWidth={2} />
            </div>
          </div>
          <div className="kpi-tile-value" style={{ fontSize: '18px', padding: '3px 0' }}>{metrics.fastestSellingSegment}</div>
          <div className="kpi-tile-sub">Highest resale liquidity</div>
        </div>
        <div className="kpi-tile">
          <div className="kpi-tile-header">
            <div className="kpi-tile-label">High Risk Vehicles</div>
            <div className="kpi-icon" style={{ background:'#fef2f2' }}>
              <Icon name="warning" size={14} color="#dc2626" strokeWidth={2} />
            </div>
          </div>
          <div className="kpi-tile-value" style={{ color:'#dc2626' }}>{metrics.highRiskCount}</div>
          <div className="kpi-tile-sub">Score &gt; 60 risk factor</div>
        </div>
        <div className="kpi-tile">
          <div className="kpi-tile-header">
            <div className="kpi-tile-label">Avg Confidence</div>
            <div className="kpi-icon" style={{ background:'#e0f2fe' }}>
              <Icon name="brain" size={14} color="#0284c7" strokeWidth={2} />
            </div>
          </div>
          <div className="kpi-tile-value">{metrics.avgConfidence}%</div>
          <div className="kpi-tile-sub">ML uncertainty score</div>
        </div>
      </div>
      <div className="cd-card" style={{ marginBottom: 20, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '14px 20px' }}>
        <div>
          <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.4px' }}>Active Acquisition Pipeline</div>
          <div style={{ fontSize: '13px', color: 'var(--text-2)', marginTop: 2 }}>Monthly volume computed from active valuations</div>
        </div>
        <div style={{ fontSize: '22px', fontWeight: 800, color: 'var(--text-1)', letterSpacing: '-0.5px' }}>
          {fmtL(metrics.monthlyPipeline)}
        </div>
      </div>
      <div className="analytics-tabs">
        {TABS.map(t => (
          <button
            key={t}
            className={`analytics-tab ${activeTab===t?'active':''}`}
            onClick={() => setActiveTab(t)}
          >
            {t}
          </button>
        ))}
      </div>
      {activeTab === 'Overview' && (
        <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(320px,1fr))', gap:16 }}>
          <div className="chart-card">
            <div className="chart-card-title">Decision Distribution</div>
            <div className="chart-card-sub">BUY / NEGOTIATE / REJECT breakdown</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={metrics.actionDist}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="action" tick={{ fill:'#475569', fontSize:12 }} />
                <YAxis tick={{ fill:'#94a3b8', fontSize:11 }} />
                <Tooltip content={<Tip />} />
                <Bar dataKey="count" name="Count" radius={[6,6,0,0]}>
                  {metrics.actionDist.map((d,i) => (
                    <Cell key={i} fill={ACTION_COLORS[d.action==='REVIEW'?'MANUAL REVIEW':d.action] || '#94a3b8'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-card">
            <div className="chart-card-title">Profit Distribution</div>
            <div className="chart-card-sub">Number of deals by profit range</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={metrics.profitBuckets}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="range" tick={{ fill:'#475569', fontSize:11 }} />
                <YAxis tick={{ fill:'#94a3b8', fontSize:11 }} />
                <Tooltip content={<Tip />} />
                <Bar dataKey="count" name="Deals" fill="#16a34a" radius={[6,6,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          {metrics.scatter.length > 1 && (
            <div className="chart-card">
              <div className="chart-card-title">Odometer vs Market Value</div>
              <div className="chart-card-sub">Depreciation pattern across inventory</div>
              <ResponsiveContainer width="100%" height={220}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="x" name="Odometer (k km)" tick={{ fill:'#94a3b8', fontSize:11 }} />
                  <YAxis dataKey="y" name="Market Value (₹L)" tick={{ fill:'#94a3b8', fontSize:11 }} />
                  <Tooltip cursor={{ strokeDasharray:'3 3' }} />
                  <Scatter data={metrics.scatter} fill="#f75d34" opacity={0.7} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
      {activeTab === 'Brands' && (
        <div className="chart-card">
          <div className="chart-card-title">Brand Performance</div>
          <div className="chart-card-sub">Average market value by brand</div>
          <ResponsiveContainer width="100%" height={Math.max(200, metrics.brandPerf.length*40)}>
            <BarChart data={metrics.brandPerf} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
              <XAxis type="number" tick={{ fill:'#94a3b8', fontSize:11 }} tickFormatter={v=>`₹${(v/100000).toFixed(1)}L`} />
              <YAxis dataKey="brand" type="category" tick={{ fill:'#475569', fontSize:12 }} width={90} />
              <Tooltip content={<Tip />} />
              <Bar dataKey="avgVal" name="Avg Market Value" radius={[0,6,6,0]}>
                {metrics.brandPerf.map((_,i) => <Cell key={i} fill={COLORS[i%COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      {activeTab === 'Profit' && (
        <div style={{ display:'grid', gap:16 }}>
          <div className="chart-card">
            <div className="chart-card-title">City Profitability</div>
            <div className="chart-card-sub">Average dealer profit per city</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={metrics.cityProf}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="city" tick={{ fill:'#475569', fontSize:12 }} />
                <YAxis tick={{ fill:'#94a3b8', fontSize:11 }} tickFormatter={v=>`₹${Math.round(v/1000)}K`} />
                <Tooltip content={<Tip />} />
                <Bar dataKey="avgProfit" name="Avg Profit" fill="#16a34a" radius={[6,6,0,0]}>
                  {metrics.cityProf.map((_,i) => <Cell key={i} fill={COLORS[i%COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-card">
            <div className="chart-card-title">Brand Profit Ranking</div>
            <div className="chart-card-sub">Average net profit per brand</div>
            <ResponsiveContainer width="100%" height={Math.max(180, metrics.brandPerf.length*36)}>
              <BarChart data={metrics.brandPerf} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
                <XAxis type="number" tick={{ fill:'#94a3b8', fontSize:11 }} tickFormatter={v=>`₹${Math.round(v/1000)}K`} />
                <YAxis dataKey="brand" type="category" tick={{ fill:'#475569', fontSize:12 }} width={90} />
                <Tooltip content={<Tip />} />
                <Bar dataKey="avgProfit" name="Avg Profit" radius={[0,6,6,0]}>
                  {metrics.brandPerf.map((_,i) => <Cell key={i} fill={COLORS[i%COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
      {activeTab === 'Trends' && (
        <div className="chart-card">
          <div className="chart-card-title">Evaluation Activity</div>
          <div className="chart-card-sub">Cumulative evaluations over time</div>
          {(() => {
            const trendData = evaluations
              .slice()
              .reverse()
              .map((v, i) => ({
                idx: i + 1,
                marketValue: Math.round(Number(v.marketValue||0)/100000*10)/10,
                profit: Math.round(Number(v.expectedProfit||0)),
              }));
            return (
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={trendData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="idx" tick={{ fill:'#94a3b8', fontSize:11 }} />
                  <YAxis yAxisId="left" tick={{ fill:'#94a3b8', fontSize:11 }} tickFormatter={v=>`₹${v}L`} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fill:'#94a3b8', fontSize:11 }} tickFormatter={v=>`₹${Math.round(v/1000)}K`} />
                  <Tooltip content={<Tip />} />
                  <Area yAxisId="left" type="monotone" dataKey="marketValue" name="Market Value (₹L)" stroke="#2563eb" fill="#dbeafe" strokeWidth={2} />
                  <Area yAxisId="right" type="monotone" dataKey="profit" name="Profit (₹)" stroke="#16a34a" fill="#dcfce7" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            );
          })()}
        </div>
      )}
    </div>
  );
}
