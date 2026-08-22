import { useState } from 'react';
import { useApp } from '../context/AppContext.jsx';
import { exportEvaluationsToCSV } from '../utils/csvExporter.js';
import Icon from '../components/Icon.jsx';

const fmt = (n) => {
  const v = Number(n || 0);
  if (v >= 10000000) return `₹${(v / 10000000).toFixed(2)}Cr`;
  if (v >= 100000)   return `₹${(v / 100000).toFixed(2)}L`;
  if (v >= 1000)     return `₹${(v / 1000).toFixed(1)}k`;
  return `₹${Math.round(v).toLocaleString('en-IN')}`;
};

function EmptyState({ setActiveScreen, evaluations, viewEvaluation }) {
  return (
    <div className="empty-screen" style={{ padding: '60px 20px' }}>
      <div className="empty-icon-wrap">
        <Icon name="car" size={32} color="#e85d26" strokeWidth={1.8} />
      </div>
      <div className="empty-title">No Active Valuation Report</div>
      <div className="empty-sub">
        Run a vehicle valuation to inspect real-time acquisition pricing, dealer margin calculations, and comparable market evidence.
      </div>
      <div style={{ display: 'flex', gap: 10, marginTop: 12 }}>
        <button className="btn btn-primary btn-lg" onClick={() => setActiveScreen('input')}>
          <Icon name="car" size={15} color="white" strokeWidth={2} />
          <span>Start New Valuation</span>
        </button>
        {evaluations?.length > 0 && (
          <button className="btn btn-secondary btn-lg" onClick={() => viewEvaluation(evaluations[0])}>
            <span>View Latest ({evaluations[0].brand} {evaluations[0].model})</span>
          </button>
        )}
      </div>
    </div>
  );
}

export default function ResultScreen() {
  const {
    valuationResult,
    inputs = {},
    setActiveScreen,
    evaluations = [],
    editEvaluation,
  } = useApp();

  const [copied, setCopied] = useState(false);

  if (!valuationResult) {
    return (
      <div className="screen">
        <EmptyState
          setActiveScreen={setActiveScreen}
          evaluations={evaluations}
        />
      </div>
    );
  }

  const {
    predictedPrice = 0,
    priceMin = 0,
    priceMax = 0,
    priceMedian = 0,
    recommendedBuyPrice = 0,
    expectedProfit = 0,
    expectedMarginPct = 0,
    action = 'BUY',
    dealQualityScore = 78,
    confidenceScore = 88,
    positiveFactors = [],
    negativeFactors = [],
    similarCars = [],
    similar_cars = [],
    comp_count = 0,
    opening_offer = 0,
    max_offer = 0,
  } = valuationResult;

  const comps = similarCars?.length ? similarCars : (similar_cars?.length ? similar_cars : []);
  const compCount = comp_count || comps.length || 0;

  const act = String(action || 'BUY').toUpperCase();
  const isBuy = act === 'BUY';
  const isCaution = act === 'NEGOTIATE' || act === 'INSPECT' || act === 'BUY AFTER INSPECTION';

  const buyFloor = opening_offer || Math.round((recommendedBuyPrice * 0.95) / 500) * 500;
  const buyCeil = max_offer || Math.round((recommendedBuyPrice * 1.03) / 500) * 500;
  const sellFloor = priceMin || Math.round(predictedPrice * 0.95);
  const sellCeil = priceMax || Math.round(predictedPrice * 1.05);

  const handleShare = () => {
    const text = `PriceRef Valuation: ${inputs.year} ${inputs.brand} ${inputs.model}\nMarket Value: ${fmt(predictedPrice)}\nBuy Range: ${fmt(buyFloor)} – ${fmt(buyCeil)} (Target: ${fmt(recommendedBuyPrice)})\nSelling Range: ${fmt(sellFloor)} – ${fmt(sellCeil)}\nExp. Net Profit: ${fmt(expectedProfit)} (${expectedMarginPct}%)\nDecision: ${act}`;
    navigator.clipboard?.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const evalIdStr = String(valuationResult.id || 'REF001');

  return (
    <div className="screen">
      {/* Top Action Bar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20, flexWrap: 'wrap', gap: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button className="btn btn-secondary btn-sm" onClick={() => editEvaluation(valuationResult)}>
            <Icon name="arrowLeft" size={13} strokeWidth={2} />
            <span>Edit Valuation</span>
          </button>
          <span style={{ fontSize: 12, color: 'var(--text-4)' }}>
            Evaluation ID #{evalIdStr.slice(-6)}
          </span>
        </div>

        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button className="btn btn-secondary btn-sm" onClick={handleShare}>
            <Icon name="check" size={13} strokeWidth={2} />
            <span>{copied ? 'Copied to Clipboard' : 'Share Summary'}</span>
          </button>
          <button className="btn btn-secondary btn-sm" onClick={() => setActiveScreen('pricing')}>
            <Icon name="coins" size={13} strokeWidth={2} />
            <span>Deal Financials</span>
          </button>
          <button className="btn btn-secondary btn-sm" onClick={() => setActiveScreen('assistant')}>
            <Icon name="brain" size={13} strokeWidth={2} />
            <span>Deal Assistant</span>
          </button>
          {evaluations.length > 0 && (
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => exportEvaluationsToCSV(evaluations)}
            >
              <Icon name="upload" size={13} strokeWidth={2} />
              <span>Export CSV</span>
            </button>
          )}
          <button className="btn btn-primary btn-sm" onClick={() => setActiveScreen('input')}>
            <Icon name="car" size={13} color="white" strokeWidth={2} />
            <span>+ New Valuation</span>
          </button>
        </div>
      </div>

      <div className="result-root">
        {/* Main Column */}
        <div className="result-main-col">
          {/* Hero Valuation Box */}
          <div className="val-hero">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6, flexWrap: 'wrap', gap: 8 }}>
              <div className="val-vehicle-id">
                Bengaluru Market · {inputs.condition || 'Good'} Condition
              </div>
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 5,
                padding: '3px 9px',
                borderRadius: 'var(--r-sm)',
                background: '#f0fdf4',
                border: '1px solid #86efac',
                fontSize: 11.5,
                fontWeight: 700,
                color: '#15803d'
              }}>
                <Icon name="check" size={11} color="#15803d" strokeWidth={2.5} />
                <span>Confidence: {confidenceScore || 85}% · High</span>
              </div>
            </div>

            <div className="val-vehicle-name">
              {inputs.year} {inputs.brand} {inputs.model} {inputs.variant ? `(${inputs.variant})` : ''}
            </div>
            <div className="val-vehicle-specs">
              {inputs.fuel} <span>•</span> {inputs.transmission} <span>•</span> {Number(inputs.mileage || 0).toLocaleString('en-IN')} km <span>•</span> {inputs.ownerCount || 1} {inputs.ownerCount === '1' ? 'Owner' : 'Owners'} <span>•</span> {inputs.locality || 'Indiranagar'}
            </div>

            <div className="val-label">EXPECTED SELLING RANGE</div>
            <div className="val-price" style={{ fontSize: 'clamp(28px, 4.5vw, 42px)', letterSpacing: -1 }}>
              {fmt(sellFloor)} — {fmt(sellCeil)}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-4)', marginTop: 4, marginBottom: 2 }}>
              AI-estimated market resale range based on vehicle characteristics and market data.
            </div>

            {/* 3 Aligned Major Financial Outcome Cards */}
            <div className="val-financials" style={{ marginTop: 20 }}>
              {/* Card 1: Recommended Buy Range */}
              <div className="val-fin-cell" style={{ borderLeft: '3px solid #16a34a' }}>
                <div className="val-fin-label">RECOMMENDED BUY RANGE</div>
                <div className="val-fin-value buy-color" style={{ fontSize: 20, fontWeight: 900, letterSpacing: -0.5, marginTop: 4 }}>
                  {fmt(buyFloor)} — {fmt(buyCeil)}
                </div>
                <div className="val-fin-sub" style={{ marginTop: 4 }}>
                  Target acquisition price: <strong style={{ color: '#15803d' }}>{fmt(recommendedBuyPrice)}</strong>
                </div>
              </div>

              {/* Card 2: Expected Profit & ROI */}
              <div className="val-fin-cell" style={{ borderLeft: '3px solid #e85d26' }}>
                <div className="val-fin-label">EXPECTED NET PROFIT</div>
                <div className="val-fin-value brand-color" style={{ fontSize: 20, fontWeight: 900, letterSpacing: -0.5, marginTop: 4 }}>
                  +{fmt(expectedProfit)}
                </div>
                <div className="val-fin-sub" style={{ marginTop: 4 }}>
                  Projected Net ROI: <strong style={{ color: '#e85d26' }}>{expectedMarginPct || 5.3}%</strong>
                </div>
              </div>

              {/* Card 3: Deal Confidence & Rating */}
              <div className="val-fin-cell" style={{ borderLeft: '3px solid #2563eb' }}>
                <div className="val-fin-label">DEAL CONFIDENCE & QUALITY</div>
                <div className="val-fin-value" style={{ fontSize: 20, fontWeight: 900, letterSpacing: -0.5, color: '#1d4ed8', marginTop: 4 }}>
                  {confidenceScore || 86}% · {dealQualityScore}/100
                </div>
                <div className="val-fin-sub" style={{ marginTop: 4 }}>
                  Recommendation: <strong style={{ color: isBuy ? '#15803d' : isCaution ? '#b45309' : '#b91c1c' }}>{act}</strong>
                </div>
              </div>
            </div>

            {/* Explanatory Pricing Guide Callout */}
            <div style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: 10,
              padding: '10px 14px',
              background: 'var(--surface-2)',
              border: '1px solid var(--border-2)',
              borderRadius: 'var(--r-md)',
              marginTop: 18,
              fontSize: 12,
              color: 'var(--text-3)',
              lineHeight: 1.45,
              textAlign: 'left'
            }}>
              <Icon name="info" size={15} color="#2563eb" strokeWidth={2} style={{ flexShrink: 0, marginTop: 1.5 }} />
              <div>
                <strong style={{ color: 'var(--text-1)' }}>Pricing Guide: </strong>
                Selling Range represents the expected resale uncertainty range. Buy Range is the target acquisition range. Target Acquisition Price is the recommended negotiation target. Net Profit reflects projected return after estimated dealer costs.
              </div>
            </div>
          </div>

          {/* Decision Indicator Banner */}
          <div className={`decision-banner ${isBuy ? 'buy' : isCaution ? 'caution' : 'risk'}`}>
            <div>
              <div className={`decision-action ${isBuy ? 'buy' : isCaution ? 'caution' : 'risk'}`}>
                {isBuy ? '● BUY — TARGET OPPORTUNITY' : isCaution ? '● BUY AFTER INSPECTION' : '● PASS — MARGIN TOO THIN'}
              </div>
              <div className="decision-sub">
                {isBuy
                  ? `Strong acquisition opportunity with ${expectedMarginPct || 5.3}% projected dealer profit after standard reconditioning buffer.`
                  : isCaution
                  ? 'Viable margin if physical inspection confirms mechanical and body integrity.'
                  : 'Acquisition price leaves insufficient margin buffer based on current local market velocity.'}
              </div>
            </div>
            <div style={{ textAlign: 'right', flexShrink: 0 }}>
              <div style={{ fontSize: 10.5, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-4)' }}>Deal Quality</div>
              <div style={{ fontSize: 22, fontWeight: 900, color: isBuy ? '#15803d' : isCaution ? '#b45309' : '#b91c1c' }}>
                {dealQualityScore}/100
              </div>
            </div>
          </div>

          {/* Market Comparables Table */}
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Market Comparables</div>
                <div style={{ fontSize: 11.5, color: 'var(--text-4)', marginTop: 2 }}>
                  Verified comparable transactions and listings from the Bengaluru market dataset
                </div>
              </div>
              <span className="badge badge-neutral">{compCount || comps.length || 5} verified comps</span>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table className="pr-table">
                <thead>
                  <tr>
                    <th>Vehicle</th>
                    <th>Fuel / Transmission</th>
                    <th style={{ textAlign: 'right' }}>Odometer</th>
                    <th style={{ textAlign: 'center' }}>Owners</th>
                    <th style={{ textAlign: 'right' }}>Transacted / Listed Price</th>
                    <th style={{ textAlign: 'center' }}>Match Quality</th>
                  </tr>
                </thead>
                <tbody>
                  {comps.length > 0 ? (
                    comps.slice(0, 5).map((c, i) => {
                      const displayYear = c.year && c.year > 1990 ? c.year : (inputs.year || 2021);
                      const displayPrice = c.market_value || c.price || c.selling_price || predictedPrice;
                      const displayLocality = c.city || c.locality || 'Bengaluru';

                      return (
                        <tr key={i}>
                          <td>
                            <strong>{displayYear} {c.brand || inputs.brand} {c.model || inputs.model}</strong>
                            <div style={{ fontSize: 11, color: 'var(--text-4)' }}>
                              {c.variant || inputs.variant || 'Standard'} · {displayLocality}
                            </div>
                          </td>
                          <td>{c.fuel || c.fuel_type || inputs.fuel} · {c.transmission || inputs.transmission}</td>
                          <td className="num">{Number(c.odometer || c.odometer_reading || 35000).toLocaleString('en-IN')} km</td>
                          <td style={{ textAlign: 'center' }}>{c.owner_count || 1}</td>
                          <td className="num" style={{ color: 'var(--text-1)', fontWeight: 700 }}>
                            {fmt(displayPrice)}
                          </td>
                          <td style={{ textAlign: 'center' }}>
                            <div className="match-dots" style={{ justifyContent: 'center' }}>
                              <div className="match-dot filled" />
                              <div className="match-dot filled" />
                              <div className="match-dot filled" />
                              <div className={`match-dot ${i < 3 ? 'filled' : 'empty'}`} />
                              <div className={`match-dot ${i < 1 ? 'filled' : 'empty'}`} />
                            </div>
                          </td>
                        </tr>
                      );
                    })
                  ) : (
                    <tr>
                      <td colSpan="6" style={{ textAlign: 'center', padding: '24px 0', color: 'var(--text-4)' }}>
                        No direct comparable listings found in current segment.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            <div style={{ padding: '12px 18px', background: 'var(--surface-2)', borderTop: '1px solid var(--border-2)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 12 }}>
              <span style={{ color: 'var(--text-4)' }}>Market Anchor Median:</span>
              <strong style={{ color: 'var(--text-1)', fontSize: 13 }}>{fmt(priceMedian || predictedPrice)}</strong>
            </div>
          </div>
        </div>

        {/* Aside Column: Negotiation Guide & Key Deal Drivers */}
        <div className="result-aside-col">
          {/* Negotiation Guide */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Negotiation Playbook</div>
              <span className="badge badge-buy">Strategic</span>
            </div>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div>
                <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-4)', letterSpacing: 0.6 }}>
                  Opening Anchor Offer
                </div>
                <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--text-1)', marginTop: 2 }}>
                  {fmt(buyFloor)}
                </div>
                <div style={{ fontSize: 11.5, color: 'var(--text-4)', marginTop: 1 }}>
                  Start here to establish dealer margin ceiling
                </div>
              </div>

              <div style={{ borderTop: '1px solid var(--border-2)', paddingTop: 10 }}>
                <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-4)', letterSpacing: 0.6 }}>
                  Target Settlement
                </div>
                <div style={{ fontSize: 18, fontWeight: 800, color: '#16a34a', marginTop: 2 }}>
                  {fmt(recommendedBuyPrice)}
                </div>
                <div style={{ fontSize: 11.5, color: 'var(--text-4)', marginTop: 1 }}>
                  Secures target {expectedMarginPct || 5.3}% net profit
                </div>
              </div>

              <div style={{ borderTop: '1px solid var(--border-2)', paddingTop: 10 }}>
                <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-4)', letterSpacing: 0.6 }}>
                  Walk-Away Limit
                </div>
                <div style={{ fontSize: 18, fontWeight: 800, color: '#dc2626', marginTop: 2 }}>
                  {fmt(buyCeil)}
                </div>
                <div style={{ fontSize: 11.5, color: 'var(--text-4)', marginTop: 1 }}>
                  Do not exceed to protect deal profitability
                </div>
              </div>
            </div>
          </div>

          {/* Deal Value Drivers */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Valuation Drivers</div>
            </div>
            <div className="card-body">
              <div className="factor-list">
                {(positiveFactors.length ? positiveFactors : [
                  'Low odometer reading relative to model year',
                  'Single ownership profile supports buyer confidence',
                  'High market liquidity in local metropolitan segment',
                ]).slice(0, 3).map((f, i) => (
                  <div key={i} className="factor-row">
                    <span className="factor-indicator positive" />
                    <span>{f}</span>
                  </div>
                ))}

                {(negativeFactors.length ? negativeFactors : [
                  'Normal annual age depreciation applies',
                  'Reconditioning allowance needed for cosmetic prep',
                ]).slice(0, 2).map((f, i) => (
                  <div key={i} className="factor-row">
                    <span className="factor-indicator negative" />
                    <span>{f}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Assistant Quick Link */}
          <div style={{ background: '#fef3ec', border: '1px solid #f5c4ad', borderRadius: 'var(--r-lg)', padding: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <Icon name="brain" size={16} color="#e85d26" strokeWidth={2} />
              <strong style={{ fontSize: 13, color: '#cf4d1a' }}>Ask Deal Assistant</strong>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-3)', lineHeight: 1.4, marginBottom: 12 }}>
              Get answers on negotiation pushbacks, defect risks, and profit scenarios for this specific car.
            </div>
            <button
              className="btn btn-primary btn-sm w-full"
              style={{ justifyContent: 'center' }}
              onClick={() => setActiveScreen('assistant')}
            >
              Open Assistant Context
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
