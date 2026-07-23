import { useState } from 'react';
import { useApp } from '../context/AppContext.jsx';
import { CAR_IMAGES } from '../utils/mockData.js';
import { formatINR, getSeasonalContext } from '../utils/format.js';
import { DEAL_HEALTH_META, GRADE_OPTIONS } from '../utils/wheelrCosts.js';
import { NegotiationPlaybook, ExpandableBreakdownTable } from '../components/WheelrPanels.jsx';
import Icon from '../components/Icon.jsx';

function actionClass(action) {
  const a = String(action || '').toUpperCase();
  if (a === 'BUY') return 'buy';
  if (a === 'NEGOTIATE') return 'negotiate';
  if (a === 'REJECT' || a === 'PASS') return 'reject';
  return 'review';
}

function gradeLabel(category, value) {
  return GRADE_OPTIONS[category]?.find(o => o.value === value)?.label || value;
}

// IDV Comparison Banner
function IDVBanner({ idvAnalysis }) {
  if (!idvAnalysis) return null;
  const { idv_value, ml_value, idv_gap_pct, flag, flag_type } = idvAnalysis;
  const sign = idv_gap_pct >= 0 ? '+' : '';

  let bannerClass = 'neutral';
  if (flag_type === 'warning') bannerClass = 'warning';
  else if (flag_type === 'positive') bannerClass = 'positive';

  return (
    <div className={`idv-banner ${bannerClass}`}>
      <div className="label-xs" style={{ marginBottom: 6 }}>
        IDV vs ML Valuation
      </div>
      <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-1)', marginBottom: 4 }}>
        IDV: {formatINR(idv_value)} &nbsp;·&nbsp; ML Value: {formatINR(ml_value)} &nbsp;·&nbsp;
        <span style={{ color: flag_type === 'warning' ? 'var(--warning)' : flag_type === 'positive' ? 'var(--success)' : 'var(--text-2)', fontWeight: 800 }}>
          Gap: {sign}{idv_gap_pct}%
        </span>
      </div>
      <div style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 6, fontWeight: 500 }}>
        <Icon
          name={flag_type === 'warning' ? 'warning' : flag_type === 'positive' ? 'check' : 'clipboard'}
          size={14}
          color={flag_type === 'warning' ? 'var(--warning)' : flag_type === 'positive' ? 'var(--success)' : 'var(--text-3)'}
          strokeWidth={2.2}
        />
        {flag}
      </div>
      {flag_type === 'warning' && (
        <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 4 }}>
          ₹15,000 additional risk deduction applied to max buy price
        </div>
      )}
    </div>
  );
}

// Connector arrow between pipeline steps
function PipelineArrow() {
  return (
    <div className="pipeline-connector">
      <span className="pipeline-connector-arrow">▼</span>
    </div>
  );
}

// Single pipeline step
function PipelineStep({ icon, iconClass, bodyClass, label, amount, amountClass, barPct, barClass, subRows }) {
  return (
    <div className="pipeline-step">
      <div className={`pipeline-icon ${iconClass}`}>
        {icon}
      </div>
      <div className={`pipeline-body ${bodyClass || ''}`}>
        <div className="pipeline-step-top">
          <div className="pipeline-step-label">{label}</div>
          <div className={`pipeline-step-amount ${amountClass}`}>{amount}</div>
        </div>
        {subRows && subRows.length > 0 && (
          <div className="pipeline-sub-rows">
            {subRows.map((row, i) => (
              <div key={i} className="pipeline-sub-row">
                <span>{row.label}</span>
                <strong style={{ color: row.color || undefined }}>{row.value}</strong>
              </div>
            ))}
          </div>
        )}
        {barPct !== undefined && (
          <div className="pipeline-bar-wrap">
            <div className="pipeline-bar-track">
              <div className={`pipeline-bar-fill ${barClass}`} style={{ width: `${Math.min(100, Math.max(2, barPct))}%` }} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Main Pricing Pipeline section
function RuleBasedPricingPipeline({ result, inputs }) {
  const {
    predictedPrice,
    recommendedBuyPrice,
    recommendedSellPrice,
    expectedProfit,
    expectedMarginPct,
    targetMarginPct,
    riskScore,
    recon,
    wheelrRisk,
    holdingCost,
    riskBuffer,
    repairBuffer,
    dealHealth,
  } = result;

  const mlBasePrice = predictedPrice || 0;
  const reconTotal = recon?.total || repairBuffer || 0;
  const wheelrRiskTotal = wheelrRisk?.total || 0;
  const hCost = holdingCost || Math.round(mlBasePrice * 0.025);
  const rBuf = riskBuffer || Math.round(mlBasePrice * (riskScore / 100) * 0.08);
  const targetProfit = Math.round(mlBasePrice * ((targetMarginPct || 10) / 100));
  const finalBuyPrice = recommendedBuyPrice || 0;
  // Sell price must always be above buy price — guard against inversion
  const rawSellPrice = recommendedSellPrice || Math.round(mlBasePrice * 1.05);
  const sellPrice = rawSellPrice > finalBuyPrice ? rawSellPrice : Math.round(finalBuyPrice * 1.10 / 500) * 500;
  const marginPct = expectedMarginPct || 0;
  const targetMPct = targetMarginPct || 10;

  const healthColor = dealHealth === 'green' ? 'green' : dealHealth === 'yellow' ? 'yellow' : 'red';
  const healthMsg = DEAL_HEALTH_META[dealHealth]?.title || 'Deal health unknown';

  const reconBreakdown = recon?.breakdown || {};
  const reconSubRows = [
    reconBreakdown.engine > 0 && { label: '🔧 Engine', value: formatINR(reconBreakdown.engine), color: 'var(--danger)' },
    reconBreakdown.tyres > 0 && { label: '🔄 Tyres', value: formatINR(reconBreakdown.tyres), color: 'var(--danger)' },
    reconBreakdown.body_paint > 0 && { label: '🎨 Body & Paint', value: formatINR(reconBreakdown.body_paint), color: 'var(--danger)' },
    reconBreakdown.interior > 0 && { label: '🪑 Interior', value: formatINR(reconBreakdown.interior), color: 'var(--danger)' },
    reconBreakdown.electricals > 0 && { label: '⚡ Electricals', value: formatINR(reconBreakdown.electricals), color: 'var(--danger)' },
    { label: '📋 RC + detailing + ops', value: formatINR(recon?.rc_transfer_cost != null ? reconBreakdown.fixed : (reconBreakdown.fixed || 8000)), color: 'var(--text-3)' },
  ].filter(Boolean);

  const riskBreakdown = wheelrRisk?.breakdown || {};
  const riskSubRows = [
    riskBreakdown.owner_deduction > 0 && { label: `👤 Owner #${inputs.ownerCount}`, value: `−${formatINR(riskBreakdown.owner_deduction)}`, color: 'var(--danger)' },
    riskBreakdown.km_deduction > 0 && { label: '🛣️ High odometer', value: `−${formatINR(riskBreakdown.km_deduction)}`, color: 'var(--danger)' },
    riskBreakdown.accident_deduction > 0 && { label: '💥 Accident history', value: `−${formatINR(riskBreakdown.accident_deduction)}`, color: 'var(--danger)' },
    riskBreakdown.state_deduction > 0 && { label: '🗺️ Out-of-state', value: `−${formatINR(riskBreakdown.state_deduction)}`, color: 'var(--danger)' },
    riskBreakdown.loan_deduction > 0 && { label: '🏦 Loan outstanding', value: `−${formatINR(riskBreakdown.loan_deduction)}`, color: 'var(--danger)' },
  ].filter(Boolean);
  if (riskSubRows.length === 0) riskSubRows.push({ label: '✔ No risk deductions', value: '₹0', color: 'var(--success)' });

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div className="pipeline-section-head">
        <div>
          <div className="pipeline-section-title">Pricing Pipeline</div>
          <div className="pipeline-section-sub">Detailed breakdown of rule-based adjustments</div>
        </div>
      </div>

      <div className={`pipeline-health-row ${healthColor}`} style={{ marginBottom: 14 }}>
        <Icon
          name={dealHealth === 'red' ? 'warning' : 'check'}
          size={14}
          color={dealHealth === 'red' ? 'var(--danger)' : dealHealth === 'yellow' ? 'var(--warning)' : 'var(--success)'}
          strokeWidth={2.2}
        />
        <span>{healthMsg}</span>
      </div>

      <div className="pricing-pipeline">
        <PipelineStep
          icon="🤖" iconClass="start" bodyClass="start-body"
          label="ML Market Value"
          amount={formatINR(mlBasePrice)} amountClass="blue"
          barPct={100} barClass="blue"
        />
        <PipelineArrow />
        <PipelineStep
          icon="🔧" iconClass="deduct"
          label="Recon Cost"
          amount={`−${formatINR(reconTotal)}`} amountClass="red"
          barPct={(reconTotal / mlBasePrice) * 100} barClass="red"
          subRows={reconSubRows}
        />
        <PipelineArrow />
        <PipelineStep
          icon="⚠️" iconClass="deduct"
          label="Risk Deductions"
          amount={`−${formatINR(wheelrRiskTotal)}`} amountClass="red"
          barPct={(wheelrRiskTotal / mlBasePrice) * 100} barClass="red"
          subRows={riskSubRows}
        />
        <PipelineArrow />
        <PipelineStep
          icon="📦" iconClass="deduct"
          label="Holding Cost"
          amount={`−${formatINR(hCost)}`} amountClass="orange"
          barPct={(hCost / mlBasePrice) * 100} barClass="orange"
        />
        <PipelineArrow />
        <PipelineStep
          icon="🛡️" iconClass="deduct"
          label="Risk Buffer"
          amount={`−${formatINR(rBuf)}`} amountClass="red"
          barPct={(rBuf / mlBasePrice) * 100} barClass="red"
        />
        <PipelineArrow />
        <PipelineStep
          icon="💰" iconClass="margin"
          label={`Target Profit (${targetMPct}%)`}
          amount={`−${formatINR(targetProfit)}`} amountClass="amber"
          barPct={targetMPct} barClass="amber"
          subRows={[
            { label: '💵 Profit at sell', value: formatINR(expectedProfit || targetProfit), color: 'var(--success)' },
            { label: '📈 Actual margin', value: `${marginPct.toFixed(1)}%`, color: marginPct >= targetMPct ? 'var(--success)' : 'var(--danger)' },
          ]}
        />
      </div>

      <div className="pipeline-final-box">
        <div className="pipeline-final-left">
          <div className="pipeline-final-label">Recommended Buy Price</div>
          <div className="pipeline-final-price">{formatINR(finalBuyPrice)}</div>
          <div className="pipeline-final-sub">Sell at {formatINR(sellPrice)} · Net profit {formatINR(expectedProfit)}</div>
        </div>
        <div className="pipeline-final-right">
          <div className="pipeline-final-margin-label">Actual Margin</div>
          <div className="pipeline-final-margin-pct" style={{ color: 'var(--success)' }}>{marginPct.toFixed(1)}%</div>
          <div className="pipeline-final-margin-tag">Target: {targetMPct}%</div>
        </div>
      </div>
    </div>
  );
}

// Main Enhanced Result Screen
export default function EnhancedResultScreen() {
  const { enhancedResult, inputs, setActiveScreen, isLoading } = useApp();
  const [showNegotiation, setShowNegotiation] = useState(true);
  const carImage = CAR_IMAGES[`${inputs.brand} ${inputs.model}`] || '/cars/placeholder.png';

  if (isLoading) {
    return (
      <div className="screen loading-screen">
        <div className="loading-spinner" />
        <div className="loading-label">Running enhanced evaluation…</div>
      </div>
    );
  }

  if (!enhancedResult) {
    return (
      <div className="screen empty-screen">
        <img src={carImage} alt="Car" style={{ width: '50%', opacity: 0.25, margin: '0 auto 16px', display: 'block' }} />
        <h2 className="empty-title">No enhanced result yet</h2>
        <p className="empty-sub">Run enhanced valuation with detailed multi-point inspection checks to see the premium breakdown.</p>
        <button className="btn btn-primary btn-lg" onClick={() => setActiveScreen('enhanced-input')}>
          <Icon name="zap" size={16} color="white" strokeWidth={2} />
          Enhanced Valuation
        </button>
      </div>
    );
  }

  const {
    predictedPrice, recommendedBuyPrice, action, confidenceScore,
    enhancedMaxBuyPrice, recon, wheelrRisk, negotiation,
    idvAnalysis, segmentClass, routingNote,
  } = enhancedResult;

  const inspection = enhancedResult.inspection || {};
  const reconRows = [
    { Category: 'Engine', Grade: gradeLabel('engine', inspection.engineGrade || 'good'), Type: inspection.vendorType?.engine || 'vendor', Cost: recon?.breakdown?.engine || 0 },
    { Category: 'Tyres', Grade: gradeLabel('tyre', inspection.tyreGrade || 'good'), Type: inspection.vendorType?.tyre || 'vendor', Cost: recon?.breakdown?.tyres || 0 },
    { Category: 'Body & Paint', Grade: gradeLabel('body', inspection.bodyGrade || 'clean'), Type: inspection.vendorType?.body || 'vendor', Cost: recon?.breakdown?.body_paint || 0 },
    { Category: 'Interior', Grade: gradeLabel('interior', inspection.interiorGrade || 'clean'), Type: inspection.vendorType?.interior || 'vendor', Cost: recon?.breakdown?.interior || 0 },
    { Category: 'Electricals', Grade: gradeLabel('electrical', inspection.electricalGrade || 'all_good'), Type: inspection.vendorType?.electrical || 'vendor', Cost: recon?.breakdown?.electricals || 0 },
    { Category: 'Fixed costs', Grade: '—', Type: '—', Cost: recon?.breakdown?.fixed || recon?.fixed_cost || 0 },
  ];

  const riskRows = [
    { Factor: 'Owner no.', Value: inputs.ownerCount, Deduction: wheelrRisk?.breakdown?.owner_deduction || 0 },
    { Factor: 'Odometer', Value: `${Number(inputs.mileage || 0).toLocaleString('en-IN')} km`, Deduction: wheelrRisk?.breakdown?.km_deduction || 0 },
    { Factor: 'Accident', Value: inspection.accidentHistory || 'none', Deduction: wheelrRisk?.breakdown?.accident_deduction || 0 },
    { Factor: 'State', Value: inspection.registrationState || '—', Deduction: wheelrRisk?.breakdown?.state_deduction || 0 },
    { Factor: 'Loan', Value: inspection.loanOutstanding ? 'Yes' : 'No', Deduction: wheelrRisk?.breakdown?.loan_deduction || 0 },
  ];

  const aClass = actionClass(action);
  const actionColor = aClass === 'buy' ? 'var(--success)' : aClass === 'negotiate' ? 'var(--warning)' : aClass === 'reject' ? 'var(--danger)' : 'var(--text-3)';

  return (
    <div className="screen enhanced-screen">
      <div className="page-header">
        <div>
          <div className="page-title">Enhanced Evaluation Results</div>
          <div className="page-subtitle">Detailed multi-point risk and cost assessment</div>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={() => setActiveScreen('enhanced-input')}>
          ← Back
        </button>
      </div>

      {/* ── IDV Banner (only when IDV was provided) ── */}
      <IDVBanner idvAnalysis={idvAnalysis} />

      {/* ── Action + Key Numbers ── */}
      <div className="card" style={{
        background: `linear-gradient(135deg, ${actionColor}0a 0%, ${actionColor}03 100%)`,
        border: `2px solid ${actionColor}30`,
        padding: '20px',
        marginBottom: 16,
      }}>
        <div style={{ display: 'flex', justifycontent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div>
            <div className="label-xs" style={{ marginBottom: 6 }}>
              Dealer Recommendation
            </div>
            <div className={`action-badge ${aClass}`} style={{ fontSize: 16, padding: '8px 16px' }}>{action}</div>
          </div>
          <div style={{ textAlign: 'right', display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 }}>
            <div className="label-xs">Confidence & Routing</div>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              {segmentClass && (
                <span className={`segment-badge ${segmentClass}`}>
                  {segmentClass.toUpperCase()}
                </span>
              )}
              <span className={`confidence-pill ${confidenceScore >= 75 ? 'good' : confidenceScore >= 55 ? 'medium' : 'bad'}`}>
                {confidenceScore}%
              </span>
            </div>
            {routingNote && (
              <div style={{ fontSize: 11, color: 'var(--text-3)', fontStyle: 'italic', marginTop: 2 }}>
                {routingNote}
              </div>
            )}
          </div>
        </div>

        {/* ML value row */}
        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, padding: '12px 14px', marginBottom: 12 }}>
          <div className="label-xs" style={{ marginBottom: 4 }}>ML Market Value</div>
          <div style={{ fontSize: 20, fontWeight: 800, color: 'var(--info)' }}>{formatINR(predictedPrice)}</div>
        </div>

        {/* Buy & Sell Range Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          {/* Buy Range */}
          <div style={{
            background: 'linear-gradient(135deg, rgba(34,197,94,0.08) 0%, rgba(34,197,94,0.03) 100%)',
            border: '1.5px solid rgba(34,197,94,0.25)',
            borderRadius: 10,
            padding: '14px 12px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 10 }}>
              <div style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--success)' }} />
              <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', color: 'var(--success)', textTransform: 'uppercase' }}>Buy Range</div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
              <div>
                <div style={{ fontSize: 9, color: 'var(--text-3)', marginBottom: 2 }}>Open at</div>
                <div style={{ fontSize: 14, fontWeight: 800, color: 'var(--success)' }}>
                  {formatINR(negotiation?.opening_offer || Math.round((enhancedMaxBuyPrice || recommendedBuyPrice) * 0.95))}
                </div>
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-3)', alignSelf: 'center', padding: '0 4px' }}>→</div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: 9, color: 'var(--text-3)', marginBottom: 2 }}>Max pay</div>
                <div style={{ fontSize: 14, fontWeight: 800, color: 'var(--warning)' }}>
                  {formatINR(enhancedMaxBuyPrice || recommendedBuyPrice)}
                </div>
              </div>
            </div>
            <div style={{ marginTop: 8, height: 3, borderRadius: 2, background: 'linear-gradient(90deg, var(--success), var(--warning))' }} />
          </div>

          {/* Sell Range */}
          <div style={{
            background: 'linear-gradient(135deg, rgba(14,165,233,0.08) 0%, rgba(14,165,233,0.03) 100%)',
            border: '1.5px solid rgba(14,165,233,0.25)',
            borderRadius: 10,
            padding: '14px 12px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 10 }}>
              <div style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--info)' }} />
              <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', color: 'var(--info)', textTransform: 'uppercase' }}>Sell Range</div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
              <div>
                <div style={{ fontSize: 9, color: 'var(--text-3)', marginBottom: 2 }}>Min sell</div>
                <div style={{ fontSize: 14, fontWeight: 800, color: 'var(--info)' }}>
                  {formatINR(enhancedResult.recommendedSellPrice || Math.round(predictedPrice * 1.03))}
                </div>
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-3)', alignSelf: 'center', padding: '0 4px' }}>→</div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: 9, color: 'var(--text-3)', marginBottom: 2 }}>Ideal sell</div>
                <div style={{ fontSize: 14, fontWeight: 800, color: 'var(--accent)' }}>
                  {formatINR(Math.round((enhancedResult.recommendedSellPrice || predictedPrice) * 1.05))}
                </div>
              </div>
            </div>
            <div style={{ marginTop: 8, height: 3, borderRadius: 2, background: 'linear-gradient(90deg, var(--info), var(--accent))' }} />
          </div>
        </div>
      </div>

      {/* ── Inspection Deductions summary cards ── */}
      <div className="enhanced-result-cards" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
        <div className="card" style={{ borderLeft: '3px solid var(--warning)' }}>
          <div className="label-xs" style={{ marginBottom: 8 }}>Reconditioning Cost</div>
          <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--text-1)' }}>{formatINR(recon?.total)}</div>
        </div>
        <div className="card" style={{ borderLeft: '3px solid var(--danger)' }}>
          <div className="label-xs" style={{ marginBottom: 8 }}>Risk Deductions</div>
          <div style={{ fontSize: 18, fontWeight: 800, color: 'var(--text-1)' }}>{formatINR(wheelrRisk?.total)}</div>
        </div>
      </div>

      {/* ── Negotiation Playbook (collapsible) ── */}
      <div className="card" style={{ marginBottom: 16, overflow: 'hidden' }}>
        <div
          style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '16px 20px',
            cursor: 'pointer',
            borderBottom: showNegotiation ? '1px solid var(--border)' : 'none',
            transition: 'border-color 0.2s',
          }}
          onClick={() => setShowNegotiation(v => !v)}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Icon name="tag" size={13} color="var(--accent)" strokeWidth={2.2} />
            <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.05em', color: 'var(--text-1)', textTransform: 'uppercase' }}>
              Negotiation Playbook
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 11, color: 'var(--text-3)' }}>{showNegotiation ? 'Hide' : 'Show'}</span>
            <svg
              width="14" height="14" viewBox="0 0 24 24" fill="none"
              stroke="var(--text-3)" strokeWidth="2.5" strokeLinecap="round"
              style={{ transform: showNegotiation ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.25s ease' }}
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        </div>
        {showNegotiation && (
          <div style={{ padding: '20px' }}>
            <NegotiationPlaybook negotiation={negotiation} confidenceScore={confidenceScore} />
          </div>
        )}
      </div>

      {/* ── Rule-Based Pricing Pipeline ── */}
      <RuleBasedPricingPipeline result={enhancedResult} inputs={inputs} />

      {/* ── Expandable breakdown tables ── */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="label-xs" style={{ marginBottom: 12 }}>Detailed Breakdowns</div>
        <ExpandableBreakdownTable title="View reconditioning breakdown" rows={reconRows} totalLabel="Total" totalValue={recon?.total} />
        <div style={{ height: 1, background: 'var(--border)', margin: '12px 0' }} />
        <ExpandableBreakdownTable title="View risk breakdown" rows={riskRows} totalLabel="Total" totalValue={wheelrRisk?.total} />
      </div>

      <button className="btn btn-secondary btn-full btn-lg" onClick={() => setActiveScreen('enhanced-input')}>
        <Icon name="arrowLeft" size={16} color="var(--text-2)" strokeWidth={2.2} />
        Back to Valuation Setup
      </button>
    </div>
  );
}
