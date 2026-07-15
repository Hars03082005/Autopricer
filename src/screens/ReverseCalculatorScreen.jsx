import { useMemo, useState } from 'react';
import { useApp } from '../context/AppContext.jsx';
import { formatINR, getSeasonalContext } from '../utils/format.js';
import {
  INDIAN_STATES,
  SELLER_REASONS,
  DEFAULT_VENDOR_TYPE,
  DEAL_HEALTH_META,
  getReconCost,
} from '../utils/wheelrCosts.js';
import { runReverseCalculate } from '../utils/apiValuation.js';
import { ConditionGradesSection } from '../components/ConditionGradeField.jsx';
import { DealHealthBanner, NegotiationPlaybook } from '../components/WheelrPanels.jsx';
import Icon from '../components/Icon.jsx';

const OWNER_COUNTS = ['1', '2', '3', '4'];

export default function ReverseCalculatorScreen() {
  const { setReverseResult, reverseResult, inputs } = useApp();
  const [expectedSellPrice, setExpectedSellPrice] = useState('950000');
  const [targetMarginPct, setTargetMarginPct] = useState(15);
  const [ownerCount, setOwnerCount] = useState('1');
  const [odometer, setOdometer] = useState('28000');
  const [year, setYear] = useState(inputs.year || '2021');
  const [accidentHistory, setAccidentHistory] = useState('none');
  const [registrationState, setRegistrationState] = useState('Maharashtra');
  const [sameState, setSameState] = useState(true);
  const [loanOutstanding, setLoanOutstanding] = useState(false);
  const [sellerReason, setSellerReason] = useState('upgrading');
  const [grades, setGrades] = useState({
    engine: 'average',
    tyre: 'good',
    body: 'minor',
    interior: 'clean',
    electrical: 'all_good',
  });
  const [vendorType, setVendorType] = useState({ ...DEFAULT_VENDOR_TYPE });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const seasonal = getSeasonalContext(new Date().getMonth() + 1);
  const liveRecon = useMemo(() => getReconCost(grades, vendorType), [grades, vendorType]);
  const profitPreview = Math.round(Number(expectedSellPrice || 0) * (targetMarginPct / 100));

  const handleGradeChange = (category, value) => {
    setGrades(prev => ({ ...prev, [category]: value }));
  };

  const handleVendorChange = (category, value) => {
    setVendorType(prev => ({ ...prev, [category]: value }));
  };

  const handleCalculate = async () => {
    setError('');
    setLoading(true);
    try {
      const saleState = sameState ? registrationState : 'Other State';
      const payload = {
        expected_sell_price: Math.trunc(Number(expectedSellPrice || 0)),
        year: Math.trunc(Number(year)),
        accident_history: accidentHistory,
        registration_state: registrationState,
        sale_state: saleState,
        loan_outstanding: loanOutstanding,
        seller_reason: sellerReason,
        engine_grade: grades.engine,
        tyre_grade: grades.tyre,
        body_grade: grades.body,
        interior_grade: grades.interior,
        electrical_grade: grades.electrical,
        vendor_type: vendorType,
        owner_count: Math.trunc(Number(ownerCount)),
        odometer: Math.trunc(Number(odometer)),
        target_margin_pct: targetMarginPct / 100,
      };
      const result = await runReverseCalculate(payload);
      setReverseResult(result);
    } catch (e) {
      console.error(e);
      setError('Reverse calculation failed. Ensure the FastAPI backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="screen screen-wide enhanced-screen">
      <div className="page-header">
        <div>
          <div className="page-title">Reverse Calculator</div>
          <div className="page-subtitle">Determine acquisition ceiling by working backwards from expected sale price</div>
        </div>
      </div>

      {error && (
        <div className="error-banner" style={{ marginBottom: 16 }}>
          <Icon name="warning" size={14} color="#dc2626" strokeWidth={2} />
          {error}
        </div>
      )}

      {/* Main 2-column workspace */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16, alignItems: 'start' }}>
        
        {/* Left Column: Input Form parameters */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          
          {/* Expected Sell Price Card */}
          <div className="card">
            <div className="label-xs" style={{ marginBottom: 12 }}>Expected Listing Price</div>
            <div style={{ position: 'relative' }}>
              <span style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', fontSize: 18, color: 'var(--text-3)', fontWeight: 700 }}>₹</span>
              <input
                type="number"
                className="field-input field-input-lg"
                value={expectedSellPrice}
                onChange={e => setExpectedSellPrice(e.target.value)}
                placeholder="950000"
                style={{ paddingLeft: 28 }}
              />
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 6 }}>
              Expected retail value of this vehicle post-recon
            </div>
          </div>

          {/* Target Margin Input */}
          <div className="card">
            <div className="label-xs" style={{ marginBottom: 12 }}>Target Margin %</div>
            <div className="vws-money-wrap" style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <input
                type="number"
                min="5"
                max="30"
                className="field-input"
                value={targetMarginPct}
                onChange={e => setTargetMarginPct(Number(e.target.value))}
                placeholder="15"
                style={{ width: '100%', paddingRight: '24px' }}
              />
              <span style={{ position: 'absolute', right: '12px', fontSize: '13px', fontWeight: 'bold', color: 'var(--text-3)' }}>%</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-3)', marginTop: 6 }}>
              <span>Conservative profit (10%)</span>
              <span>Target Profit: <strong>{formatINR(profitPreview)}</strong></span>
            </div>
          </div>

          {/* Condition Grades */}
          <div className="card">
            <div className="label-xs" style={{ marginBottom: 12 }}>Condition Grades</div>
            <ConditionGradesSection
              grades={grades}
              vendorType={vendorType}
              onGradeChange={handleGradeChange}
              onVendorChange={handleVendorChange}
            />
            <div style={{ height: 1, background: 'var(--border)', margin: '14px 0' }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
              <span style={{ color: 'var(--text-3)' }}>Estimated Recon Costs:</span>
              <strong style={{ color: 'var(--text-1)' }}>{formatINR(liveRecon.total)}</strong>
            </div>
          </div>

          {/* Risk Factors */}
          <div className="card">
            <div className="label-xs" style={{ marginBottom: 16 }}>Risk Adjustments</div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <div className="field-group">
                <label className="field-label">Year</label>
                <input type="number" className="field-input" value={year} onChange={e => setYear(e.target.value)} />
              </div>
              <div className="field-group">
                <label className="field-label">Owners</label>
                <select className="field-input field-select" value={ownerCount} onChange={e => setOwnerCount(e.target.value)}>
                  {OWNER_COUNTS.map(o => <option key={o} value={o}>{o}</option>)}
                </select>
              </div>
            </div>

            <div className="field-group">
              <label className="field-label">Odometer (km)</label>
              <input type="number" className="field-input" value={odometer} onChange={e => setOdometer(e.target.value)} />
            </div>

            <div className="field-group">
              <label className="field-label">Accident History</label>
              <div className="seg-control" style={{ display: 'flex', gap: 4 }}>
                {['none', 'minor', 'major'].map(v => (
                  <button
                    key={v}
                    type="button"
                    className={`seg-btn ${accidentHistory === v ? 'active' : ''}`}
                    onClick={() => setAccidentHistory(v)}
                  >
                    {v === 'none' ? 'None' : v === 'minor' ? 'Minor' : 'Major'}
                  </button>
                ))}
              </div>
            </div>

            <div className="field-group">
              <label className="field-label">Registration State</label>
              <select className="field-input field-select" value={registrationState} onChange={e => setRegistrationState(e.target.value)}>
                {INDIAN_STATES.map(s => <option key={s}>{s}</option>)}
              </select>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <div className="field-group">
                <label className="field-label">Sale Jurisdiction</label>
                <div className="seg-control">
                  <button type="button" className={`seg-btn ${sameState ? 'active' : ''}`} onClick={() => setSameState(true)}>In-State</button>
                  <button type="button" className={`seg-btn ${!sameState ? 'active' : ''}`} onClick={() => setSameState(false)}>Out-State</button>
                </div>
              </div>

              <div className="field-group">
                <label className="field-label">Loan Outstanding</label>
                <div className="seg-control">
                  <button type="button" className={`seg-btn ${!loanOutstanding ? 'active' : ''}`} onClick={() => setLoanOutstanding(false)}>No</button>
                  <button type="button" className={`seg-btn ${loanOutstanding ? 'active' : ''}`} onClick={() => setLoanOutstanding(true)}>Yes</button>
                </div>
              </div>
            </div>

            <div className="field-group" style={{ marginBottom: 0 }}>
              <label className="field-label">Seller Reason</label>
              <select className="field-input field-select" value={sellerReason} onChange={e => setSellerReason(e.target.value)}>
                {SELLER_REASONS.map(r => <option key={r.value} value={r.value}>{r.label}</option>)}
              </select>
            </div>

          </div>

          <button className="btn btn-primary btn-full btn-lg" onClick={handleCalculate} disabled={loading}>
            <Icon name="arrowLeftRight" size={16} color="white" strokeWidth={2.2} />
            {loading ? 'Performing calculations…' : 'Calculate Max Buy Price'}
          </button>
        </div>

        {/* Right Column: Results & Playbook */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          
          {/* Seasonal context banner */}
          <div className="card" style={{ background: 'var(--info-light)', borderLeft: '3px solid var(--info)' }}>
            <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
              <Icon name="calendar" size={16} color="var(--info)" strokeWidth={2.2} />
              <div>
                <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--info)' }}>
                  {seasonal.label} · {seasonal.mult}x Demand
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-2)', marginTop: 4, lineHeight: 1.4 }}>
                  {seasonal.note}
                </div>
              </div>
            </div>
          </div>

          {/* Pre-screening failure */}
          {reverseResult?.disqualifier?.disqualified && (
            <div className="card card-danger" style={{ borderLeft: '3px solid' }}>
              <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                <Icon name="warning" size={18} color="var(--danger)" strokeWidth={2} />
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--danger)' }}>
                    Disqualified Vehicle
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-2)', marginTop: 4 }}>
                    {reverseResult.disqualifier.reason}
                  </div>
                </div>
              </div>
            </div>
          )}

          {reverseResult ? (
            <>
              {reverseResult.dealHealth && (
                <DealHealthBanner dealHealth={reverseResult.dealHealth} meta={DEAL_HEALTH_META} />
              )}

              {/* Price Waterfall */}
              <div className="card">
                <div className="label-xs" style={{ marginBottom: 12 }}>Cost Deductions Waterfall</div>
                
                <div className="waterfall">
                  {(reverseResult.priceBreakdown || []).map((row, i) => {
                    const isTotal = row.sign === '=';
                    const isDeduct = row.sign === '-';
                    const valPct = (Math.abs(row.value) / Number(expectedSellPrice || 1)) * 100;
                    return (
                      <div key={i} className="waterfall-row" style={isTotal ? { borderTop: '2px solid var(--border)', paddingTop: 10, marginTop: 6 } : {}}>
                        <div className="waterfall-label" style={isTotal ? { fontWeight: 700 } : {}}>{row.label}</div>
                        {!isTotal && (
                          <div className="waterfall-bar-track">
                            <div
                              className={`waterfall-bar-fill ${isDeduct ? 'red' : 'green'}`}
                              style={{ width: `${Math.min(100, Math.max(2, valPct))}%` }}
                            />
                          </div>
                        )}
                        {isTotal && <div style={{ flex: 1 }} />}
                        <div className="waterfall-val" style={isTotal ? { fontSize: 16, fontWeight: 800 } : isDeduct ? { color: 'var(--danger)' } : { color: 'var(--success)' }}>
                          {isDeduct ? '−' : ''}{formatINR(row.value)}
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div
                  style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    borderTop: '2px solid var(--border)', paddingTop: 14, marginTop: 12
                  }}
                >
                  <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-1)' }}>Acquisition Ceiling (Max Offer)</div>
                  <div style={{ fontSize: 20, fontWeight: 800, color: 'var(--accent)', letterSpacing: '-0.5px' }}>
                    {formatINR(reverseResult.maxBuyPrice)}
                  </div>
                </div>
              </div>

              {/* Negotiation Playbook */}
              <div className="card" style={{ padding: '20px' }}>
                <NegotiationPlaybook negotiation={reverseResult.negotiation} variant="reverse" />
              </div>
            </>
          ) : (
            <div className="card" style={{ textAlign: 'center', padding: '40px 20px' }}>
              <div className="home-empty-icon" style={{ marginBottom: 12 }}>
                <Icon name="arrowLeftRight" size={24} color="var(--text-3)" strokeWidth={1.8} />
              </div>
              <p style={{ fontSize: 13, color: 'var(--text-3)', lineHeight: 1.5 }}>
                Fill in the details on the left and click <strong>Calculate Max Buy Price</strong> to visualize the pricing waterfall and generate a negotiation playbook.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
