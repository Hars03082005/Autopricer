import { useMemo, useState } from 'react';
import { useApp } from '../context/AppContext.jsx';
import { BRANDS, CITY_DEMAND } from '../utils/mockData.js';
import { formatINR, getSeasonalContext } from '../utils/format.js';
import {
  INDIAN_STATES,
  SELLER_REASONS,
  checkDisqualifier,
  getReconCost,
} from '../utils/wheelrCosts.js';
import { runEnhancedEvaluation } from '../utils/apiValuation.js';
import { ConditionGradesSection } from '../components/ConditionGradeField.jsx';
import Icon from '../components/Icon.jsx';

const FUELS = ['Petrol', 'Diesel', 'Electric', 'CNG', 'Hybrid'];
const TRANSMISSIONS = ['Manual', 'Automatic', 'CVT', 'DCT'];
const YEARS = Array.from({ length: 15 }, (_, i) => String(2025 - i));
const CITIES = Object.keys(CITY_DEMAND).sort();
const OWNER_COUNTS = ['1', '2', '3', '4'];
const CONDITIONS = ['Excellent', 'Good', 'Average', 'Poor'];

function inspectionGrades(inspection) {
  return {
    engine: inspection.engineGrade,
    tyre: inspection.tyreGrade,
    body: inspection.bodyGrade,
    interior: inspection.interiorGrade,
    electrical: inspection.electricalGrade,
  };
}

export default function EnhancedValuationScreen() {
  const {
    inputs, updateInput,
    enhancedInspection, updateEnhancedInspection, updateVendorType,
    setEnhancedResult, setActiveScreen, setIsLoading, addEvaluation,
  } = useApp();

  const [error, setError] = useState('');

  const brands = Object.keys(BRANDS).sort();
  const models = BRANDS[inputs.brand] || [];
  const vehicleAge = new Date().getFullYear() - Number(inputs.year || 2021);
  const odometer = Number(inputs.mileage || 0);
  const ownerCount = Number(inputs.ownerCount || 1);
  const grades = inspectionGrades(enhancedInspection);
  const recon = useMemo(
    () => getReconCost(grades, enhancedInspection.vendorType, enhancedInspection.rcTransferCost),
    [grades, enhancedInspection.vendorType, enhancedInspection.rcTransferCost],
  );
  const disqualifier = useMemo(
    () => checkDisqualifier(vehicleAge, odometer, ownerCount, enhancedInspection.accidentHistory),
    [vehicleAge, odometer, ownerCount, enhancedInspection.accidentHistory],
  );
  const seasonal = getSeasonalContext(new Date().getMonth() + 1);

  const handleGradeChange = (category, value) => {
    const map = {
      engine: 'engineGrade',
      tyre: 'tyreGrade',
      body: 'bodyGrade',
      interior: 'interiorGrade',
      electrical: 'electricalGrade',
    };
    updateEnhancedInspection(map[category], value);
  };

  const handleSubmit = async () => {
    setError('');
    setIsLoading(true);
    setEnhancedResult(null);
    setActiveScreen('enhanced-result');
    try {
      const result = await runEnhancedEvaluation(inputs, enhancedInspection);
      setEnhancedResult({ ...result, inspection: { ...enhancedInspection } });
      addEvaluation({ ...inputs }, result, 'Enhanced Valuation');
    } catch (e) {
      console.error(e);
      setActiveScreen('enhanced-input');
      setError('Enhanced evaluation failed. Start FastAPI backend with: uvicorn backend.main:app --reload');
    } finally {
      setIsLoading(false);
    }
  };

  const isFormValid = inputs.brand && inputs.model && inputs.year && inputs.mileage;

  return (
    <div className="screen screen-wide enhanced-screen">
      <div className="page-header">
        <div>
          <div className="page-title">Enhanced Valuation Setup</div>
          <div className="page-subtitle">Configure detailed multi-point inspection variables & cost parameters</div>
        </div>
      </div>

      {error && (
        <div className="error-banner" style={{ marginBottom: 16 }}>
          <Icon name="warning" size={14} color="#dc2626" strokeWidth={2} />
          {error}
        </div>
      )}

      {/* 3-Column layout */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16, alignItems: 'start' }}>
        
        {/* Column 1: Vehicle Details */}
        <div className="card">
          <div className="label-xs" style={{ marginBottom: 16 }}>Vehicle Details</div>
          
          <div className="field-group">
            <label className="field-label">Brand</label>
            <select className="field-input field-select" value={inputs.brand} onChange={e => updateInput('brand', e.target.value)}>
              {brands.map(b => <option key={b}>{b}</option>)}
            </select>
          </div>

          <div className="field-group">
            <label className="field-label">Model</label>
            <select className="field-input field-select" value={inputs.model} onChange={e => updateInput('model', e.target.value)}>
              {models.map(m => <option key={m}>{m}</option>)}
            </select>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div className="field-group">
              <label className="field-label">Year</label>
              <select className="field-input field-select" value={inputs.year} onChange={e => updateInput('year', e.target.value)}>
                {YEARS.map(y => <option key={y}>{y}</option>)}
              </select>
            </div>
            <div className="field-group">
              <label className="field-label">Fuel type</label>
              <select className="field-input field-select" value={inputs.fuel} onChange={e => updateInput('fuel', e.target.value)}>
                {FUELS.map(f => <option key={f}>{f}</option>)}
              </select>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div className="field-group">
              <label className="field-label">Transmission</label>
              <select className="field-input field-select" value={inputs.transmission} onChange={e => updateInput('transmission', e.target.value)}>
                {TRANSMISSIONS.map(t => <option key={t}>{t}</option>)}
              </select>
            </div>
            <div className="field-group">
              <label className="field-label">Odometer (km)</label>
              <input type="number" className="field-input" value={inputs.mileage} onChange={e => updateInput('mileage', e.target.value)} />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div className="field-group">
              <label className="field-label">Fuel eff. (km/l)</label>
              <input type="number" step="0.1" className="field-input" value={inputs.fuelEfficiency} onChange={e => updateInput('fuelEfficiency', e.target.value)} />
            </div>
            <div className="field-group">
              <label className="field-label">Owners</label>
              <select className="field-input field-select" value={inputs.ownerCount} onChange={e => updateInput('ownerCount', e.target.value)}>
                {OWNER_COUNTS.map(o => <option key={o} value={o}>{o}</option>)}
              </select>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div className="field-group">
              <label className="field-label">Engine CC</label>
              <input type="number" className="field-input" value={inputs.engineCc} onChange={e => updateInput('engineCc', e.target.value)} />
            </div>
            <div className="field-group">
              <label className="field-label">City</label>
              <select className="field-input field-select" value={inputs.city} onChange={e => updateInput('city', e.target.value)}>
                {CITIES.map(c => <option key={c}>{c}</option>)}
              </select>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div className="field-group" style={{ marginBottom: 0 }}>
              <label className="field-label">Condition</label>
              <select className="field-input field-select" value={inputs.condition} onChange={e => updateInput('condition', e.target.value)}>
                {CONDITIONS.map(c => <option key={c}>{c}</option>)}
              </select>
            </div>
            <div className="field-group" style={{ marginBottom: 0 }}>
              <label className="field-label">Target margin %</label>
              <input type="number" className="field-input" min="5" max="30" value={inputs.targetMarginPct} onChange={e => updateInput('targetMarginPct', e.target.value)} />
            </div>
          </div>
        </div>

        {/* Column 2: Inspection & Grading Details */}
        <div className="card">
          <div className="label-xs" style={{ marginBottom: 16 }}>Inspection Details</div>
          
          <div className="field-group">
            <label className="field-label">Accident History</label>
            <div className="seg-control" style={{ display: 'flex', gap: 4 }}>
              {['none', 'minor', 'major'].map(v => (
                <button
                  key={v}
                  type="button"
                  className={`seg-btn ${enhancedInspection.accidentHistory === v ? 'active' : ''}`}
                  onClick={() => updateEnhancedInspection('accidentHistory', v)}
                >
                  {v === 'none' ? 'None' : v === 'minor' ? 'Minor' : 'Major'}
                </button>
              ))}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div className="field-group">
              <label className="field-label">Loan Outstanding</label>
              <div className="seg-control">
                <button
                  type="button"
                  className={`seg-btn ${!enhancedInspection.loanOutstanding ? 'active' : ''}`}
                  onClick={() => updateEnhancedInspection('loanOutstanding', false)}
                >
                  No
                </button>
                <button
                  type="button"
                  className={`seg-btn ${enhancedInspection.loanOutstanding ? 'active' : ''}`}
                  onClick={() => updateEnhancedInspection('loanOutstanding', true)}
                >
                  Yes
                </button>
              </div>
            </div>

            <div className="field-group">
              <label className="field-label">Registration State</label>
              <select className="field-input field-select" value={enhancedInspection.registrationState} onChange={e => updateEnhancedInspection('registrationState', e.target.value)}>
                {INDIAN_STATES.map(s => <option key={s}>{s}</option>)}
              </select>
            </div>
          </div>

          <div className="field-group">
            <label className="field-label">Seller Reason</label>
            <select className="field-input field-select" value={enhancedInspection.sellerReason} onChange={e => updateEnhancedInspection('sellerReason', e.target.value)}>
              {SELLER_REASONS.map(r => <option key={r.value} value={r.value}>{r.label}</option>)}
            </select>
          </div>

          <div style={{ height: 1, background: 'var(--border)', margin: '14px 0' }} />
          <div className="label-xs" style={{ marginBottom: 12 }}>Condition Grades</div>
          
          <ConditionGradesSection
            grades={grades}
            vendorType={enhancedInspection.vendorType}
            onGradeChange={handleGradeChange}
            onVendorChange={updateVendorType}
          />

          <button
            className="btn btn-primary btn-full btn-lg"
            style={{ marginTop: 20 }}
            onClick={handleSubmit}
            disabled={!isFormValid}
          >
            <Icon name="zap" size={16} color="white" strokeWidth={2} />
            Enhanced Evaluate
          </button>
        </div>

        {/* Column 3: Live Cost Prescreening & Diagnostics */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          
          {/* Pre-screening status */}
          <div className={`card ${disqualifier.disqualified ? 'card-danger' : 'card-success'}`} style={{ borderLeft: '3px solid' }}>
            <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
              <Icon
                name={disqualifier.disqualified ? 'warning' : 'check'}
                size={18}
                color={disqualifier.disqualified ? 'var(--danger)' : 'var(--success)'}
                strokeWidth={disqualifier.disqualified ? 2 : 2.5}
              />
              <div>
                <div style={{ fontSize: 13, fontWeight: 700, color: disqualifier.disqualified ? 'var(--danger)' : 'var(--success)' }}>
                  {disqualifier.disqualified ? 'Pre-screening Rejected' : 'Pre-screening Passed'}
                </div>
                {disqualifier.disqualified && (
                  <div style={{ fontSize: 12, color: 'var(--text-2)', marginTop: 4 }}>
                    {disqualifier.reason}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Seasonal marker */}
          <div className="card" style={{ background: 'var(--info-light)', borderLeft: '3px solid var(--info)' }}>
            <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
              <Icon name="calendar" size={16} color="var(--info)" strokeWidth={2} />
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

          {/* Live recon total card */}
          <div className="card">
            <div className="label-xs" style={{ marginBottom: 12 }}>Live Recon Estimate</div>
            <div style={{ fontSize: 24, fontWeight: 800, color: 'var(--text-1)', marginBottom: 14 }}>
              {formatINR(recon.total)}
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {[
                { label: 'Category Repairs', val: recon.total - recon.fixed_cost },
                { label: 'RC Transfer Cost', val: recon.rc_transfer_cost },
                { label: 'Det.+Ops Fixed Costs', val: recon.fixed_cost - recon.rc_transfer_cost },
              ].map((r, idx) => (
                <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                  <span style={{ color: 'var(--text-3)' }}>{r.label}</span>
                  <strong style={{ color: 'var(--text-2)' }}>{formatINR(r.val)}</strong>
                </div>
              ))}
            </div>
          </div>

          {/* RC & IDV overrides */}
          <div className="card">
            <div className="label-xs" style={{ marginBottom: 12 }}>Overrides & Insurance</div>
            
            <div className="field-group">
              <label className="field-label">RC Transfer Cost (₹)</label>
              <input
                type="number"
                className="field-input"
                min="0"
                step="500"
                placeholder="e.g. 3500"
                value={enhancedInspection.rcTransferCost}
                onChange={e => updateEnhancedInspection('rcTransferCost', e.target.value)}
              />
              <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 4 }}>
                Detailing (₹2,500) + Ops (₹2,000) added automatically
              </div>
            </div>

            <div className="field-group" style={{ marginBottom: 0 }}>
              <label className="field-label">IDV from Policy (₹, optional)</label>
              <input
                type="number"
                className="field-input"
                min="0"
                step="5000"
                placeholder="e.g. 380000"
                value={enhancedInspection.idvValue === '0' ? '' : enhancedInspection.idvValue}
                onChange={e => updateEnhancedInspection('idvValue', e.target.value || '0')}
              />
              <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 4 }}>
                Used to flag excessive variance against predicted ML price
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
