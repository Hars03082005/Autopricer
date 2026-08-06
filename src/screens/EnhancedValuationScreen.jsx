import { useEffect, useMemo, useState } from 'react';
import { useApp } from '../context/AppContext.jsx';
import { formatINR, getSeasonalContext } from '../utils/format.js';
import {
  INDIAN_STATES,
  SELLER_REASONS,
  checkDisqualifier,
  getReconCost,
} from '../utils/wheelrCosts.js';
import { runEnhancedEvaluation, fetchCatalog } from '../utils/apiValuation.js';
import { ConditionGradesSection } from '../components/ConditionGradeField.jsx';
import Icon from '../components/Icon.jsx';
const FUELS         = ['Petrol', 'Diesel', 'Electric', 'CNG', 'Hybrid'];
const TRANSMISSIONS = ['Manual', 'Automatic', 'CVT', 'DCT'];
const YEARS         = Array.from({ length: 15 }, (_, i) => String(2025 - i));
const OWNER_COUNTS  = ['1', '2', '3', '4'];
const CONDITIONS    = ['Excellent', 'Good', 'Average', 'Poor'];
function titleCase(str) {
  return String(str || '')
    .split(' ')
    .map(w => w ? w[0].toUpperCase() + w.slice(1) : '')
    .join(' ');
}
function inspectionGrades(inspection) {
  return {
    engine:     inspection.engineGrade,
    tyre:       inspection.tyreGrade,
    body:       inspection.bodyGrade,
    interior:   inspection.interiorGrade,
    electrical: inspection.electricalGrade,
  };
}
export default function EnhancedValuationScreen() {
  const {
    inputs, updateInput,
    enhancedInspection, updateEnhancedInspection, updateVendorType,
    setEnhancedResult, setActiveScreen, setIsLoading, addEvaluation,
  } = useApp();
  const [error,   setError]   = useState('');
  // Dataset catalog state
  const [catalog,       setCatalog]       = useState({});
  const [catalogLoaded, setCatalogLoaded] = useState(false);
  const [catalogError,  setCatalogError]  = useState(false);
  // Load catalog on mount
  useEffect(() => {
    fetchCatalog()
      .then(data => {
        setCatalog(data);
        setCatalogLoaded(true);
      })
      .catch(() => {
        setCatalogLoaded(true);
        setCatalogError(true);
      });
  }, []);
  const brandOptions = useMemo(() =>
    Object.keys(catalog).sort(), [catalog]);
  const selectedBrandKey = (inputs.brand || '').toLowerCase();
  const modelOptions = useMemo(() => {
    const modelsMap = catalog[selectedBrandKey] || {};
    return Object.keys(modelsMap).sort();
  }, [catalog, selectedBrandKey]);
  const selectedModelKey = (inputs.model || '').toLowerCase();
  const variantOptions = useMemo(() => {
    const modelsMap = catalog[selectedBrandKey] || {};
    const variants  = modelsMap[selectedModelKey] || [];
    return variants.sort();
  }, [catalog, selectedBrandKey, selectedModelKey]);
  // Handlers for cascading resets
  function handleBrandChange(brandKey) {
    updateInput('brand',   brandKey);
    updateInput('model',   '');
    updateInput('variant', '');
  }
  function handleModelChange(modelKey) {
    updateInput('model',   modelKey);
    updateInput('variant', '');
  }
  const vehicleAge  = new Date().getFullYear() - Number(inputs.year || 2021);
  const odometer    = Number(inputs.mileage || 0);
  const ownerCount  = Number(inputs.ownerCount || 1);
  const grades      = inspectionGrades(enhancedInspection);
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
      engine:     'engineGrade',
      tyre:       'tyreGrade',
      body:       'bodyGrade',
      interior:   'interiorGrade',
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
      const inputsWithCity = { ...inputs, city: 'Bangalore' };
      const result = await runEnhancedEvaluation(inputsWithCity, enhancedInspection);
      setEnhancedResult({ ...result, inspection: { ...enhancedInspection } });
      addEvaluation({ ...inputsWithCity }, result, 'Enhanced Valuation');
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
          <div className="page-subtitle">Configure detailed multi-point inspection variables &amp; cost parameters</div>
        </div>
      </div>
      {error && (
        <div className="error-banner" style={{ marginBottom: 16 }}>
          <Icon name="warning" size={14} color="#dc2626" strokeWidth={2} />
          {error}
        </div>
      )}
      {!catalogLoaded && (
        <div className="error-banner" style={{ marginBottom: 12, background: 'var(--info-light)', borderColor: 'var(--info)', color: 'var(--info)' }}>
          <Icon name="spinner" size={14} color="var(--info)" strokeWidth={2} />
          Loading vehicle catalog from dataset…
        </div>
      )}
      {catalogError && (
        <div className="error-banner" style={{ marginBottom: 12 }}>
          <Icon name="warning" size={14} color="#dc2626" strokeWidth={2} />
          Could not load dataset catalog — start the FastAPI backend first.
        </div>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16, alignItems: 'start' }}>
        <div className="card">
          <div className="label-xs" style={{ marginBottom: 16 }}>Vehicle Details</div>
          <div className="field-group">
            <label className="field-label">
              Brand
              {catalogLoaded && !catalogError && (
                <span style={{ fontSize: 10, color: 'var(--text-3)', marginLeft: 6 }}>
                  ({brandOptions.length} from dataset)
                </span>
              )}
            </label>
            <select
              className="field-input field-select"
              value={inputs.brand}
              onChange={e => handleBrandChange(e.target.value)}
              disabled={!catalogLoaded || catalogError}
            >
              <option value="">— Select brand —</option>
              {brandOptions.map(b => (
                <option key={b} value={b}>{titleCase(b)}</option>
              ))}
            </select>
          </div>
          <div className="field-group">
            <label className="field-label">
              Model
              {inputs.brand && modelOptions.length > 0 && (
                <span style={{ fontSize: 10, color: 'var(--text-3)', marginLeft: 6 }}>
                  ({modelOptions.length} available)
                </span>
              )}
            </label>
            <select
              className="field-input field-select"
              value={inputs.model}
              onChange={e => handleModelChange(e.target.value)}
              disabled={!inputs.brand || modelOptions.length === 0}
            >
              <option value="">— Select model —</option>
              {modelOptions.map(m => (
                <option key={m} value={m}>{titleCase(m)}</option>
              ))}
            </select>
          </div>
          <div className="field-group">
            <label className="field-label">
              Variant
              {inputs.model && variantOptions.length > 0 && (
                <span style={{ fontSize: 10, color: 'var(--text-3)', marginLeft: 6 }}>
                  ({variantOptions.length} available)
                </span>
              )}
            </label>
            <select
              className="field-input field-select"
              value={inputs.variant || ''}
              onChange={e => updateInput('variant', e.target.value)}
              disabled={!inputs.model || variantOptions.length === 0}
            >
              <option value="">— Select variant (optional) —</option>
              {variantOptions.map(v => (
                <option key={v} value={v}>{titleCase(v)}</option>
              ))}
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
              <input
                type="text"
                className="field-input"
                value="Bangalore"
                disabled
                style={{ opacity: 0.6, cursor: 'not-allowed' }}
              />
              <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 4 }}>
                Dataset covers Bangalore listings only
              </div>
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
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
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
          <div className="card">
            <div className="label-xs" style={{ marginBottom: 12 }}>Live Recon Estimate</div>
            <div style={{ fontSize: 24, fontWeight: 800, color: 'var(--text-1)', marginBottom: 14 }}>
              {formatINR(recon.total)}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {[
                { label: 'Category Repairs',     val: recon.total - recon.fixed_cost },
                { label: 'RC Transfer Cost',      val: recon.rc_transfer_cost },
                { label: 'Det.+Ops Fixed Costs',  val: recon.fixed_cost - recon.rc_transfer_cost },
              ].map((r, idx) => (
                <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                  <span style={{ color: 'var(--text-3)' }}>{r.label}</span>
                  <strong style={{ color: 'var(--text-2)' }}>{formatINR(r.val)}</strong>
                </div>
              ))}
            </div>
          </div>
          <div className="card">
            <div className="label-xs" style={{ marginBottom: 12 }}>Overrides &amp; Insurance</div>
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
