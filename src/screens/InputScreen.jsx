import { useEffect, useMemo, useState } from 'react';
import { useApp } from '../context/AppContext.jsx';
import { LOCALITIES } from '../utils/mockData.js';
import { fetchBrands, fetchOptions, runMLValuation } from '../utils/apiValuation.js';
import { DATASET_CATALOG } from '../utils/variantCatalog.js';
import SearchableDropdown from '../components/SearchableDropdown.jsx';
import Icon from '../components/Icon.jsx';

const CURRENT_YEAR = new Date().getFullYear();
const YEARS        = Array.from({ length: 25 }, (_, i) => String(CURRENT_YEAR - i));
const FUELS        = ['Petrol', 'Diesel', 'Electric', 'CNG', 'Hybrid'];
const TRANSMISSIONS = ['Manual', 'Automatic', 'CVT', 'DCT', 'AMT', 'IMT'];
const OWNERS       = ['1', '2', '3', '4+'];

const COLORS = [
  { name: 'White',  hex: '#f5f5f5', border: '#d4d4d4' },
  { name: 'Silver', hex: '#d1d5db', border: '#9ca3af' },
  { name: 'Grey',   hex: '#6b7280', border: '#4b5563' },
  { name: 'Black',  hex: '#1f2937', border: '#111827' },
  { name: 'Blue',   hex: '#2563eb', border: '#1d4ed8' },
  { name: 'Red',    hex: '#dc2626', border: '#b91c1c' },
  { name: 'Brown',  hex: '#78350f', border: '#451a03' },
  { name: 'Beige',  hex: '#e5e7eb', border: '#d1d5db' },
  { name: 'Orange', hex: '#ea580c', border: '#c2410c' },
  { name: 'Green',  hex: '#16a34a', border: '#15803d' },
];

function healthScore(inputs) {
  if (!inputs.brand) return 0;
  const age  = new Date().getFullYear() - Number(inputs.year || 2020);
  const km   = Number(inputs.mileage || 0);
  const own  = Number(inputs.ownerCount || 1);
  const cond = inputs.condition || 'Good';

  const ageS  = age <= 2 ? 100 : age <= 4 ? 85 : age <= 6 ? 70 : age <= 8 ? 55 : age <= 10 ? 40 : 25;
  const kmS   = km < 20000 ? 100 : km < 40000 ? 85 : km < 60000 ? 70 : km < 90000 ? 55 : km < 120000 ? 40 : 20;
  const ownS  = own === 1 ? 100 : own === 2 ? 70 : own === 3 ? 45 : 20;
  const condS = { Excellent: 100, Good: 75, Average: 45, Poor: 20 }[cond] ?? 60;

  return Math.round(ageS * 0.25 + kmS * 0.30 + ownS * 0.20 + condS * 0.25);
}

function healthMeta(score) {
  if (score >= 75) return { label: 'High Confidence Asset', color: '#16a34a', bg: '#f0fdf4' };
  if (score >= 55) return { label: 'Viable Opportunity',   color: '#d97706', bg: '#fffbeb' };
  if (score >= 35) return { label: 'Requires Inspection',  color: '#ea580c', bg: '#fff7ed' };
  return              { label: 'High Holding Risk',       color: '#dc2626', bg: '#fef2f2' };
}

function titleCase(str) {
  if (!str) return '';
  return str
    .toLowerCase()
    .split(' ')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

export default function InputScreen() {
  const {
    inputs = {},
    setInputs,
    updateInput,
    setValuationResult,
    setActiveScreen,
    addEvaluation,
    appendEvaluation,
  } = useApp();

  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState('');
  const [brandsMap, setBrandsMap]   = useState({});
  const [optYears, setOptYears]     = useState(YEARS);
  const [optFuels, setOptFuels]     = useState(FUELS);
  const [optTrans, setOptTrans]     = useState(TRANSMISSIONS);

  const upd = (k, v) => {
    if (typeof setInputs === 'function') {
      setInputs(prev => ({ ...prev, [k]: v }));
    } else if (typeof updateInput === 'function') {
      updateInput(k, v);
    }
  };

  // Load brands and options on mount
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const [brandsData, optList] = await Promise.all([fetchBrands(), fetchOptions()]);
        if (!active) return;
        if (brandsData && typeof brandsData === 'object') {
          setBrandsMap(brandsData);
        }
        if (optList?.years?.length) setOptYears(optList.years);
        if (optList?.fuel_types?.length) setOptFuels(optList.fuel_types);
        if (optList?.transmissions?.length) setOptTrans(optList.transmissions);
      } catch (err) {
        console.warn('[InputScreen] options fetch notice:', err.message);
      }
    })();
    return () => { active = false; };
  }, []);

  // Compute Brand list
  const brandList = useMemo(() => {
    const fromApi = Object.keys(brandsMap);
    if (fromApi.length > 0) return fromApi.sort();
    const fromCatalog = Object.keys(DATASET_CATALOG).map(titleCase);
    return fromCatalog.length ? fromCatalog.sort() : ['Honda', 'Hyundai', 'Maruti', 'Tata', 'Toyota', 'Mahindra', 'KIA', 'BMW', 'Mercedes-Benz', 'Audi'];
  }, [brandsMap]);

  // Compute Model list based on selected brand
  const modelList = useMemo(() => {
    if (!inputs.brand) return [];
    
    // Check API brandsMap first
    const brandKey = Object.keys(brandsMap).find(b => b.toLowerCase() === inputs.brand.toLowerCase());
    if (brandKey && Array.isArray(brandsMap[brandKey]) && brandsMap[brandKey].length > 0) {
      return brandsMap[brandKey].sort();
    }

    // Check dataset catalog
    const catalogBrandKey = Object.keys(DATASET_CATALOG).find(b => b.toLowerCase() === inputs.brand.toLowerCase());
    if (catalogBrandKey && DATASET_CATALOG[catalogBrandKey]) {
      const models = Object.keys(DATASET_CATALOG[catalogBrandKey]).map(titleCase);
      if (models.length > 0) return models.sort();
    }

    return ['City', 'Creta', 'Swift', 'Nexon', 'Innova Crysta', 'Thar', 'Seltos', 'Fortuner', '3 Series', 'C-Class'];
  }, [inputs.brand, brandsMap]);

  // Compute Variant list based on selected brand & model
  const variantList = useMemo(() => {
    if (!inputs.brand || !inputs.model) return [];
    
    const catalogBrandKey = Object.keys(DATASET_CATALOG).find(b => b.toLowerCase() === inputs.brand.toLowerCase());
    if (catalogBrandKey && DATASET_CATALOG[catalogBrandKey]) {
      const modelKey = Object.keys(DATASET_CATALOG[catalogBrandKey]).find(m => m.toLowerCase() === inputs.model.toLowerCase());
      if (modelKey && Array.isArray(DATASET_CATALOG[catalogBrandKey][modelKey])) {
        const rawVariants = DATASET_CATALOG[catalogBrandKey][modelKey];
        if (rawVariants.length > 0) {
          return rawVariants.map(v => v.toUpperCase()).sort();
        }
      }
    }

    return ['Standard', 'Base Trim', 'V MT', 'VX MT', 'ZX', 'ZX CVT', 'SX', 'SX(O)'];
  }, [inputs.brand, inputs.model]);

  const score = useMemo(() => healthScore(inputs), [inputs]);
  const meta  = useMemo(() => healthMeta(score), [score]);

  const handleSubmit = async (e) => {
    e?.preventDefault();
    if (!inputs.brand) {
      setError('Please select a vehicle brand');
      return;
    }
    if (!inputs.model) {
      setError('Please select a vehicle model');
      return;
    }

    setError('');
    setLoading(true);
    try {
      const payload = {
        ...inputs,
        city: 'Bangalore',
        sellerType: inputs.sellerType || 'Individual',
      };
      const res = await runMLValuation(payload);
      if (typeof setValuationResult === 'function') {
        setValuationResult(res);
      }
      const saveFn = addEvaluation || appendEvaluation;
      if (typeof saveFn === 'function') {
        await saveFn(payload, res, 'Single Vehicle');
      }
      if (typeof setActiveScreen === 'function') {
        setActiveScreen('result');
      }
    } catch (err) {
      setError(err?.message || 'Valuation service encountered an issue. Please retry.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="screen">
      <div className="page-header" style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <div className="page-title">New Vehicle Valuation</div>
          <div className="page-subtitle">Input vehicle specs, physical condition, and target dealer parameters for instant ML valuation.</div>
        </div>
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => setActiveScreen('enhanced-input')}
        >
          <Icon name="zap" size={13} strokeWidth={2} />
          <span>Full Inspection Form (Enhanced)</span>
        </button>
      </div>

      {error && (
        <div className="toast toast-error" style={{ marginBottom: 18 }}>
          <Icon name="warning" size={16} color="#dc2626" strokeWidth={2} />
          <div>{error}</div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="nv-root">
        {/* Left Column: Form Sections */}
        <div className="nv-form-col">
          {/* SECTION 1: Vehicle Identity */}
          <div className="nv-section">
            <div className="nv-section-header">
              <div className="nv-section-num">1</div>
              <div className="nv-section-title">Vehicle Identity</div>
            </div>
            <div className="nv-section-body">
              <div className="nv-grid">
                <div className="form-group">
                  <label className="form-label form-label-req">Brand</label>
                  <SearchableDropdown
                    options={brandList}
                    value={inputs.brand}
                    onChange={(val) => {
                      upd('brand', val);
                      upd('model', '');
                      upd('variant', '');
                    }}
                    placeholder="Select brand"
                  />
                </div>

                <div className="form-group">
                  <label className="form-label form-label-req">Model</label>
                  <SearchableDropdown
                    options={modelList}
                    value={inputs.model}
                    onChange={(val) => {
                      upd('model', val);
                      upd('variant', '');
                    }}
                    placeholder={inputs.brand ? "Select model" : "Select brand first"}
                    disabled={!inputs.brand}
                  />
                </div>
              </div>

              <div className="nv-grid" style={{ marginTop: 12 }}>
                <div className="form-group">
                  <label className="form-label form-label-req">Variant / Trim</label>
                  <SearchableDropdown
                    options={variantList}
                    value={inputs.variant}
                    onChange={(val) => upd('variant', val)}
                    placeholder={inputs.model ? "Select variant" : "Select model first"}
                    disabled={!inputs.model}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label form-label-req">Manufacturing Year</label>
                  <select
                    className="form-select"
                    value={inputs.year}
                    onChange={(e) => upd('year', e.target.value)}
                  >
                    {optYears.map((yr) => (
                      <option key={yr} value={yr}>{yr}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* SECTION 2: Physical & Technical Condition */}
          <div className="nv-section">
            <div className="nv-section-header">
              <div className="nv-section-num">2</div>
              <div className="nv-section-title">Physical & Technical Condition</div>
            </div>
            <div className="nv-section-body">
              <div className="nv-grid-4">
                <div className="form-group">
                  <label className="form-label form-label-req">Odometer (km)</label>
                  <input
                    type="number"
                    className="form-input"
                    placeholder="e.g. 35000"
                    value={inputs.mileage}
                    onChange={(e) => upd('mileage', e.target.value)}
                    min="0"
                    max="2000000"
                  />
                </div>

                <div className="form-group">
                  <label className="form-label form-label-req">Previous Owners</label>
                  <select
                    className="form-select"
                    value={inputs.ownerCount}
                    onChange={(e) => upd('ownerCount', e.target.value)}
                  >
                    {OWNERS.map((o) => (
                      <option key={o} value={o}>{o} {o === '1' ? 'Owner' : 'Owners'}</option>
                    ))}
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label form-label-req">Fuel Type</label>
                  <select
                    className="form-select"
                    value={inputs.fuel}
                    onChange={(e) => upd('fuel', e.target.value)}
                  >
                    {optFuels.map((f) => (
                      <option key={f} value={f}>{f}</option>
                    ))}
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label form-label-req">Transmission</label>
                  <select
                    className="form-select"
                    value={inputs.transmission}
                    onChange={(e) => upd('transmission', e.target.value)}
                  >
                    {optTrans.map((t) => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="nv-grid" style={{ marginTop: 14 }}>
                <div className="form-group">
                  <label className="form-label">Overall Condition</label>
                  <select
                    className="form-select"
                    value={inputs.condition || 'Good'}
                    onChange={(e) => upd('condition', e.target.value)}
                  >
                    <option value="Excellent">Excellent (Like new, zero blemishes)</option>
                    <option value="Good">Good (Minor wear, well maintained)</option>
                    <option value="Average">Average (Wear visible, needs recon)</option>
                    <option value="Poor">Poor (Major cosmetic/mechanical work)</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">Vehicle Color</label>
                  <div className="color-swatch-grid">
                    {COLORS.map((c) => (
                      <div
                        key={c.name}
                        className={`color-swatch ${(inputs.color || 'White').toLowerCase() === c.name.toLowerCase() ? 'selected' : ''}`}
                        style={{ background: c.hex, borderColor: c.border }}
                        title={c.name}
                        onClick={() => upd('color', c.name)}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* SECTION 3: Locality & Deal Parameters */}
          <div className="nv-section">
            <div className="nv-section-header" style={{ background: '#fef3ec' }}>
              <div className="nv-section-num" style={{ background: '#e85d26' }}>3</div>
              <div className="nv-section-title" style={{ color: '#cf4d1a' }}>Locality & Acquisition Parameters</div>
            </div>
            <div className="nv-section-body">
              <div className="nv-grid-3">
                <div className="form-group">
                  <label className="form-label">Locality / RTO Zone</label>
                  <SearchableDropdown
                    options={LOCALITIES}
                    value={inputs.locality || 'Indiranagar'}
                    onChange={(v) => upd('locality', v)}
                    placeholder="Select Locality"
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Target Margin (%)</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                    <input
                      type="range"
                      min="5"
                      max="25"
                      step="0.5"
                      value={inputs.targetMarginPct || 10}
                      onChange={(e) => upd('targetMarginPct', e.target.value)}
                      style={{ flex: 1, accentColor: '#e85d26' }}
                    />
                    <span style={{ fontWeight: 800, fontSize: 13.5, color: '#e85d26', width: 42, textAlign: 'right' }}>
                      {inputs.targetMarginPct || 10}%
                    </span>
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">Recon Buffer (₹)</label>
                  <input
                    type="number"
                    className="form-input"
                    value={inputs.repairBuffer || 25000}
                    onChange={(e) => upd('repairBuffer', e.target.value)}
                    placeholder="25000"
                    step="5000"
                  />
                </div>
              </div>
            </div>

            {/* Primary Action Button */}
            <div className="nv-cta">
              <button
                type="submit"
                className="btn btn-primary btn-xl"
                disabled={loading}
                style={{ flex: 1 }}
              >
                {loading ? (
                  <>
                    <div className="loading-spinner" style={{ width: 16, height: 16, borderWidth: 2, borderColor: '#fff', borderTopColor: 'transparent' }} />
                    <span>Analyzing Vehicle & Computing Margins...</span>
                  </>
                ) : (
                  <>
                    <Icon name="car" size={17} color="white" strokeWidth={2} />
                    <span>ANALYZE VEHICLE</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Asset Readiness Preview */}
        <div className="nv-preview-col">
          <div className="card" style={{ position: 'sticky', top: 72 }}>
            <div className="card-header">
              <div className="card-title">Asset Readiness</div>
              <span className="badge" style={{ background: meta.bg, color: meta.color, border: `1px solid ${meta.color}40` }}>
                {meta.label}
              </span>
            </div>

            <div className="card-body">
              {/* Score ring */}
              <div className="health-gauge-wrap">
                <div className="health-score-ring">
                  <svg width="80" height="80" viewBox="0 0 80 80">
                    <circle cx="40" cy="40" r="34" fill="none" stroke="var(--border)" strokeWidth="6" />
                    <circle
                      cx="40"
                      cy="40"
                      r="34"
                      fill="none"
                      stroke={meta.color}
                      strokeWidth="6"
                      strokeDasharray={2 * Math.PI * 34}
                      strokeDashoffset={2 * Math.PI * 34 * (1 - score / 100)}
                      strokeLinecap="round"
                    />
                  </svg>
                  <div className="health-score-num">{score}</div>
                </div>
                <div className="health-label">Asset Health Score / 100</div>
              </div>

              {/* Summary details */}
              <div style={{ marginTop: 16, borderTop: '1px solid var(--border-2)', paddingTop: 14 }}>
                <div style={{ fontSize: 10.5, fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-4)', letterSpacing: 0.6, marginBottom: 8 }}>
                  Evaluation Target
                </div>
                <div style={{ fontSize: 16, fontWeight: 800, color: 'var(--text-1)', marginBottom: 4 }}>
                  {inputs.year} {inputs.brand || '—'} {inputs.model || ''}
                </div>
                <div style={{ fontSize: 12.5, color: 'var(--text-3)' }}>
                  {inputs.variant || 'Select Variant'} · {inputs.fuel} · {inputs.transmission}
                </div>
              </div>

              <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 6, fontSize: 12, color: 'var(--text-3)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Usage:</span>
                  <strong style={{ color: 'var(--text-1)' }}>{Number(inputs.mileage || 0).toLocaleString('en-IN')} km</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Ownership:</span>
                  <strong style={{ color: 'var(--text-1)' }}>{inputs.ownerCount} {inputs.ownerCount === '1' ? 'Owner' : 'Owners'}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Locality:</span>
                  <strong style={{ color: 'var(--text-1)' }}>{inputs.locality || 'Indiranagar'}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Target Margin:</span>
                  <strong style={{ color: '#e85d26' }}>{inputs.targetMarginPct || 10}%</strong>
                </div>
              </div>

              {/* Clean Business-Only Notice */}
              <div style={{ marginTop: 18, padding: 12, background: 'var(--surface-2)', borderRadius: 'var(--r-md)', border: '1px solid var(--border-2)' }}>
                <div style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text-2)', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Icon name="check" size={13} color="#16a34a" strokeWidth={2.5} />
                  <span>Market Valuation Ready</span>
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-4)', marginTop: 4 }}>
                  Real-time acquisition pricing based on verified transaction data and condition adjustments.
                </div>
              </div>
            </div>
          </div>
        </div>
      </form>
    </div>
  );
}
