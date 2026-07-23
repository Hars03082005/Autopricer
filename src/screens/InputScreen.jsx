import { useEffect, useMemo, useState } from 'react';
import { useApp } from '../context/AppContext.jsx';
import { CITY_DEMAND } from '../utils/mockData.js';
import { fetchBrands, fetchCatalog, runMLValuation, fetchRegistry } from '../utils/apiValuation.js';
import SearchableDropdown from '../components/SearchableDropdown.jsx';


/* ─── Static Constants ──────────────────────────────────────────── */
const YEARS        = Array.from({ length: 20 }, (_, i) => String(2025 - i));
const CITIES       = Object.keys(CITY_DEMAND).sort();
const FUELS        = ['Petrol', 'Diesel', 'Electric', 'CNG', 'Hybrid'];
const TRANSMISSIONS = ['Manual', 'Automatic', 'CVT', 'DCT', 'AMT', 'IMT'];
const CONDITIONS   = ['Excellent', 'Good', 'Average', 'Poor'];
const OWNERS       = ['1', '2', '3', '4+'];

const COLORS = [
  { name: 'White',  hex: '#f0f0f0', border: '#d0d0d0' },
  { name: 'Silver', hex: '#c0c0c0', border: '#a0a0a0' },
  { name: 'Grey',   hex: '#787878', border: '#555'     },
  { name: 'Black',  hex: '#1a1a1a', border: '#000'     },
  { name: 'Blue',   hex: '#1e5fa3', border: '#1447a0'  },
  { name: 'Red',    hex: '#c01b1b', border: '#a01818'  },
  { name: 'Brown',  hex: '#6d4c41', border: '#4e342e'  },
  { name: 'Beige',  hex: '#d7ccc8', border: '#bcaaa4'  },
  { name: 'Gold',   hex: '#d4a024', border: '#b88820'  },
  { name: 'Green',  hex: '#2e7a32', border: '#226026'  },
  { name: 'Orange', hex: '#d4531c', border: '#b84418'  },
  { name: 'Maroon', hex: '#78003f', border: '#5c0030'  },
];

const LUXURY_BRANDS  = new Set(['BMW','Mercedes-Benz','Audi','Lexus','Volvo','Land Rover','Jaguar','Porsche','Tesla']);
const PREMIUM_BRANDS = new Set(['Toyota','Honda','Volkswagen','Skoda','Kia','MG','Jeep','Ford','Renault','Nissan']);

/* ─── Helpers ───────────────────────────────────────────────────── */
function getSegment(brand) {
  if (!brand) return null;
  if (LUXURY_BRANDS.has(brand))  return 'luxury';
  if (PREMIUM_BRANDS.has(brand)) return 'premium';
  return 'economy';
}

function healthScore(inputs) {
  if (!inputs.brand) return 0;
  const age  = new Date().getFullYear() - Number(inputs.year || 2020);
  const km   = Number(inputs.mileage || 0);
  const own  = Number(inputs.ownerCount || 1);
  const cond = inputs.condition || 'Good';

  const ageS  = age <= 2 ? 100 : age <= 4 ? 85 : age <= 6 ? 70 : age <= 8 ? 55 : age <= 10 ? 40 : 25;
  const kmS   = km < 20000 ? 100 : km < 40000 ? 85 : km < 60000 ? 70 : km < 90000 ? 55 : km < 120000 ? 40 : 20;
  const ownS  = own === 1 ? 100 : own === 2 ? 70 : own === 3 ? 45 : 20;
  const condS = { Excellent:100, Good:75, Average:45, Poor:20 }[cond] ?? 60;

  return Math.round(ageS * 0.25 + kmS * 0.30 + ownS * 0.20 + condS * 0.25);
}

function healthMeta(score) {
  if (score >= 75) return { label: 'Strong Candidate',     color: '#15803d', fill: '#22c55e' };
  if (score >= 55) return { label: 'Viable Deal',          color: '#b45309', fill: '#f59e0b' };
  if (score >= 35) return { label: 'Review Carefully',     color: '#c2410c', fill: '#f97316' };
  return              { label: 'High Risk Asset',       color: '#be123c', fill: '#f43f5e' };
}

function formatReg(v) {
  return String(v || '').toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 11);
}

function formatLakh(n) {
  const v = Number(n || 0);
  return v >= 100000 ? `₹${(v / 100000).toFixed(2)}L` : v > 0 ? `₹${(v / 1000).toFixed(0)}k` : '';
}

function getValidFuels(brand, model) {
  const m = (model || '').toLowerCase();
  if (m.includes('ev') || (brand || '').toLowerCase() === 'tesla') return ['Electric'];
  return FUELS;
}

/* ─── Sub-components ────────────────────────────────────────────── */
function SectionHeader({ n, title, sub }) {
  return (
    <div className="vws-head">
      <div className="vws-num">{n}</div>
      <div>
        <div className="vws-title">{title}</div>
        {sub && <div className="vws-sub">{sub}</div>}
      </div>
    </div>
  );
}

function FieldLabel({ children, required }) {
  return (
    <label className="vws-label">
      {children}
      {required && <span className="vws-req" aria-hidden>*</span>}
    </label>
  );
}

/* ─── Main Component ────────────────────────────────────────────── */
export default function InputScreen() {
  const {
    inputs, updateInput,
    setValuationResult, setActiveScreen, setIsLoading, addEvaluation,
  } = useApp();

  const [brandCatalog, setBrandCatalog] = useState({});
  const [datasetCatalog, setDatasetCatalog] = useState({});
  const [registry, setRegistry]         = useState({ default: null, variants: [] });
  const [selectedVariant, setSelectedVariant] = useState('auto');
  const [loading, setLoading]           = useState(true);
  const [error, setError]               = useState('');
  const [submitting, setSubmitting]     = useState(false);

  /* Derived */
  const brandList   = useMemo(() => Object.keys(brandCatalog).sort(), [brandCatalog]);
  const modelList   = useMemo(() => brandCatalog[inputs.brand] || [], [brandCatalog, inputs.brand]);
  
  const variantList = useMemo(() => {
    if (!inputs.brand || !inputs.model) return [];

    // 1. Try resolving variants directly from dataset_catalog.json
    const brandKey = inputs.brand.trim().toLowerCase();
    const modelKey = inputs.model.trim().toLowerCase();

    let brandModels = datasetCatalog[brandKey];
    if (!brandModels) {
      // Find matching brand key
      const foundB = Object.keys(datasetCatalog).find(k => k.includes(brandKey) || brandKey.includes(k));
      if (foundB) brandModels = datasetCatalog[foundB];
    }

    if (brandModels) {
      let variants = brandModels[modelKey];
      if (!variants) {
        const foundM = Object.keys(brandModels).find(m => m.includes(modelKey) || modelKey.includes(m));
        if (foundM) variants = brandModels[foundM];
      }
      if (Array.isArray(variants) && variants.length > 0) {
        return variants.map(v => String(v).toUpperCase());
      }
    }

    return [];
  }, [inputs.brand, inputs.model, datasetCatalog]);

  const validFuels  = useMemo(() => getValidFuels(inputs.brand, inputs.model), [inputs.brand, inputs.model]);

  const segment  = getSegment(inputs.brand);
  const score    = healthScore(inputs);
  const meta     = healthMeta(score);
  const required = [inputs.brand, inputs.model, inputs.year, inputs.mileage, inputs.fuel, inputs.city].filter(Boolean).length;
  const isReady  = required === 6;

  /* Load brands, dataset catalog & registry */
  useEffect(() => {
    let alive = true;
    fetchBrands()
      .then(b => { if (alive) setBrandCatalog(b); })
      .catch(() => { if (alive) setError('Backend unavailable — run: uvicorn backend.main:app --reload'); })
      .finally(() => { if (alive) setLoading(false); });
    fetchCatalog()
      .then(cat => { if (alive && cat) setDatasetCatalog(cat); })
      .catch(() => {});
    fetchRegistry()
      .then(r => { if (alive && r) setRegistry(r); })
      .catch(() => {});
    return () => { alive = false; };
  }, []);


  /* Handlers */
  const onBrand = (b) => {
    updateInput('brand', b);
    const models = brandCatalog[b] || [];
    updateInput('model', models[0] || '');
    updateInput('variant', '');
  };

  const onModel = (m) => {
    updateInput('model', m);
    updateInput('variant', '');
  };

  const onSubmit = async () => {
    if (!isReady) return;
    setError('');
    setSubmitting(true);
    setIsLoading(true);
    setValuationResult(null);
    setActiveScreen('result');
    try {
      const payload = {
        ...inputs,
        // Send model and variant as separate clean fields — do NOT concatenate.
        // Concatenating "Swift" + "VXI" → "Swift VXI" causes an UNKNOWN category
        // hit in the ML model because the dataset stores them separately.
        model: inputs.model,
        variant: inputs.variant || 'unknown',
        modelVariant: selectedVariant,
      };
      const result = await runMLValuation(payload);

      setValuationResult(result);
      addEvaluation({ ...inputs }, result, 'Single Vehicle');
    } catch {
      setActiveScreen('input');
      setError('ML backend unavailable. Run: uvicorn backend.main:app --reload');
    } finally {
      setSubmitting(false);
      setIsLoading(false);
    }
  };

  /* ── Render ─────────────────────────────────────────────────── */
  return (
    <div className="vws-root">

      {/* ══════════════ LEFT: FORM ═══════════════════════════ */}
      <div className="vws-form">

        {/* Compact Form Header */}
        <div style={{ marginBottom: 16 }}>
          <h2 style={{ fontSize: 16, fontWeight: 800, color: 'var(--text-1)' }}>Vehicle Valuation Parameters</h2>
          <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>Configure all inputs side-by-side to estimate buy and sell pricing bands.</p>
        </div>

        {/* Row 1: Brand, Model, Year, Variant */}
        <div className="vws-row-4">
          <div className="vws-field">
            <FieldLabel required>Brand</FieldLabel>
            {loading ? (
              <div className="vws-skeleton" style={{ height: 38 }} />
            ) : (
              <SearchableDropdown
                options={brandList}
                value={inputs.brand}
                onChange={onBrand}
                placeholder="Brand"
                searchPlaceholder="Search brands…"
              />
            )}
          </div>
          <div className="vws-field">
            <FieldLabel required>Model</FieldLabel>
            <SearchableDropdown
              options={modelList}
              value={inputs.model}
              onChange={onModel}
              placeholder="Model"
              disabled={!inputs.brand || modelList.length === 0}
              searchPlaceholder="Search models…"
            />
          </div>
          <div className="vws-field">
            <FieldLabel required>Year</FieldLabel>
            <SearchableDropdown
              options={YEARS}
              value={inputs.year}
              onChange={v => updateInput('year', v)}
              placeholder="Year"
            />
          </div>
          <div className="vws-field">
            <FieldLabel>Variant</FieldLabel>
            <SearchableDropdown
              options={variantList}
              value={inputs.variant}
              onChange={v => updateInput('variant', v)}
              placeholder="Variant"
              disabled={!inputs.model || variantList.length === 0}
              searchPlaceholder="Search variants…"
            />
          </div>
        </div>

        {/* Row 2: Odometer, Fuel Type, Transmission, Owners */}
        <div className="vws-row-4">
          <div className="vws-field">
            <FieldLabel required>Odometer Reading</FieldLabel>
            <div className="vws-odo-wrap" style={{ display: 'flex', alignItems: 'center', position: 'relative' }}>
              <input
                className="vws-input"
                type="number"
                value={inputs.mileage || ''}
                onChange={e => updateInput('mileage', e.target.value)}
                placeholder="Odometer"
                min={0}
                style={{ width: '100%', paddingRight: '36px' }}
              />
              <span className="vws-odo-unit" style={{ position: 'absolute', right: '12px', fontSize: '12px', color: 'var(--text-3)' }}>km</span>
            </div>
          </div>
          <div className="vws-field">
            <FieldLabel required>Fuel Type</FieldLabel>
            <select
              className="vws-input field-select"
              value={inputs.fuel || ''}
              onChange={e => updateInput('fuel', e.target.value)}
            >
              <option value="">Select Fuel</option>
              {validFuels.map(f => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </div>
          <div className="vws-field">
            <FieldLabel>Transmission</FieldLabel>
            <select
              className="vws-input field-select"
              value={inputs.transmission || ''}
              onChange={e => updateInput('transmission', e.target.value)}
            >
              <option value="">Select Transmission</option>
              {TRANSMISSIONS.map(t => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
          <div className="vws-field">
            <FieldLabel>Owners</FieldLabel>
            <select
              className="vws-input field-select"
              value={inputs.ownerCount || ''}
              onChange={e => updateInput('ownerCount', e.target.value)}
            >
              <option value="">Select Owners</option>
              {OWNERS.map(o => (
                <option key={o} value={o.replace('+','')}>{o}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Row 3: Color, Target Margin %, Repair Budget, Registration No. */}
        <div className="vws-row-4">
          <div className="vws-field">
            <FieldLabel>Color</FieldLabel>
            <div style={{ position: 'relative' }}>
              {inputs.color && (
                <span style={{
                  position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)',
                  width: 14, height: 14, borderRadius: '50%', pointerEvents: 'none',
                  background: COLORS.find(c => c.name === inputs.color)?.hex || '#ccc',
                  border: `1.5px solid ${COLORS.find(c => c.name === inputs.color)?.border || '#aaa'}`,
                  zIndex: 1,
                }} />
              )}
              <select
                className="vws-input field-select"
                value={inputs.color || ''}
                onChange={e => updateInput('color', e.target.value)}
                style={{ paddingLeft: inputs.color ? 30 : 10 }}
              >
                <option value="">Select color</option>
                {COLORS.map(c => (
                  <option key={c.name} value={c.name}>{c.name}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="vws-field">
            <FieldLabel>Target Margin %</FieldLabel>
            <div className="vws-money-wrap" style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <input
                className="vws-input"
                type="number"
                min={8}
                max={30}
                step={1}
                value={inputs.targetMarginPct || 10}
                onChange={e => updateInput('targetMarginPct', e.target.value)}
                placeholder="10"
                style={{ paddingRight: 24 }}
              />
              <span style={{ position: 'absolute', right: 12, fontSize: 12, color: 'var(--text-3)' }}>%</span>
            </div>
          </div>
          <div className="vws-field">
            <FieldLabel>Repair Budget Estimate</FieldLabel>
            <div className="vws-money-wrap" style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <span style={{ position: 'absolute', left: 12, fontSize: 13, color: 'var(--text-3)' }}>₹</span>
              <input
                className="vws-input"
                type="number"
                value={inputs.repairBuffer || '25000'}
                onChange={e => updateInput('repairBuffer', e.target.value)}
                placeholder="25000"
                min={0}
                style={{ paddingLeft: 24 }}
              />
            </div>
          </div>
          <div className="vws-field">
            <FieldLabel>Registration No.</FieldLabel>
            <input
              className="vws-input vws-mono"
              type="text"
              value={inputs.vin || ''}
              onChange={e => updateInput('vin', formatReg(e.target.value))}
              placeholder="MH 01 AB 1234"
              maxLength={11}
            />
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="vws-error" style={{ marginTop: 12 }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
              <path d="M12 9v4M12 17h.01"/>
            </svg>
            {error}
          </div>
        )}

      </div>{/* /vws-form */}

      {/* ══════════════ RIGHT: SUMMARY PANEL ════════════════ */}
      <div className="vws-panel">
        <div className="vws-panel-inner">

          {/* Label */}
          <div className="vwsp-heading">Valuation Summary</div>

          {/* Vehicle identity card */}
          <div className="vwsp-card">
            <div className="vwsp-vehicle-name">
              {inputs.brand && inputs.model
                ? `${inputs.brand} ${inputs.model}`
                : <span style={{ color: 'var(--text-3)', fontWeight: 400 }}>No vehicle selected</span>}
            </div>
            {inputs.year && (
              <div className="vwsp-vehicle-sub">
                {[inputs.year, inputs.variant].filter(Boolean).join(' · ')}
              </div>
            )}
            <div className="vwsp-tags">
              {inputs.fuel         && <span className="vwsp-tag">{inputs.fuel}</span>}
              {inputs.transmission && <span className="vwsp-tag">{inputs.transmission}</span>}
              {inputs.ownerCount   && <span className="vwsp-tag">{inputs.ownerCount} Owner{inputs.ownerCount !== '1' ? 's' : ''}</span>}
              {Number(inputs.mileage) > 0 && (
                <span className="vwsp-tag">{(Number(inputs.mileage)/1000).toFixed(0)}k km</span>
              )}
              {inputs.condition    && <span className="vwsp-tag">{inputs.condition}</span>}
              {inputs.city         && <span className="vwsp-tag">{inputs.city}</span>}
            </div>
          </div>

          {/* Health score */}
          {inputs.brand && (
            <div className="vwsp-card">
              <div className="vwsp-stat-label">Deal Health Preview</div>
              <div className="vwsp-health-bar">
                <div
                  className="vwsp-health-fill"
                  style={{ width: `${score}%`, background: meta.fill }}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 6 }}>
                <span style={{ fontSize: 12.5, fontWeight: 600, color: meta.color }}>{meta.label}</span>
                <span style={{ fontSize: 12, fontWeight: 700, color: meta.color }}>{score}/100</span>
              </div>
            </div>
          )}

          {/* Stats grid */}
          {segment && (
            <div className="vwsp-grid">
              <div className="vwsp-stat">
                <div className="vwsp-stat-label">Segment</div>
                <div className="vwsp-stat-val" style={{
                  color: segment === 'luxury' ? '#7c3aed' : segment === 'premium' ? '#b45309' : '#2563eb',
                }}>
                  {segment.toUpperCase()}
                </div>
              </div>
              <div className="vwsp-stat">
                <div className="vwsp-stat-label">Fields Filled</div>
                <div className="vwsp-stat-val">{required}/6</div>
              </div>
              {inputs.sellerAskingPrice > 0 && (
                <div className="vwsp-stat">
                  <div className="vwsp-stat-label">Asking Price</div>
                  <div className="vwsp-stat-val">{formatLakh(inputs.sellerAskingPrice)}</div>
                </div>
              )}
              <div className="vwsp-stat">
                <div className="vwsp-stat-label">Target Margin</div>
                <div className="vwsp-stat-val">{inputs.targetMarginPct || 10}%</div>
              </div>
            </div>
          )}

          {/* Required fields checklist (only when not ready) */}
          {!isReady && inputs.brand && (
            <div className="vwsp-checklist">
              <div className="vwsp-check-head">Required fields</div>
              {[
                { key: 'brand',   label: 'Brand' },
                { key: 'model',   label: 'Model' },
                { key: 'year',    label: 'Year' },
                { key: 'mileage', label: 'Odometer' },
                { key: 'fuel',    label: 'Fuel type' },
                { key: 'city',    label: 'City' },
              ].map(f => {
                const done = !!inputs[f.key];
                return (
                  <div key={f.key} className={`vwsp-check-row${done ? ' done' : ''}`}>
                    <span className="vwsp-check-icon">
                      {done
                        ? <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#16a34a" strokeWidth="2.5" strokeLinecap="round"><polyline points="20 6 9 17 4 12"/></svg>
                        : <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#cbd5e1" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="9"/></svg>}
                    </span>
                    {f.label}
                  </div>
                );
              })}
            </div>
          )}

          <div style={{ flex: 1 }} />

          {/* Model Registry Selector */}
          <div className="vwsp-card" style={{ marginBottom: 16 }}>
            <div className="vwsp-stat-label" style={{ marginBottom: 6 }}>Model Variant Engine</div>
            <select
              value={selectedVariant}
              onChange={(e) => setSelectedVariant(e.target.value)}
              style={{
                width: '100%',
                padding: '8px 10px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                fontSize: '13px',
                background: '#f8fafc',
                color: '#1e293b',
                fontWeight: '500',
                outline: 'none',
                cursor: 'pointer',
              }}
            >
              <option value="auto">
                ⚡ Automatic (Best Model — {registry.default ? registry.default.replace('_', ' ').toUpperCase() : 'Default'})
              </option>
              {registry.variants && registry.variants.map((v) => (
                <option key={v.variant_id} value={v.variant_id}>
                  {v.variant_id.replace('_', ' ').toUpperCase()} ({v.dataset}) — MAPE: {v.metrics?.mape ? `${v.metrics.mape}%` : 'N/A'} {v.is_default ? '★ Active' : ''}
                </option>
              ))}
            </select>
          </div>

          {/* CTA */}
          <div className="vwsp-cta">

            <button
              className="vws-cta-btn"
              onClick={onSubmit}
              disabled={!isReady || submitting}
            >
              {/* ML lightning bolt icon */}
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
              </svg>
              {submitting ? 'Analysing…' : 'Analyse with ML'}
            </button>
            <div className="vws-cta-sub">
              CatBoost · LightGBM · XGBoost ensemble
            </div>
          </div>

        </div>
      </div>

    </div>
  );
}
