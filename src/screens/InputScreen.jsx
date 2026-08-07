import { useEffect, useMemo, useRef, useState } from 'react';
import { useApp } from '../context/AppContext.jsx';
import { LOCALITIES } from '../utils/mockData.js';
import { fetchBrands, fetchCatalog, fetchOptions, runMLValuation } from '../utils/apiValuation.js';
import SearchableDropdown from '../components/SearchableDropdown.jsx';

const CURRENT_YEAR = new Date().getFullYear();
const YEARS        = Array.from({ length: 25 }, (_, i) => String(CURRENT_YEAR - i));
const FUELS        = ['Petrol', 'Diesel', 'Electric', 'CNG', 'Hybrid'];
const TRANSMISSIONS = ['Manual', 'Automatic', 'CVT', 'DCT', 'AMT', 'IMT'];
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

function FieldLabel({ children, required }) {
  return (
    <label className="vws-label">
      {children}
      {required && <span className="vws-req" aria-hidden>*</span>}
    </label>
  );
}

const STRIP_TOKENS = new Set([
  'petrol', 'diesel', 'crdi', 'cng', 'lpg', 'electric', 'ev', 'vtvt', 'tdci', 'mpi', 'dci', 'ddis',
  'tsi', 'tdi', 'gdi', 'tgdi', 'cdti', 'idtec', 'ivtec', 'k10', 'k12', 'k15', 'boostjet', 'smart', 'hybrid',
  'at', 'mt', 'cvt', 'dct', 'amt', 'ivt', 'dsg', 'automatic', 'manual', 'str', 'shvs',
  'dsl', 'ptl', 'bs6', 'bs4', 'bsiv', 'bs3', 'unknown', 'nan', 'null', 'none', 'car', 'model', 'variant',
  '5sp', '6sp', '5-speed', '6-speed', '7-speed', '8-speed', '5mt', '6mt', '6at', '5at', 'speed',
  'drive', '2wd', '4wd', 'awd', '4x2', '4x4', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'
]);

function normalizeVariant(raw, modelName = '') {
  if (!raw || typeof raw !== 'string') return '';
  let text = raw.toLowerCase().trim();
  if (['unknown', 'nan', 'null', 'none', '-', '', 'base model'].includes(text)) return '';

  if (modelName) {
    modelName.toLowerCase().split(/\s+/).forEach(word => {
      if (word.length > 2) text = text.replace(word, '');
    });
  }

  text = text.replace(/\b\d+\.\d+l?\b|\b\d{3,4}cc?\b|\b\d+\.\d+\b/gi, '');
  text = text.replace(/[()[\]/\-,_.+]/g, ' ');

  const tokens = text.split(/\s+/).filter(t => t && !STRIP_TOKENS.has(t) && !/^\d+$/.test(t));
  if (tokens.length === 0) return '';

  let res = tokens.join(' ').toUpperCase();
  res = res.replace(/\bSX\s+O\b/g, 'SX (O)')
           .replace(/\bS\s+O\b/g, 'S (O)')
           .replace(/\bZX\s+O\b/g, 'ZX (O)')
           .replace(/\bZXI\s+PLUS\b/g, 'ZXI+')
           .replace(/\bVXI\s+PLUS\b/g, 'VXI+')
           .replace(/\bLXI\s+PLUS\b/g, 'LXI+')
           .replace(/\bXZ\s+PLUS\b/g, 'XZ+')
           .replace(/\bXT\s+PLUS\b/g, 'XT+');

  return res;
}

export default function InputScreen() {
  const {
    inputs, updateInput,
    setValuationResult, setActiveScreen, setIsLoading, addEvaluation,
  } = useApp();

  const [brandCatalog, setBrandCatalog]   = useState({});
  const [datasetCatalog, setDatasetCatalog] = useState({});

  const [availableFuels, setAvailableFuels]           = useState(FUELS);
  const [availableTransmissions, setAvailableTransmissions] = useState(TRANSMISSIONS);
  const [availableYears, setAvailableYears]           = useState(YEARS);
  const [optionsLoading, setOptionsLoading]           = useState(false);
  const optionsAbort = useRef(null);

  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState('');
  const [submitting, setSubmitting] = useState(false);

  const brandList   = useMemo(() => Object.keys(brandCatalog).sort(), [brandCatalog]);
  const modelList   = useMemo(() => brandCatalog[inputs.brand] || [], [brandCatalog, inputs.brand]);
  
  const variantList = useMemo(() => {
    if (!inputs.brand || !inputs.model) return [];

    const brandKey = inputs.brand.trim().toLowerCase();
    const modelKey = inputs.model.trim().toLowerCase();

    let brandModels = datasetCatalog[brandKey];
    if (!brandModels) {
      
      const foundB = Object.keys(datasetCatalog).find(k => k.includes(brandKey) || brandKey.includes(k));
      if (foundB) brandModels = datasetCatalog[foundB];
    }

    if (brandModels) {
      let rawVariants = brandModels[modelKey];
      if (!rawVariants) {
        const foundM = Object.keys(brandModels).find(m => m.includes(modelKey) || modelKey.includes(m));
        if (foundM) rawVariants = brandModels[foundM];
      }
      if (Array.isArray(rawVariants) && rawVariants.length > 0) {
        const uniqueSet = new Set();
        rawVariants.forEach(v => {
          const nv = normalizeVariant(v, inputs.model);
          if (nv && nv.length > 0) uniqueSet.add(nv);
          else if (v && typeof v === 'string' && !['unknown', 'nan', 'null', 'none'].includes(v.toLowerCase().trim())) {
            uniqueSet.add(v.trim().toUpperCase());
          }
        });
        return Array.from(uniqueSet).sort((a, b) => a.localeCompare(b));
      }
    }

    return [];
  }, [inputs.brand, inputs.model, datasetCatalog]);

  const segment  = getSegment(inputs.brand);
  const score    = healthScore(inputs);
  const meta     = healthMeta(score);
  const required = [inputs.brand, inputs.model, inputs.year, inputs.mileage, inputs.fuel, inputs.city].filter(Boolean).length;
  const isReady  = required === 6;

  useEffect(() => {
    let alive = true;
    fetchBrands()
      .then(b => { if (alive) setBrandCatalog(b); })
      .catch(() => { if (alive) setError('Backend unavailable — run: uvicorn backend.main:app --reload'); })
      .finally(() => { if (alive) setLoading(false); });
    fetchCatalog()
      .then(cat => { if (alive && cat) setDatasetCatalog(cat); })
      .catch(() => {});
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    if (!inputs.brand) {
      setAvailableFuels(FUELS);
      setAvailableTransmissions(TRANSMISSIONS);
      setAvailableYears(YEARS);
      return;
    }
    
    if (optionsAbort.current) optionsAbort.current = false;
    const token = {};
    optionsAbort.current = token;

    setOptionsLoading(true);
    fetchOptions({ brand: inputs.brand, model: inputs.model || undefined, variant: inputs.variant || undefined })
      .then(opts => {
        if (optionsAbort.current !== token) return; 
        setAvailableFuels(opts.fuel_types?.length   ? opts.fuel_types   : FUELS);
        setAvailableTransmissions(opts.transmissions?.length ? opts.transmissions : TRANSMISSIONS);
        setAvailableYears(opts.years?.length         ? opts.years        : YEARS);

        if (inputs.fuel && !opts.fuel_types?.includes(inputs.fuel))
          updateInput('fuel', '');
        if (inputs.transmission && !opts.transmissions?.includes(inputs.transmission))
          updateInput('transmission', '');
        if (inputs.year && !opts.years?.includes(inputs.year))
          updateInput('year', '');
      })
      .catch(() => {})
      .finally(() => { if (optionsAbort.current === token) setOptionsLoading(false); });
  
  }, [inputs.brand, inputs.model, inputs.variant]);

  const onBrand = (b) => {
    updateInput('brand', b);
    updateInput('model', '');
    updateInput('variant', '');
    updateInput('fuel', '');
    updateInput('transmission', '');
    updateInput('year', '');
  };

  const onModel = (m) => {
    updateInput('model', m);
    updateInput('variant', '');
    updateInput('fuel', '');
    updateInput('transmission', '');
    updateInput('year', '');
  };

  const onVariant = (v) => {
    updateInput('variant', v);
    
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
        
        model: inputs.model,
        variant: inputs.variant || 'unknown',
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

  return (
    <div className="vws-root">

      {}
      <div className="vws-form">

        {}
        <div style={{ marginBottom: 16 }}>
          <h2 style={{ fontSize: 16, fontWeight: 800, color: 'var(--text-1)' }}>Vehicle Valuation Parameters</h2>
          <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>Configure all inputs side-by-side to estimate buy and sell pricing bands.</p>
        </div>

        {}
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
              options={availableYears}
              value={inputs.year}
              onChange={v => updateInput('year', v)}
              placeholder={optionsLoading ? 'Loading…' : 'Year'}
              disabled={optionsLoading}
            />
          </div>
          <div className="vws-field">
            <FieldLabel>Variant</FieldLabel>
            <SearchableDropdown
              options={variantList}
              value={inputs.variant}
              onChange={onVariant}
              placeholder="Variant"
              disabled={!inputs.model || variantList.length === 0}
              searchPlaceholder="Search variants…"
            />
          </div>
        </div>

        {}
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
              disabled={optionsLoading}
            >
              <option value="">{optionsLoading ? 'Loading…' : 'Select Fuel'}</option>
              {availableFuels.map(f => (
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
              disabled={optionsLoading}
            >
              <option value="">{optionsLoading ? 'Loading…' : 'Select Transmission'}</option>
              {availableTransmissions.map(t => (
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

        {}
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
            <FieldLabel>Locality</FieldLabel>
            <SearchableDropdown
              options={LOCALITIES}
              value={inputs.locality || 'Indiranagar'}
              onChange={v => updateInput('locality', v)}
              placeholder="Select Locality"
              searchPlaceholder="Search locality…"
            />
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

        {}
        {error && (
          <div className="vws-error" style={{ marginTop: 12 }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
              <path d="M12 9v4M12 17h.01"/>
            </svg>
            {error}
          </div>
        )}

      </div>{}

      {}
      <div className="vws-panel">
        <div className="vws-panel-inner">

          {}
          <div className="vwsp-heading">Valuation Summary</div>

          {}
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

          {}
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

          {}
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

          {}
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

          {}
          <div className="vwsp-cta">

            <button
              className="vws-cta-btn"
              onClick={onSubmit}
              disabled={!isReady || submitting}
            >
              {}
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
