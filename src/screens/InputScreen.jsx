import { useEffect, useMemo, useState } from 'react';
import { useApp } from '../context/AppContext.jsx';
import { CITY_DEMAND } from '../utils/mockData.js';
import { fetchBrands, runMLValuation } from '../utils/apiValuation.js';
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

const VARIANT_CATALOG = {
  'Swift':        ['LXi','VXi','ZXi','ZXi+','LDi','VDi','ZDi','ZDi+'],
  'Baleno':       ['Sigma','Delta','Zeta','Alpha','Delta Turbo','Zeta Turbo','Alpha Turbo'],
  'WagonR':       ['LXi','VXi','ZXi','ZXi+','LXi CNG','VXi CNG'],
  'Vitara Brezza':['LXi','VXi','ZXi','ZXi+'],
  'Grand Vitara': ['Sigma','Delta','Zeta','Alpha','Zeta Hybrid','Alpha Hybrid'],
  'Creta':        ['E','EX','S','S+','SX','SX Tech','SX(O)'],
  'i20':          ['Era','Magna','Sportz','Asta','Asta(O)','N Line N6','N Line N8'],
  'Venue':        ['E','S','S+','SX','SX(O)'],
  'Nexon':        ['Smart','Smart+','Pure','Creative','Fearless','Fearless+'],
  'Nexon EV':     ['Medium Range','Long Range','Max'],
  'Harrier':      ['Smart','Smart+','Pure','Adventure','Fearless','Fearless+'],
  'City':         ['SV','V','VX','ZX','RS'],
  'Fortuner':     ['2WD MT','2WD AT','4WD AT','Legender 2WD AT'],
  'Innova Crysta':['GX MT','GX AT','VX MT','VX AT','ZX AT'],
  'Seltos':       ['HTE','HTK','HTK+','HTX','HTX+','GTX+'],
  'Sonet':        ['HTE','HTK','HTK+','HTX','HTX+','GTX+'],
  'Thar':         ['AX Opt','LX Petrol MT','LX Diesel MT 4WD','LX Diesel AT 4WD'],
  'XUV700':       ['MX','AX3','AX5','AX7'],
  'Scorpio N':    ['Z2','Z4','Z6','Z8','Z8 L'],
  '3 Series':     ['320i Sport','320d Sport','330i M Sport','M340i xDrive'],
  '5 Series':     ['520d Luxury','520d M Sport','530d M Sport'],
};

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

function SectionHeader({ n, title, sub }) {
  return (
    <div className="vws-head" style={{ marginBottom: 16 }}>
      <div className="vws-num">{n}</div>
      <div>
        <div className="vws-title" style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-1)' }}>{title}</div>
        {sub && <div className="vws-sub" style={{ fontSize: 12.5, color: 'var(--text-2)', fontWeight: 500 }}>{sub}</div>}
      </div>
    </div>
  );
}

function FieldLabel({ children, required }) {
  return (
    <label className="vws-label" style={{ fontSize: 13.5, fontWeight: 650, color: 'var(--text-1)', marginBottom: 6 }}>
      {children}
      {required && <span className="vws-req" style={{ color: '#dc2626', marginLeft: 3 }}>*</span>}
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
  const [loading, setLoading]           = useState(true);
  const [error, setError]               = useState('');
  const [submitting, setSubmitting]     = useState(false);

  /* Derived */
  const brandList   = useMemo(() => Object.keys(brandCatalog).sort(), [brandCatalog]);
  const modelList   = useMemo(() => brandCatalog[inputs.brand] || [], [brandCatalog, inputs.brand]);
  const variantList = useMemo(() => {
    if (!inputs.model) return [];
    const direct  = VARIANT_CATALOG[inputs.model];
    const stripped = inputs.model.replace(new RegExp(`^${inputs.brand}\\s+`, 'i'), '');
    return direct || VARIANT_CATALOG[stripped] || [];
  }, [inputs.brand, inputs.model]);
  const validFuels  = useMemo(() => getValidFuels(inputs.brand, inputs.model), [inputs.brand, inputs.model]);

  const segment  = getSegment(inputs.brand);
  const score    = healthScore(inputs);
  const meta     = healthMeta(score);
  const required = [inputs.brand, inputs.model, inputs.year, inputs.mileage, inputs.fuel, inputs.city].filter(Boolean).length;
  const isReady  = required === 6;

  /* Fetch brands seed */
  useEffect(() => {
    let active = true;
    fetchBrands()
      .then(data => { if (active) { setBrandCatalog(data); setLoading(false); } })
      .catch(() => { if (active) { setLoading(false); setError('Failed to load manufacturers catalogue.'); } });
    return () => { active = false; };
  }, []);

  /* Route triggers */
  const onBrand = (b) => {
    updateInput('brand', b);
    updateInput('model', '');
    updateInput('variant', '');
  };

  const onModel = (m) => {
    updateInput('model', m);
    updateInput('variant', '');
    // Auto preset fuel and trans defaults if match found
    const f = getValidFuels(inputs.brand, m);
    if (f.length === 1) updateInput('fuel', f[0]);
  };

  const onSubmit = async () => {
    if (!isReady || submitting) return;
    setSubmitting(true);
    setError('');
    setIsLoading(true);
    setValuationResult(null);
    setActiveScreen('result');
    try {
      const payload = {
        ...inputs,
        model: inputs.variant ? `${inputs.model} ${inputs.variant}` : inputs.model,
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

      {/* ══════════════ LEFT: FORM ═══════════════════════════ */}
      <div className="vws-form">

        {/* ── § 1 Vehicle Identity & Specs (Dense Row-by-Row Layout) ── */}
        <section className="vws-section">
          <SectionHeader n="1" title="Vehicle Identity & Specifications" sub="Configure vehicle tags, options, and model credentials" />

          {/* Row 1: Brand, Model, Variant */}
          <div className="vws-row-3">
            <div className="vws-field">
              <FieldLabel required>Brand</FieldLabel>
              {loading ? (
                <div className="vws-skeleton" style={{ height: 38 }} />
              ) : (
                <SearchableDropdown
                  options={brandList}
                  value={inputs.brand}
                  onChange={onBrand}
                  placeholder="Select manufacturer"
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
                placeholder={inputs.brand ? 'Select model' : 'Select brand first'}
                disabled={!inputs.brand || modelList.length === 0}
                searchPlaceholder="Search models…"
              />
            </div>
            <div className="vws-field">
              <FieldLabel>Variant</FieldLabel>
              <SearchableDropdown
                options={variantList}
                value={inputs.variant}
                onChange={v => updateInput('variant', v)}
                placeholder={inputs.model ? 'Select variant' : '—'}
                disabled={!inputs.model || variantList.length === 0}
                searchPlaceholder="Search variants…"
              />
            </div>
          </div>

          {/* Row 2: Year, Reg No, City */}
          <div className="vws-row-3">
            <div className="vws-field">
              <FieldLabel required>Year</FieldLabel>
              <SearchableDropdown
                options={YEARS}
                value={inputs.year}
                onChange={v => updateInput('year', v)}
                placeholder="Select year"
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
            <div className="vws-field">
              <FieldLabel required>City</FieldLabel>
              <SearchableDropdown
                options={CITIES}
                value={inputs.city}
                onChange={v => updateInput('city', v)}
                placeholder="Select city"
                searchPlaceholder="Search cities…"
              />
            </div>
          </div>

          {/* Row 3: Odometer, Fuel Type, Transmission */}
          <div className="vws-row-3">
            <div className="vws-field">
              <FieldLabel required>Odometer Reading (km)</FieldLabel>
              <input
                className="vws-input"
                type="number"
                value={inputs.mileage || ''}
                onChange={e => updateInput('mileage', e.target.value)}
                placeholder="Enter km reading…"
                min={0}
              />
            </div>
            <div className="vws-field">
              <FieldLabel required>Fuel Type</FieldLabel>
              <select
                className="vws-input"
                value={inputs.fuel || ''}
                onChange={e => updateInput('fuel', e.target.value)}
              >
                <option value="">Select fuel type</option>
                {validFuels.map(f => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
            </div>
            <div className="vws-field">
              <FieldLabel>Transmission</FieldLabel>
              <select
                className="vws-input"
                value={inputs.transmission || ''}
                onChange={e => updateInput('transmission', e.target.value)}
              >
                <option value="">Select transmission</option>
                {TRANSMISSIONS.map(t => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Row 4: Owners, Color, Physical Condition */}
          <div className="vws-row-3">
            <div className="vws-field">
              <FieldLabel>Owners</FieldLabel>
              <select
                className="vws-input"
                value={inputs.ownerCount || ''}
                onChange={e => updateInput('ownerCount', e.target.value)}
              >
                <option value="">Select owners</option>
                {OWNERS.map(o => (
                  <option key={o} value={o.replace('+','')}>{o} Owner{o !== '1' ? 's' : ''}</option>
                ))}
              </select>
            </div>
            <div className="vws-field">
              <FieldLabel>Color</FieldLabel>
              <select
                className="vws-input"
                value={inputs.color || ''}
                onChange={e => updateInput('color', e.target.value)}
              >
                <option value="">Select color</option>
                {COLORS.map(c => (
                  <option key={c.name} value={c.name}>{c.name}</option>
                ))}
              </select>
            </div>
            <div className="vws-field">
              <FieldLabel>Physical Condition</FieldLabel>
              <select
                className="vws-input"
                value={inputs.condition || ''}
                onChange={e => updateInput('condition', e.target.value)}
              >
                <option value="">Select condition</option>
                {CONDITIONS.map(c => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>
          </div>
        </section>

        <div className="vws-divider" />

        {/* ── § 2 Acquisition & Dealer Preferences ── */}
        <section className="vws-section">
          <SectionHeader n="2" title="Acquisition & Dealer Preferences" sub="Configure target dealer margins, inspection status, and repair budgets" />

          {/* Row 5: Asking Price & Target Margin */}
          <div className="vws-row-2">
            <div className="vws-field">
              <FieldLabel>Seller Asking Price (₹)</FieldLabel>
              <div className="vws-money-wrap">
                <span className="vws-money-pfx" style={{ fontSize: 14, fontWeight: 700 }}>₹</span>
                <input
                  className="vws-input vws-money"
                  type="number"
                  value={inputs.sellerAskingPrice === '0' ? '' : inputs.sellerAskingPrice}
                  onChange={e => updateInput('sellerAskingPrice', e.target.value || '0')}
                  placeholder="e.g. 550000"
                  min={0}
                />
              </div>
            </div>
            <div className="vws-field">
              <FieldLabel>Target Dealer Margin (%)</FieldLabel>
              <div style={{ position: 'relative' }}>
                <input
                  className="vws-input"
                  type="number"
                  value={inputs.targetMarginPct || 15}
                  onChange={e => updateInput('targetMarginPct', e.target.value)}
                  placeholder="15"
                  min={1}
                  max={100}
                  style={{ paddingRight: 32 }}
                />
                <span style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', fontWeight: 700, color: 'var(--text-1)', fontSize: 13.5 }}>%</span>
              </div>
            </div>
          </div>

          {/* Row 6: Repair Budget & Certified Inspection */}
          <div className="vws-row-2">
            <div className="vws-field">
              <FieldLabel>Repair Budget Estimate (₹)</FieldLabel>
              <div className="vws-money-wrap">
                <span className="vws-money-pfx" style={{ fontSize: 14, fontWeight: 700 }}>₹</span>
                <input
                  className="vws-input vws-money"
                  type="number"
                  value={inputs.repairBuffer || '25000'}
                  onChange={e => updateInput('repairBuffer', e.target.value)}
                  placeholder="25000"
                  min={0}
                />
              </div>
            </div>
            <div className="vws-field" style={{ justifyContent: 'center' }}>
              <div
                className={`vws-inspect${inputs.inspected ? ' vws-inspect-on' : ''}`}
                onClick={() => updateInput('inspected', !inputs.inspected)}
                role="checkbox"
                aria-checked={!!inputs.inspected}
                tabIndex={0}
                onKeyDown={e => (e.key === ' ' || e.key === 'Enter') && updateInput('inspected', !inputs.inspected)}
                style={{ marginTop: 12, height: 40 }}
              >
                <div className="vws-toggle">
                  <div className="vws-toggle-knob" />
                </div>
                <div className="vws-inspect-body">
                  <div className="vws-inspect-label" style={{ fontSize: 12.5, fontWeight: 700 }}>Certified Inspection</div>
                </div>
                {inputs.inspected && (
                  <span className="vws-badge-verified" style={{ padding: '2px 8px', fontSize: 10.5 }}>Verified</span>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* Error banner */}
        {error && (
          <div className="vws-error" style={{ marginTop: 16 }}>
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
                <span style={{ fontSize: 12.5, fontWeight: 650, color: meta.color }}>{meta.label}</span>
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
                <div className="vwsp-stat-val">{inputs.targetMarginPct || 15}%</div>
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
              PriceRef Advanced Machine Learning Valuation Engine
            </div>
          </div>

        </div>
      </div>

    </div>
  );
}
