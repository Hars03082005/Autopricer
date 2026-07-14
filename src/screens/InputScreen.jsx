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

  /* Load brands */
  useEffect(() => {
    let alive = true;
    fetchBrands()
      .then(b => { if (alive) setBrandCatalog(b); })
      .catch(() => { if (alive) setError('Backend unavailable — run: uvicorn backend.main:app --reload'); })
      .finally(() => { if (alive) setLoading(false); });
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

  /* ── Render ─────────────────────────────────────────────────── */
  return (
    <div className="vws-root">

      {/* ══════════════ LEFT: FORM ═══════════════════════════ */}
      <div className="vws-form">

        {/* ── § 1 Vehicle Identity ─────────────────────────── */}
        <section className="vws-section">
          <SectionHeader n="1" title="Vehicle Identity" sub="Manufacturer, model, and registration year" />

          <div className="vws-row-2">
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
          </div>

          <div className="vws-row-3">
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
          </div>
        </section>

        <div className="vws-divider" />

        {/* ── § 2 Physical Condition ───────────────────────── */}
        <section className="vws-section">
          <SectionHeader n="2" title="Physical Condition" sub="Mileage, powertrain, ownership and state" />

          {/* Odometer — prominent */}
          <div className="vws-field" style={{ marginBottom: 20 }}>
            <FieldLabel required>Odometer Reading</FieldLabel>
            <div className="vws-odo-wrap">
              <input
                className="vws-odo"
                type="number"
                value={inputs.mileage || ''}
                onChange={e => updateInput('mileage', e.target.value)}
                placeholder="0"
                min={0}
              />
              <span className="vws-odo-unit">km</span>
            </div>
            {Number(inputs.mileage) > 0 && (
              <div className="vws-hint">
                {Number(inputs.mileage) >= 100000
                  ? `${(Number(inputs.mileage)/1000).toFixed(0)}k km · High mileage`
                  : Number(inputs.mileage) >= 50000
                  ? `${(Number(inputs.mileage)/1000).toFixed(0)}k km · Moderate`
                  : `${(Number(inputs.mileage)/1000).toFixed(0)}k km · Low mileage`}
              </div>
            )}
          </div>

          {/* Fuel Type */}
          <div className="vws-field">
            <FieldLabel required>Fuel Type</FieldLabel>
            <div className="vws-chips">
              {validFuels.map(f => (
                <button
                  key={f}
                  type="button"
                  className={`vws-chip${inputs.fuel === f ? ' vws-chip-on' : ''}`}
                  onClick={() => updateInput('fuel', f)}
                >
                  {f}
                </button>
              ))}
            </div>
          </div>

          {/* Transmission */}
          <div className="vws-field">
            <FieldLabel>Transmission</FieldLabel>
            <div className="vws-chips">
              {TRANSMISSIONS.map(t => (
                <button
                  key={t}
                  type="button"
                  className={`vws-chip${inputs.transmission === t ? ' vws-chip-on' : ''}`}
                  onClick={() => updateInput('transmission', t)}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>

          {/* Owner count + Color */}
          <div className="vws-row-2">
            <div className="vws-field">
              <FieldLabel>Owners</FieldLabel>
              <div className="vws-chips">
                {OWNERS.map(o => (
                  <button
                    key={o}
                    type="button"
                    className={`vws-chip vws-chip-sq${inputs.ownerCount === o.replace('+','') ? ' vws-chip-on' : ''}`}
                    onClick={() => updateInput('ownerCount', o.replace('+',''))}
                  >
                    {o}
                  </button>
                ))}
              </div>
            </div>
            <div className="vws-field">
              <FieldLabel>Color</FieldLabel>
              <div className="vws-color-row">
                {COLORS.map(c => (
                  <button
                    key={c.name}
                    type="button"
                    className={`vws-color-dot${inputs.color === c.name ? ' vws-color-sel' : ''}`}
                    style={{ background: c.hex, borderColor: c.border }}
                    onClick={() => updateInput('color', c.name)}
                    title={c.name}
                  />
                ))}
              </div>
              {inputs.color && (
                <div className="vws-hint">{inputs.color}</div>
              )}
            </div>
          </div>

          {/* Condition */}
          <div className="vws-field">
            <FieldLabel>Physical Condition</FieldLabel>
            <div className="vws-chips">
              {CONDITIONS.map(c => (
                <button
                  key={c}
                  type="button"
                  className={`vws-chip vws-chip-cond-${c.toLowerCase()}${inputs.condition === c ? ' vws-chip-on-cond' : ''}`}
                  onClick={() => updateInput('condition', c)}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>
        </section>

        <div className="vws-divider" />

        {/* ── § 3 Inspection ───────────────────────────────── */}
        <section className="vws-section">
          <SectionHeader n="3" title="Inspection" sub="Pre-sale certification status" />

          <div
            className={`vws-inspect${inputs.inspected ? ' vws-inspect-on' : ''}`}
            onClick={() => updateInput('inspected', !inputs.inspected)}
            role="checkbox"
            aria-checked={!!inputs.inspected}
            tabIndex={0}
            onKeyDown={e => (e.key === ' ' || e.key === 'Enter') && updateInput('inspected', !inputs.inspected)}
          >
            <div className="vws-toggle">
              <div className="vws-toggle-knob" />
            </div>
            <div className="vws-inspect-body">
              <div className="vws-inspect-label">Certified Multi-Point Inspection</div>
              <div className="vws-inspect-sub">Vehicle has been inspected by a qualified technician</div>
            </div>
            {inputs.inspected && (
              <span className="vws-badge-verified">Verified</span>
            )}
          </div>
        </section>

        <div className="vws-divider" />

        {/* ── § 4 Market Context ───────────────────────────── */}
        <section className="vws-section">
          <SectionHeader n="4" title="Market Context" sub="Target city and seller's asking price" />

          <div className="vws-row-2">
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
            <div className="vws-field">
              <FieldLabel>Seller Asking Price</FieldLabel>
              <div className="vws-money-wrap">
                <span className="vws-money-pfx">₹</span>
                <input
                  className="vws-input vws-money"
                  type="number"
                  value={inputs.sellerAskingPrice === '0' ? '' : inputs.sellerAskingPrice}
                  onChange={e => updateInput('sellerAskingPrice', e.target.value || '0')}
                  placeholder="0"
                  min={0}
                />
              </div>
              {Number(inputs.sellerAskingPrice) > 0 && (
                <div className="vws-hint">{formatLakh(inputs.sellerAskingPrice)}</div>
              )}
            </div>
          </div>
        </section>

        <div className="vws-divider" />

        {/* ── § 5 Dealer Preferences ───────────────────────── */}
        <section className="vws-section">
          <SectionHeader n="5" title="Dealer Preferences" sub="Target return and repair budget" />

          {/* Margin slider */}
          <div className="vws-field">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <FieldLabel>Target Margin</FieldLabel>
              <span className="vws-margin-val">{inputs.targetMarginPct || 15}%</span>
            </div>
            <input
              type="range"
              min={8} max={25} step={1}
              className="vws-slider"
              value={inputs.targetMarginPct || 15}
              onChange={e => updateInput('targetMarginPct', e.target.value)}
              style={{ '--pct': `${((Number(inputs.targetMarginPct || 15) - 8) / 17) * 100}%` }}
            />
            <div className="vws-slider-labels">
              <span>8% — Conservative</span>
              <span>25% — Aggressive</span>
            </div>
          </div>

          {/* Repair buffer */}
          <div className="vws-field">
            <FieldLabel>Repair Budget Estimate</FieldLabel>
            <div className="vws-money-wrap">
              <span className="vws-money-pfx">₹</span>
              <input
                className="vws-input vws-money"
                type="number"
                value={inputs.repairBuffer || '25000'}
                onChange={e => updateInput('repairBuffer', e.target.value)}
                placeholder="25000"
                min={0}
              />
            </div>
            <div className="vws-hint">Pre-sale reconditioning estimate</div>
          </div>
        </section>

        {/* Error */}
        {error && (
          <div className="vws-error">
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
              CatBoost · LightGBM · XGBoost ensemble
            </div>
          </div>

        </div>
      </div>

    </div>
  );
}
