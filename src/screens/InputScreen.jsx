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
  // ── Maruti ────────────────────────────────────────────────────────
  'Swift':          ['LXi','VXi','ZXi','ZXi+','LDi','VDi','ZDi','ZDi+'],
  'Dzire':          ['LXi','VXi','ZXi','ZXi+','LDi','VDi','ZDi','ZDi+'],
  'Baleno':         ['Sigma','Delta','Zeta','Alpha','Delta Turbo','Zeta Turbo','Alpha Turbo'],
  'Alto':           ['LXi','VXi','VXi+'],
  'Alto K10':       ['STD','LXi','VXi','VXi+','VXi AGS'],
  'WagonR':         ['LXi','VXi','ZXi','ZXi+','LXi CNG','VXi CNG'],
  'Vitara Brezza':  ['LXi','VXi','ZXi','ZXi+'],
  'Grand Vitara':   ['Sigma','Delta','Zeta','Alpha','Zeta Hybrid','Alpha Hybrid'],
  'Ertiga':         ['LXi','VXi','ZXi','ZXi+','LDi','VDi','ZDi'],
  'Ciaz':           ['Sigma','Delta','Zeta','Alpha'],
  'S-Cross':        ['Sigma','Delta','Zeta','Alpha'],
  'Ignis':          ['Sigma','Delta','Zeta','Alpha'],
  'Celerio':        ['LXi','VXi','ZXi','ZXi+'],
  'S-Presso':       ['STD','LXi','VXi','VXi+','VXi AGS'],
  'XL6':            ['Zeta','Alpha','Zeta MT','Alpha MT'],

  // ── Hyundai ───────────────────────────────────────────────────────
  'Creta':          ['E','EX','S','S+','SX','SX Tech','SX(O)'],
  'i10':            ['ERA','Magna','Sportz','Asta'],
  'i20':            ['Era','Magna','Sportz','Asta','Asta(O)','N Line N6','N Line N8'],
  'Verna':          ['E','EX','S','SX','SX(O)','SX Turbo'],
  'Venue':          ['E','S','S+','SX','SX(O)'],
  'Alcazar':        ['Prestige','Prestige (O)','Platinum','Platinum (O)'],
  'Tucson':         ['GLS 2WD AT','GLS 4WD AT','Signature 2WD AT','Signature 4WD AT'],
  'Elantra':        ['S MT','S AT','SX MT','SX AT'],
  'Santro':         ['ERA','Magna','Sportz','Asta','Asta AMT'],
  'Aura':           ['E','S','SX','SX+'],
  'Ioniq 5':        ['Standard Range RWD','Long Range RWD','Long Range AWD'],
  'Kona':           ['E','S','SX'],

  // ── Tata ──────────────────────────────────────────────────────────
  'Nexon':          ['Smart','Smart+','Pure','Creative','Fearless','Fearless+'],
  'Harrier':        ['Smart','Smart+','Pure','Adventure','Fearless','Fearless+'],
  'Safari':         ['Smart','Smart+','Pure','Adventure','Fearless','Fearless+'],
  'Tiago':          ['XE','XM','XT','XZ','XZ+','XZA','XZA+'],
  'Tigor':          ['XE','XM','XT','XZ','XZ+'],
  'Altroz':         ['XE','XM','XT','XZ','XZ+','XT Turbo','XZ Turbo','XZ+ Turbo'],
  'Punch':          ['Pure','Adventure','Accomplished','Creative'],
  'Hexa':           ['XE','XM','XT','XTA','XMA'],
  'Bolt':           ['XE','XM','XT','XZ'],
  'Zest':           ['XE','XM','XT','XZ'],

  // ── Honda ─────────────────────────────────────────────────────────
  'City':           ['SV','V','VX','ZX','RS'],
  'Amaze':          ['E','S','V','VX','SX MT','SX CVT'],
  'Jazz':           ['S','V','VX','ZX'],
  'WR-V':           ['S','V','VX'],
  'CR-V':           ['2WD CVT','4WD CVT'],
  'Civic':          ['V CVT','ZX CVT','ZX Diesel MT'],
  'Accord':         ['2.4L AT','3.5L AT'],
  'BR-V':           ['S MT','V MT','V AT'],
  'HR-V':           ['V CVT','VX CVT'],

  // ── Toyota ────────────────────────────────────────────────────────
  'Innova':         ['GX MT','GX AT','VX MT','VX AT','ZX AT'],
  'Fortuner':       ['2WD MT','2WD AT','4WD AT','Legender 2WD AT'],
  'Camry':          ['Hybrid'],
  'Corolla':        ['J','G','V','Altis'],
  'Glanza':         ['E','S','G','V'],
  'Urban Cruiser':  ['Mid','High','Premium'],
  'Hilux':          ['Standard','High'],
  'Yaris':          ['J','G','V','SV'],
  'Etios':          ['J','G','V','VX'],
  'Vellfire':       ['ZX'],

  // ── Mahindra ──────────────────────────────────────────────────────
  'Scorpio':        ['S3','S5','S7','S9','S11'],
  'XUV500':         ['W3','W5','W7','W9','W11'],
  'XUV300':         ['W4','W6','W8','W8 (O)'],
  'XUV700':         ['MX','AX3','AX5','AX7'],
  'Thar':           ['AX Opt','LX Petrol MT','LX Diesel MT 4WD','LX Diesel AT 4WD'],
  'Bolero':         ['SLX','SLE','SLT','B4','B6'],
  'KUV100':         ['K2','K4','K6','K6+'],
  'Marazzo':        ['M2','M4','M6','M8'],
  'Alturas G4':     ['2WD AT','4WD AT'],
  'BE6':            ['Pack One','Pack Two','Pack Three'],
  'XEV9e':          ['Pack One','Pack Two'],

  // ── Kia ───────────────────────────────────────────────────────────
  'Seltos':         ['HTE','HTK','HTK+','HTX','HTX+','GTX+'],
  'Sonet':          ['HTE','HTK','HTK+','HTX','HTX+','GTX+'],
  'Carens':         ['Premium','Prestige','Prestige Plus','Luxury','Luxury Plus'],
  'Carnival':       ['Premium','Prestige'],
  'EV6':            ['GT Line RWD','GT Line AWD','GT AWD'],

  // ── Renault ───────────────────────────────────────────────────────
  'Kwid':           ['STD','RXE','RXL','RXT','RXT(O)','Climber'],
  'Duster':         ['RXE','RXL','RXS','RXZ'],
  'Triber':         ['RXE','RXL','RXT','RXZ','RXZ(O)'],
  'Kiger':          ['RXE','RXL','RXT','RXT(O)','RXZ','RXZ(O)'],
  'Captur':         ['RXE','RXL','RXT','Platine'],

  // ── Nissan ────────────────────────────────────────────────────────
  'Magnite':        ['XE','XL','XV','XV(O)','XV Premium','XV Premium(O)'],
  'Kicks':          ['XL','XV','XV Premium'],
  'Terrano':        ['XL','XV','XV D THP'],
  'Sunny':          ['XE','XL','XV'],
  'Micra':          ['XE','XL','XV'],

  // ── Volkswagen ────────────────────────────────────────────────────
  'Polo':           ['Trendline','Comfortline','Highline','GT TSI'],
  'Vento':          ['Comfortline','Highline','Highline AT','GT Plus TSI'],
  'Taigun':         ['Comfortline','Highline','GT Plus'],
  'Virtus':         ['Comfortline','Highline','GT'],
  'Tiguan':         ['Comfortline','Highline'],
  'T-Roc':          ['Sport'],

  // ── Skoda ─────────────────────────────────────────────────────────
  'Rapid':          ['Rider','Active','Ambition','Style'],
  'Octavia':        ['Ambition','Style'],
  'Superb':         ['Style','Sportline'],
  'Kushaq':         ['Active','Ambition','Style'],
  'Slavia':         ['Active','Ambition','Style'],
  'Kodiaq':         ['Style','Sportline'],
  'Karoq':          ['Style'],

  // ── Ford ──────────────────────────────────────────────────────────
  'EcoSport':       ['Ambiente','Trend','Titanium','Titanium+','S'],
  'Endeavour':      ['Trend AT 4x2','Titanium AT 4x2','Titanium AT 4x4'],
  'Figo':           ['Ambiente','Trend','Titanium'],
  'Freestyle':      ['Ambiente','Trend','Titanium'],
  'Aspire':         ['Ambiente','Trend','Titanium'],
  'Mustang':        ['Fastback'],

  // ── Jeep ──────────────────────────────────────────────────────────
  'Compass':        ['Sport','Sport+','Longitude','Limited','Limited Plus','Trailhawk'],
  'Meridian':       ['Longitude','Longitude (O)','Limited','X'],
  'Wrangler':       ['Unlimited Petrol AT'],
  'Grand Cherokee': ['Limited 4x4','Overland 4x4','Summit'],

  // ── MG ────────────────────────────────────────────────────────────
  'Hector':         ['Style','Super','Smart','Sharp','Savvy'],
  'Astor':          ['Style','Super','Smart','Sharp'],
  'Gloster':        ['Super 2WD','Sharp 2WD','Sharp 4WD','Savvy AWD'],
  'ZS EV':          ['Excite','Exclusive'],
  'Comet EV':       ['EX','Executive'],

  // ── BMW ───────────────────────────────────────────────────────────
  '3 Series':       ['320i Sport','320d Sport','330i M Sport','M340i xDrive'],
  '5 Series':       ['520d Luxury','520d M Sport','530d M Sport'],
  '7 Series':       ['730Ld DPE','730Ld M Sport','740Li DPE','M760Li xDrive'],
  'X1':             ['sDrive20i xLine','sDrive20d xLine','xDrive28i M Sport'],
  'X3':             ['xDrive20d Luxury','xDrive30d M Sport','xDrive30i M Sport'],
  'X5':             ['xDrive30d Sport','xDrive40i M Sport','xDrive40i Pure Excellence'],
  'X7':             ['xDrive40i DPE','xDrive40i M Sport'],
  'M3':             ['Competition'],
  'M5':             ['Competition'],
  'i4':             ['eDrive40 M Sport','M50 xDrive'],
  'iX':             ['xDrive40','xDrive50'],

  // ── Mercedes-Benz ─────────────────────────────────────────────────
  'A-Class':        ['A 200 Progressive Line','A 200 AMG Line'],
  'C-Class':        ['C 200 Progressive','C 200 AMG Line','C 300d AMG Line'],
  'E-Class':        ['E 200 Exclusive','E 220d','E 350d'],
  'S-Class':        ['S 450d','S 580 Maybach'],
  'GLA':            ['GLA 200','GLA 220d 4MATIC'],
  'GLC':            ['GLC 220d','GLC 300d 4MATIC'],
  'GLE':            ['GLE 300d 4MATIC','GLE 450 4MATIC'],
  'GLS':            ['GLS 400d 4MATIC'],
  'AMG GT':         ['AMG GT 63 S'],
  'EQS':            ['EQS 580 4MATIC'],
  'EQB':            ['EQB 300 4MATIC'],

  // ── Audi ──────────────────────────────────────────────────────────
  'A3':             ['30 TFSI Premium Plus','35 TFSI Technology'],
  'A4':             ['30 TFSI Premium Plus','35 TFSI Technology','45 TFSI Technology'],
  'A6':             ['45 TFSI Technology','55 TFSI Technology'],
  'A8':             ['L 55 TFSI','L 60 TFSI e'],
  'Q3':             ['30 TFSI Premium Plus','35 TFSI Technology'],
  'Q5':             ['40 TFSI Technology','45 TFSI Technology'],
  'Q7':             ['45 TFSI Technology','55 TFSI Technology'],
  'Q8':             ['55 TFSI Technology'],
  'e-tron':         ['50 quattro','55 quattro','GT quattro'],
  'RS5':            ['Sportback'],
  'TT':             ['TTS Coupe'],

  // ── Volvo ─────────────────────────────────────────────────────────
  'XC40':           ['B4 AWD Momentum','B4 AWD R-Design','Recharge Pure Electric'],
  'XC60':           ['B5 AWD Momentum','B5 AWD R-Design','T8 Recharge'],
  'XC90':           ['B6 AWD Momentum','B6 AWD R-Design','T8 Recharge'],
  'S60':            ['B4 Momentum','B4 R-Design'],
  'S90':            ['B6 AWD Momentum','B6 AWD R-Design'],
  'V60':            ['B4 Momentum','B4 R-Design'],

  // ── Land Rover ────────────────────────────────────────────────────
  'Defender':       ['90 S','90 X-Dynamic S','110 S','110 X-Dynamic S','110 X'],
  'Discovery':      ['SE 2.0 TD4','HSE 3.0 TD6'],
  'Range Rover':    ['HSE 3.0 Diesel','Autobiography 3.0 Diesel'],
  'Range Rover Sport': ['HSE 2.0 Diesel','HSE 3.0 Diesel','Autobiography 3.0 Diesel'],
  'Range Rover Evoque': ['SE 2.0 Diesel','HSE 2.0 Diesel','HSE Dynamic 2.0 Diesel'],
  'Freelander':     ['SE TD4','HSE TD4'],

  // ── Porsche ───────────────────────────────────────────────────────
  'Cayenne':        ['E-Hybrid','Turbo','Turbo S E-Hybrid'],
  'Macan':          ['2.0 Petrol','S 3.0 Petrol','GTS'],
  'Panamera':       ['4 Executive','4S Executive','Turbo S E-Hybrid'],
  'Taycan':         ['RWD','4','4S','Turbo','Turbo S'],
  '911':            ['Carrera','Carrera S','Carrera 4S','GT3'],
  'Boxster':        ['718 Boxster','718 Boxster S'],

  // ── Jaguar ────────────────────────────────────────────────────────
  'XE':             ['Prestige','Portfolio','R-Sport'],
  'XF':             ['Prestige','Portfolio','R-Sport'],
  'XJ':             ['Premium Luxury','L Portfolio'],
  'F-Pace':         ['Prestige','Portfolio','R-Sport','SVR'],
  'E-Pace':         ['S','SE','HSE'],
  'I-Pace':         ['EV400 S','EV400 SE','EV400 HSE'],
  'F-Type':         ['Coupe','Coupe R','Convertible','SVR'],

  // ── Lexus ─────────────────────────────────────────────────────────
  'ES':             ['300h Exquisite','300h Luxury','300h Ultra Luxury'],
  'LS':             ['500h Luxury','500h Ultra Luxury'],
  'NX':             ['300h Luxury','300h F Sport'],
  'RX':             ['450h Luxury','450h F Sport','450h+ F Sport'],
  'UX':             ['250h Exquisite','250h Luxury'],
  'LC':             ['500h Sports'],
  'LX':             ['500d Luxury'],
  'IS':             ['300h Luxury'],
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

        {/* Row 2: Registration No., City, Odometer */}
        <div className="vws-row-3">
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
              placeholder="City"
              searchPlaceholder="Search cities…"
            />
          </div>
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
        </div>

        {/* Row 3: Fuel Type, Transmission, Physical Condition */}
        <div className="vws-row-3">
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
            <FieldLabel>Physical Condition</FieldLabel>
            <select
              className="vws-input field-select"
              value={inputs.condition || ''}
              onChange={e => updateInput('condition', e.target.value)}
            >
              <option value="">Select Condition</option>
              {CONDITIONS.map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Row 4: Owners, Color, Seller Asking Price */}
        <div className="vws-row-3">
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
            <FieldLabel>Seller Asking Price</FieldLabel>
            <div className="vws-money-wrap" style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <span style={{ position: 'absolute', left: 12, fontSize: 13, color: 'var(--text-3)' }}>₹</span>
              <input
                className="vws-input"
                type="number"
                value={inputs.sellerAskingPrice === '0' ? '' : inputs.sellerAskingPrice}
                onChange={e => updateInput('sellerAskingPrice', e.target.value || '0')}
                placeholder="0"
                min={0}
                style={{ paddingLeft: 24 }}
              />
            </div>
          </div>
        </div>

        {/* Row 5: Target Margin %, Repair Budget, Certified Inspection */}
        <div className="vws-row-3">
          <div className="vws-field">
            <FieldLabel>Target Margin %</FieldLabel>
            <div className="vws-money-wrap" style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <input
                className="vws-input"
                type="number"
                min={8}
                max={30}
                step={1}
                value={inputs.targetMarginPct || 15}
                onChange={e => updateInput('targetMarginPct', e.target.value)}
                placeholder="15"
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
          <div className="vws-field" style={{ justifyContent: 'center', paddingTop: 18 }}>
            <label className="vws-label" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', userSelect: 'none' }}>
              <input
                type="checkbox"
                checked={!!inputs.inspected}
                onChange={e => updateInput('inspected', e.target.checked)}
                style={{ width: 18, height: 18, accentColor: 'var(--accent)', cursor: 'pointer' }}
              />
              <span style={{ fontSize: 13, fontWeight: 600 }}>Certified Inspection</span>
            </label>
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
