/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { BRANDS } from '../utils/mockData.js';
import { useAuth } from './AuthContext.jsx';
import {
  fetchHistory,
  createHistoryEntry,
  clearHistory as clearHistoryApi,
  ApiError,
} from '../lib/apiClient.js';

const DEFAULT_INPUTS = {
  brand: 'Honda', model: 'City', variant: '', year: '2021',
  fuel: 'Petrol', transmission: 'Manual', mileage: '28000', fuelEfficiency: '17.5',
  city: 'Bangalore', locality: 'Indiranagar', vin: '',
  ownerCount: '1', engineCc: '1497', condition: 'Good',
  color: 'White', inspected: false,
  sellerAskingPrice: '0', targetMarginPct: '10', repairBuffer: '25000',
};

const DEFAULT_INSPECTION = {
  accidentHistory: 'none',
  loanOutstanding: false,
  registrationState: 'Maharashtra',
  sellerReason: 'upgrading',
  engineGrade: 'good',
  tyreGrade: 'good',
  bodyGrade: 'clean',
  interiorGrade: 'clean',
  electricalGrade: 'all_good',
  rcTransferCost: '3500',
  idvValue: '0',
  vendorType: {
    engine: 'vendor',
    tyre: 'vendor',
    body: 'vendor',
    interior: 'vendor',
    electrical: 'vendor',
  },
};

// Guest and demo sessions keep history here and never touch the database. That
// is the honest representation of "not signed in": the previous code wrote
// user_id: 'guest' into a uuid column, so every guest save failed and was
// swallowed by a console.warn.
const HISTORY_KEY = 'PriceRef_ml_evaluation_history_v1';
const AppContext = createContext(null);

function toNumber(value, fallback = 0) {
  if (value === null || value === undefined || value === '') return fallback;
  const match = String(value).replace(/,/g, '').match(/-?\d+(\.\d+)?/);
  const n = match ? Number(match[0]) : Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function loadLocalHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function recordFromResult(inputs, result, source = 'Single Vehicle') {
  const marketValue = toNumber(result?.predictedPrice ?? result?.marketValue, 0);
  const buyPrice = toNumber(result?.recommendedBuyPrice ?? result?.dealerAcqPrice ?? result?.buyPrice, 0);
  const sellPrice = toNumber(result?.recommendedSellPrice ?? result?.suggestedSellPrice ?? result?.sellPrice, 0);
  const expectedProfit = toNumber(result?.expectedProfit ?? result?.marginAmt, 0);
  const odometer = toNumber(inputs?.mileage ?? inputs?.odometer_reading ?? result?.odometer, 0);
  const riskScore = toNumber(result?.riskScore, 0);
  const dealQualityScore = toNumber(result?.dealQualityScore ?? result?.dealQuality, 0);

  return {
    id: result?.id || `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
    createdAt: new Date().toISOString(),
    source,
    brand: inputs?.brand || result?.brand || 'Unknown',
    model: inputs?.model || result?.model || 'Unknown',
    year: toNumber(inputs?.year ?? result?.year, 0),
    fuel: inputs?.fuel || inputs?.fuel_type || result?.fuel || 'Unknown',
    transmission: inputs?.transmission || result?.transmission || 'Unknown',
    city: inputs?.city || result?.city || 'Unknown',
    locality: inputs?.locality || result?.locality || 'Indiranagar',
    odometer,
    kmDriven: odometer,
    mileage: odometer,
    fuelEfficiency: toNumber(inputs?.fuelEfficiency ?? inputs?.fuel_efficiency ?? result?.fuelEfficiency, 0),
    ownerCount: toNumber(inputs?.ownerCount ?? inputs?.owner_count, 1),
    engineCc: toNumber(inputs?.engineCc ?? inputs?.engine_cc, 0),
    condition: inputs?.condition || result?.condition || 'Good',
    conditionScore: toNumber(result?.conditionScore ?? result?.condition_score, 75),
    sellerAskingPrice: toNumber(inputs?.sellerAskingPrice ?? inputs?.seller_asking_price, 0),
    marketValue,
    predictedPrice: marketValue,
    buyPrice,
    recommendedBuyPrice: buyPrice,
    sellPrice,
    recommendedSellPrice: sellPrice,
    expectedProfit,
    marginPct: toNumber(result?.expectedMarginPct ?? result?.marginPct, 0),
    riskScore,
    confidenceScore: toNumber(result?.confidenceScore, 0),
    dealQualityScore,
    dealQuality: dealQualityScore,
    action: result?.action || 'MANUAL REVIEW',
    urgencyScore: toNumber(result?.urgencyScore, 0),
    modelName: result?.modelName || 'CatBoostRegressor',
    isMLPowered: result?.isMLPowered !== false,
    valuationSource: result?.valuationSource || 'CatBoost ML Backend',
    positiveFactors: result?.positiveFactors || [],
    negativeFactors: result?.negativeFactors || [],
  };
}

/**
 * Fields the API owns and must not be sent to it.
 *
 * `id` and `createdAt` are assigned server-side from the verified token and the
 * server clock. Sending them would let a client collide with an existing id or
 * backdate a record, so POST /api/history rejects them by ignoring them — this
 * keeps the request payload honest about that.
 */
const SERVER_OWNED_FIELDS = ['id', 'createdAt'];

/**
 * UI-only aliases derived from canonical fields.
 *
 * Several screens read `mileage`/`kmDriven` for the odometer and
 * `predictedPrice`/`recommendedBuyPrice` for the valuation. Those are display
 * conveniences, not distinct data, so they are computed on read rather than
 * stored — persisting them would mean four columns that can disagree.
 */
const UI_ONLY_FIELDS = [
  'kmDriven', 'mileage', 'predictedPrice', 'recommendedBuyPrice',
  'recommendedSellPrice', 'dealQuality', 'conditionScore', 'modelName',
  'valuationSource',
];

/** Strip client-side and server-owned fields to get the POST /api/history body. */
function recordToApiPayload(rec) {
  const payload = { ...rec };
  for (const key of [...SERVER_OWNED_FIELDS, ...UI_ONLY_FIELDS]) delete payload[key];
  return payload;
}

/**
 * Re-attach the UI aliases to a record returned by the API.
 *
 * The backend emits camelCase already (see backend/routers/history.py), so this
 * is no longer a naming translation — only alias expansion. The snake_case
 * mapping that used to live here now lives server-side, in one place, next to
 * the schema it has to agree with.
 */
function hydrateRecord(rec) {
  const odometer = toNumber(rec.odometer, 0);
  const marketValue = toNumber(rec.marketValue, 0);
  const buyPrice = toNumber(rec.buyPrice, 0);
  const sellPrice = toNumber(rec.sellPrice, 0);

  return {
    ...rec,
    variant: rec.variant || '',
    locality: rec.locality || '',
    positiveFactors: rec.positiveFactors || [],
    negativeFactors: rec.negativeFactors || [],

    kmDriven: odometer,
    mileage: odometer,
    predictedPrice: marketValue,
    recommendedBuyPrice: buyPrice,
    recommendedSellPrice: sellPrice,
    dealQuality: toNumber(rec.dealQualityScore, 0),
    conditionScore: toNumber(rec.conditionScore, 75),
    modelName: rec.modelName || 'CatBoostRegressor',
    valuationSource: rec.valuationSource || 'CatBoost ML Backend',
  };
}

export function AppProvider({ children }) {

  const { currentUser } = useAuth();
  const [activeScreen, setActiveScreen] = useState('home');
  const [role] = useState(currentUser?.role || 'Dealer');
  const [inputs, setInputs] = useState(DEFAULT_INPUTS);
  const [conditionScore, setConditionScore] = useState(78);
  const [valuationResult, setValuationResult] = useState(null);
  const [enhancedResult, setEnhancedResult] = useState(null);
  const [enhancedInspection, setEnhancedInspection] = useState(DEFAULT_INSPECTION);
  const [reverseResult, setReverseResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [dashFilters, setDashFilters] = useState({ brand: 'All', city: 'All', priceRange: 'All' });
  // Non-fatal persistence problems, so the UI can say "saved locally only"
  // rather than implying a valuation was stored in the cloud when it was not.
  const [historyError, setHistoryError] = useState(null);

  // History is held in two buckets rather than one, and the list the UI reads is
  // derived during render.
  //
  // The obvious alternative — a single `evaluations` state that an effect
  // overwrites whenever the user changes — means synchronously calling setState
  // from an effect body, which triggers a second render pass before paint and is
  // what react-hooks/set-state-in-effect warns about. Deriving instead makes the
  // guest case need no effect at all.
  const [localHistory, setLocalHistory] = useState(loadLocalHistory);
  // Tagged with the user it was fetched for, as { userId, rows }.
  //
  // The tag is what makes a sign-out cleanup effect unnecessary: rather than
  // clearing this when the user changes (a synchronous setState in an effect),
  // the derived value below simply ignores rows belonging to anyone else. One
  // user's history can therefore never be read into another's session.
  const [cloudHistory, setCloudHistory] = useState(null);

  // True when history should round-trip through the API. Demo sessions have no
  // auth.users row and therefore no user id to own database records.
  const isCloudUser = Boolean(currentUser && currentUser.id !== 'demo');

  // null here means "not fetched for this user yet", which is distinct from []
  // meaning "fetched, and this user genuinely has no history".
  const cloudRows =
    cloudHistory && currentUser && cloudHistory.userId === currentUser.id
      ? cloudHistory.rows
      : null;

  // Guests read the local store. Signed-in users read the cloud copy, falling
  // back to the local mirror until the first fetch resolves — so the dashboard
  // is populated immediately instead of flashing empty.
  const evaluations = isCloudUser ? (cloudRows ?? localHistory) : localHistory;

  // Fetch cloud history for signed-in users. setState happens only inside the
  // promise callbacks, never synchronously in the effect body.
  useEffect(() => {
    if (!isCloudUser) return;

    const userId = currentUser.id;
    // Abort on user change so a slow response for the previous user cannot land
    // in the new user's dashboard.
    const controller = new AbortController();

    fetchHistory({ limit: 500, signal: controller.signal })
      .then(rows => {
        setCloudHistory({ userId, rows: rows.map(hydrateRecord) });
        setHistoryError(null);
      })
      .catch(error => {
        if (error?.name === 'AbortError') return;
        // Leave cloudHistory unset so the local mirror keeps showing, but say so:
        // the old code silently substituted local data for cloud data, which
        // looked identical to "your cloud history is intact".
        setHistoryError(
          error instanceof ApiError && error.isUnavailable
            ? 'Cloud history is unavailable — showing this browser\'s local copy.'
            : `Could not load cloud history (${error.message}) — showing local copy.`
        );
      });

    return () => controller.abort();
  }, [isCloudUser, currentUser?.id]);

  // Local mirror of the history.
  //
  // For guests this is the system of record. For signed-in users it is a cache
  // that keeps the dashboard populated when the API is unreachable. Capped at
  // 500 to stay well inside the ~5 MB localStorage quota.
  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(evaluations.slice(0, 500)));
    } catch {
      // Quota exceeded or private browsing — the cache is optional, so drop it.
    }
  }, [evaluations]);

  const updateInput = useCallback((field, value) => {
    setInputs(prev => {
      const next = { ...prev, [field]: value };
      if (field === 'brand') { next.model = BRANDS[value]?.[0] || ''; next.variant = ''; }
      if (field === 'model') next.variant = '';
      return next;
    });
  }, []);

  const fillFromVIN = useCallback((vinData) => {
    setInputs(prev => ({
      ...prev,
      brand: vinData.brand,
      model: vinData.model,
      year: String(vinData.year),
      fuel: vinData.fuel,
      transmission: vinData.transmission,
      mileage: String(vinData.mileage),
      fuelEfficiency: vinData.fuelEfficiency ? String(vinData.fuelEfficiency) : prev.fuelEfficiency,
      city: 'Bangalore',
      ownerCount: vinData.ownerCount ? String(vinData.ownerCount) : prev.ownerCount,
      engineCc: vinData.engineCc ? String(vinData.engineCc) : prev.engineCc,
      sellerAskingPrice: vinData.sellerAskingPrice ? String(vinData.sellerAskingPrice) : prev.sellerAskingPrice,
    }));
  }, []);

  const prepend = useCallback((record) => {
    setLocalHistory(prev => [record, ...prev].slice(0, 500));
    setCloudHistory(prev =>
      prev === null ? prev : { ...prev, rows: [record, ...prev.rows].slice(0, 500) }
    );
  }, []);

  const addEvaluation = useCallback(async (vehicleInputs, result, source = 'Single Vehicle') => {
    const record = hydrateRecord(recordFromResult(vehicleInputs, result, source));

    // Show it immediately; reconcile with the server's version below. A dealer
    // should never wait on a network round-trip to see the valuation they just ran.
    prepend(record);

    if (!isCloudUser) return record;

    try {
      const saved = await createHistoryEntry(recordToApiPayload(record));
      const persisted = hydrateRecord(saved);
      // Swap the optimistic row for the persisted one so the record carries the
      // server-assigned id — without this, deleting a single entry later would
      // reference an id the database never issued.
      const swapIn = rows => rows.map(item => (item.id === record.id ? persisted : item));
      setLocalHistory(swapIn);
      setCloudHistory(prev => (prev === null ? null : { ...prev, rows: swapIn(prev.rows) }));
      setHistoryError(null);
      return persisted;
    } catch (error) {
      // Keep the optimistic row: it is still in the local mirror, so the
      // valuation is not lost. Report it rather than console.warn-ing, because
      // "saved" and "saved to this browser only" are materially different.
      setHistoryError(
        error instanceof ApiError && error.isUnavailable
          ? 'Cloud history is unavailable — this valuation was saved locally only.'
          : `Could not save to cloud history (${error.message}) — saved locally only.`
      );
      return record;
    }
  }, [isCloudUser, prepend]);

  const clearEvaluations = useCallback(async () => {
    const previousLocal = localHistory;
    const previousCloud = cloudHistory;

    setLocalHistory([]);
    // Left null when it was already null: nothing has been fetched yet, and the
    // derived list falls through to localHistory — which is now empty — so the
    // UI shows a cleared history either way. Avoiding the currentUser reference
    // here also keeps this callback's dependencies to the values it actually reads.
    setCloudHistory(prev => (prev === null ? null : { ...prev, rows: [] }));
    try {
      localStorage.removeItem(HISTORY_KEY);
    } catch { /* nothing to clean up */ }

    if (!isCloudUser) return;

    try {
      await clearHistoryApi();
      setHistoryError(null);
    } catch (error) {
      // Restore rather than leave the UI claiming the history is gone when the
      // server still holds every row.
      setLocalHistory(previousLocal);
      setCloudHistory(previousCloud);
      setHistoryError(`Could not clear cloud history (${error.message}) — nothing was deleted.`);
    }
  }, [isCloudUser, localHistory, cloudHistory]);

  const updateEnhancedInspection = useCallback((field, value) => {
    setEnhancedInspection(prev => ({ ...prev, [field]: value }));
  }, []);

  const updateVendorType = useCallback((category, value) => {
    setEnhancedInspection(prev => ({
      ...prev,
      vendorType: { ...prev.vendorType, [category]: value },
    }));
  }, []);

  return (
    <AppContext.Provider value={{
      activeScreen, setActiveScreen,
      role,
      inputs, updateInput, fillFromVIN,
      conditionScore, setConditionScore,
      valuationResult, setValuationResult,
      enhancedResult, setEnhancedResult,
      enhancedInspection, setEnhancedInspection, updateEnhancedInspection, updateVendorType,
      reverseResult, setReverseResult,
      evaluations, addEvaluation, clearEvaluations, historyError,
      isLoading, setIsLoading,
      dashFilters, setDashFilters,
    }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used inside AppProvider');
  return ctx;
}
