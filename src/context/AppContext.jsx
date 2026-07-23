/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { BRANDS } from '../utils/mockData.js';
import { useAuth } from './AuthContext.jsx';
import { supabase } from '../lib/supabaseClient.js';

const DEFAULT_INPUTS = {
  brand: 'Honda', model: 'City', variant: '', year: '2021',
  fuel: 'Petrol', transmission: 'Manual', mileage: '28000', fuelEfficiency: '17.5',
  city: 'Bangalore', vin: '',
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

const HISTORY_KEY = 'PriceRef_ml_evaluation_history_v1'; // kept for demo-user fallback
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

/** Convert camelCase record → snake_case DB row for Supabase insert */
function recordToDbRow(rec, userId) {
  return {
    id:                   rec.id,
    user_id:              userId,
    created_at:           rec.createdAt,
    source:               rec.source,
    brand:                rec.brand,
    model:                rec.model,
    year:                 rec.year,
    fuel:                 rec.fuel,
    transmission:         rec.transmission,
    city:                 rec.city,
    odometer:             rec.odometer,
    fuel_efficiency:      rec.fuelEfficiency,
    owner_count:          rec.ownerCount,
    engine_cc:            rec.engineCc,
    condition:            rec.condition,
    seller_asking_price:  rec.sellerAskingPrice,
    market_value:         rec.marketValue,
    buy_price:            rec.buyPrice,
    sell_price:           rec.sellPrice,
    expected_profit:      rec.expectedProfit,
    margin_pct:           rec.marginPct,
    risk_score:           rec.riskScore,
    confidence_score:     rec.confidenceScore,
    deal_quality_score:   rec.dealQualityScore,
    action:               rec.action,
    urgency_score:        rec.urgencyScore,
    is_ml_powered:        rec.isMLPowered,
    positive_factors:     rec.positiveFactors,
    negative_factors:     rec.negativeFactors,
  };
}

/** Convert snake_case DB row → camelCase record for the UI */
function dbRowToRecord(row) {
  return {
    id:                   row.id,
    createdAt:            row.created_at,
    source:               row.source,
    brand:                row.brand,
    model:                row.model,
    year:                 row.year,
    fuel:                 row.fuel,
    transmission:         row.transmission,
    city:                 row.city,
    odometer:             row.odometer,
    kmDriven:             row.odometer,
    mileage:              row.odometer,
    fuelEfficiency:       row.fuel_efficiency,
    ownerCount:           row.owner_count,
    engineCc:             row.engine_cc,
    condition:            row.condition,
    conditionScore:       75,
    sellerAskingPrice:    row.seller_asking_price,
    marketValue:          row.market_value,
    predictedPrice:       row.market_value,
    buyPrice:             row.buy_price,
    recommendedBuyPrice:  row.buy_price,
    sellPrice:            row.sell_price,
    recommendedSellPrice: row.sell_price,
    expectedProfit:       row.expected_profit,
    marginPct:            row.margin_pct,
    riskScore:            row.risk_score,
    confidenceScore:      row.confidence_score,
    dealQualityScore:     row.deal_quality_score,
    dealQuality:          row.deal_quality_score,
    action:               row.action,
    urgencyScore:         row.urgency_score,
    isMLPowered:          row.is_ml_powered,
    positiveFactors:      row.positive_factors || [],
    negativeFactors:      row.negative_factors || [],
    modelName:            'CatBoostRegressor',
    valuationSource:      'CatBoost ML Backend',
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
  const [evaluations, setEvaluations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [dashFilters, setDashFilters] = useState({ brand: 'All', city: 'All', priceRange: 'All' });

  // Load evaluations from Supabase (or localStorage for demo user)
  useEffect(() => {
    if (!currentUser) { setEvaluations([]); return; }

    if (currentUser.id === 'demo') {
      setEvaluations(loadLocalHistory());
      return;
    }

    supabase
      .from('evaluations')
      .select('*')
      .eq('user_id', currentUser.id)
      .order('created_at', { ascending: false })
      .limit(500)
      .then(({ data, error }) => {
        if (error) { console.error('[PriceRef] Failed to load evaluations:', error.message); return; }
        // Map snake_case DB columns back to camelCase for the UI
        setEvaluations((data || []).map(dbRowToRecord));
      });
  }, [currentUser]);

  // Persist demo-user evaluations to localStorage
  useEffect(() => {
    if (currentUser?.id === 'demo') {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(evaluations.slice(0, 500)));
    }
  }, [evaluations, currentUser]);

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

  const addEvaluation = useCallback(async (vehicleInputs, result, source = 'Single Vehicle') => {
    const record = recordFromResult(vehicleInputs, result, source);
    setEvaluations(prev => [record, ...prev].slice(0, 500));

    // Persist to Supabase (skip for demo user)
    if (currentUser && currentUser.id !== 'demo') {
      const { error } = await supabase.from('evaluations').insert(recordToDbRow(record, currentUser.id));
      if (error) console.error('[PriceRef] Failed to save evaluation:', error.message);
    }

    return record;
  }, [currentUser]);

  const clearEvaluations = useCallback(async () => {
    setEvaluations([]);
    if (currentUser?.id === 'demo') {
      localStorage.removeItem(HISTORY_KEY);
      return;
    }
    if (currentUser) {
      const { error } = await supabase
        .from('evaluations')
        .delete()
        .eq('user_id', currentUser.id);
      if (error) console.error('[PriceRef] Failed to clear evaluations:', error.message);
    }
  }, [currentUser]);

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
      evaluations, addEvaluation, clearEvaluations,
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
