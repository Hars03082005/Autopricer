
function getApiBase() {
  if (typeof window !== 'undefined') {
    if (window.PriceRef_API_URL) return window.PriceRef_API_URL;

    const host = window.location.hostname;
    const isLocalHost = host === 'localhost' || host === '127.0.0.1';
    const envUrl = (import.meta.env.VITE_API_URL || import.meta.env.VITE_ML_API_URL || '').trim();

    if (envUrl) {
      const envIsLocal = envUrl.includes('localhost') || envUrl.includes('127.0.0.1');
      if (isLocalHost || !envIsLocal) {
        return envUrl.replace(/\/+$/, '');
      }
    }

    if (host.endsWith('.onrender.com')) {
      const name = host.replace('.onrender.com', '');
      if (name === 'priceref' || name.startsWith('priceref')) {
        return 'https://priceref-backend.onrender.com';
      }
      return 'https://price-prediction-backend.onrender.com';
    }
  }

  return (
    import.meta.env.VITE_API_URL ||
    import.meta.env.VITE_ML_API_URL ||
    'http://localhost:8008'
  ).replace(/\/+$/, '');
}

function toNumber(value, fallback = 0) {
  if (value === null || value === undefined || value === '') return fallback;
  if (typeof value === 'number') return Number.isFinite(value) ? value : fallback;
  const match = String(value).replace(/,/g, '').match(/-?\d+(\.\d+)?/);
  const n = match ? Number(match[0]) : Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function parseOwnerCount(value, fallback = 1) {
  const text = String(value ?? '').trim().toLowerCase();
  if (!text) return fallback;
  if (/\b1\b|1st|first/.test(text)) return 1;
  if (/\b2\b|2nd|second/.test(text)) return 2;
  if (/\b3\b|3rd|third/.test(text)) return 3;
  if (/\b4\b|4th|fourth/.test(text)) return 4;
  if (/more|fifth|5/.test(text)) return 5;
  return toNumber(value, fallback);
}

function titleCase(value) {
  return String(value || '')
    .trim()
    .split(/\s+/)
    .map(w => w ? w[0].toUpperCase() + w.slice(1).toLowerCase() : '')
    .join(' ');
}

function normalizeFuel(value) {
  const text = String(value || '').toLowerCase();
  if (text.includes('diesel')) return 'Diesel';
  if (text.includes('electric') || text.includes('ev')) return 'Electric';
  if (text.includes('cng')) return 'CNG';
  if (text.includes('hybrid')) return 'Hybrid';
  return 'Petrol';
}

function normalizeTransmission(value) {
  const text = String(value || '').toLowerCase();
  if (text.includes('auto') || text.includes('cvt') || text.includes('dct')) return 'Automatic';
  return 'Manual';
}

function normalizeCondition(value) {
  const text = String(value || '').toLowerCase();
  if (text.includes('excellent')) return 'Excellent';
  if (text.includes('poor')) return 'Poor';
  if (text.includes('average') || text.includes('fair')) return 'Average';
  return 'Good';
}

function normalizeColor(value) {
  const text = String(value || '').trim().toLowerCase();
  
  const colorMap = {
    'white': 'white', 'pearl white': 'white', 'solid white': 'white',
    'silver': 'silver', 'silver metallic': 'silver', 'grey': 'grey', 'gray': 'grey',
    'black': 'black', 'jet black': 'black', 'phantom black': 'black',
    'blue': 'blue', 'navy blue': 'blue', 'cobalt blue': 'blue', 'azure blue': 'blue',
    'red': 'red', 'magma red': 'red', 'torch red': 'red', 'fiery red': 'red',
    'brown': 'brown', 'copper brown': 'brown', 'chestnut brown': 'brown',
    'beige': 'beige', 'ivory': 'beige', 'champagne': 'beige',
    'gold': 'gold', 'golden': 'gold', 'bronze': 'gold',
    'green': 'green', 'apple green': 'green', 'mint green': 'green',
    'orange': 'orange', 'saffron': 'orange', 'tangerine': 'orange',
    'yellow': 'yellow', 'lemon': 'yellow', 'mustard': 'yellow',
    'maroon': 'maroon', 'wine': 'maroon', 'burgundy': 'maroon',
    'purple': 'purple', 'violet': 'purple',
  };
  return colorMap[text] || text || 'unknown';
}

export function payloadFromInputs(inputs) {
  return {
    brand: titleCase(inputs.brand || 'Unknown'),
    model: titleCase(inputs.model || 'Unknown'),
    variant: inputs.variant ? String(inputs.variant).trim().toLowerCase() : 'unknown',
    year: Math.trunc(toNumber(inputs.year, 2021)),
    fuel_type: normalizeFuel(inputs.fuel || inputs.fuel_type),
    transmission: normalizeTransmission(inputs.transmission),
    odometer_reading: Math.trunc(toNumber(inputs.mileage ?? inputs.odometer_reading, 0)),
    owner_count: parseOwnerCount(inputs.ownerCount ?? inputs.owner_count, 1),
    city: titleCase(inputs.city || 'Unknown'),
    locality: inputs.locality ? String(inputs.locality).trim() : 'Indiranagar',
    color: normalizeColor(inputs.color || ''),
    inspected: Boolean(inputs.inspected),
    condition: normalizeCondition(inputs.condition),
    seller_asking_price: 0,
    target_margin_pct: toNumber(inputs.targetMarginPct ?? inputs.target_margin_pct, 10),
    repair_buffer: toNumber(inputs.repairBuffer ?? inputs.repair_buffer, 0),
    ...(inputs.modelVariant && inputs.modelVariant !== 'auto' ? { model_variant: inputs.modelVariant } : {}),
  };
}

function buildCounterfactuals(inputs) {
  const km = toNumber(inputs.mileage ?? inputs.odometer_reading, 0);
  const age = new Date().getFullYear() - toNumber(inputs.year, new Date().getFullYear());
  const condition = normalizeCondition(inputs.condition);
  const items = [];
  if (km > 80000) {
    items.push({ positive: true, text: 'Lower odometer band would improve quote quality', detail: 'High km reduces resale confidence and increases risk buffer.', impact: '+value' });
  } else {
    items.push({ positive: true, text: 'Current odometer reading supports stronger resale confidence', detail: 'Lower usage helps maintain dealer margin.', impact: '+confidence' });
  }
  if (age > 6) {
    items.push({ positive: false, text: 'Older vehicle age is reducing the market value', detail: 'More age generally increases depreciation and holding risk.', impact: '-price' });
  } else {
    items.push({ positive: true, text: 'Newer vehicle age supports better acquisition quality', detail: 'Lower depreciation improves buy/sell spread.', impact: '+deal' });
  }
  if (condition === 'Poor' || condition === 'Average') {
    items.push({ positive: true, text: 'Improving condition before resale can increase margin', detail: `Current condition is ${condition}; repair/reconditioning can improve buyer confidence.`, impact: '+margin' });
  } else {
    items.push({ positive: true, text: 'Good condition is supporting the recommendation', detail: 'Condition keeps repair buffer and acquisition risk lower.', impact: '+risk' });
  }
  return items.slice(0, 4);
}

function normalizeApiResult(data, inputs) {
  const predictedPrice = data.market_value ?? 0;
  const baseMarketValue = data.base_market_value ?? predictedPrice;
  const recommendedBuyPrice = data.recommended_buy_price ?? data.dealer_acq_price ?? 0;
  const recommendedSellPrice = data.recommended_sell_price ?? data.suggested_sell_price ?? 0;
  const expectedProfit = data.expected_profit ?? data.margin_amt ?? 0;
  const expectedMarginPct = data.expected_margin_pct ?? data.margin_pct ?? 0;
  const priceMin = data.price_min ?? Math.round(predictedPrice * 0.9372);
  const priceMax = data.price_max ?? Math.round(predictedPrice * 1.0628);
  const priceMedian = data.price_median ?? predictedPrice;

  return {
    predictedPrice,
    baseMarketValue,
    conditionMultiplier: data.condition_multiplier ?? 1,
    conditionAdjustment: data.condition_adjustment ?? 0,
    conditionScore: data.condition_score ?? 75,
    priceMin,
    priceMax,
    priceMedian,
    marketRangeCompCount: data.market_range_comp_count ?? 0,
    marketRangeStage: data.market_range_stage ?? 0,
    marketRangeStageLabel: data.market_range_stage_label ?? '',
    marketRangeSource: data.market_range_source ?? 'mape_fallback',
    ci: data.ci ?? (predictedPrice - priceMin),
    dealerAcqPrice: data.dealer_acq_price ?? recommendedBuyPrice,
    suggestedSellPrice: data.suggested_sell_price ?? recommendedSellPrice,
    marginPct: data.margin_pct ?? expectedMarginPct,
    marginAmt: data.margin_amt ?? expectedProfit,
    recommendedBuyPrice,
    recommendedSellPrice,
    expectedProfit,
    expectedMarginPct,
    openingOffer: data.opening_offer ?? Math.round(recommendedBuyPrice * 0.97),
    maxOffer: data.max_offer ?? Math.round(recommendedBuyPrice * 1.03),
    targetOffer: data.target_offer ?? recommendedBuyPrice,
    sellerGap: data.seller_gap ?? 0,
    targetMarginPct: data.target_margin_pct ?? toNumber(inputs.targetMarginPct, 10),
    repairBuffer: data.repair_buffer ?? toNumber(inputs.repairBuffer, 25000),
    recon_cost: data.recon_cost ?? 18000,
    holding_cost: data.holding_cost ?? 5000,
    doc_cost: data.doc_cost ?? 4500,
    risk_buffer: data.risk_buffer ?? 3000,
    target_profit: data.target_profit ?? 35000,
    waterfall: data.waterfall || [],
    similarCars: data.similar_cars || [],
    action: data.action ?? 'MANUAL REVIEW',
    riskScore: data.risk_score ?? 0,
    riskLevel: data.risk_level ?? 'Medium',
    confidenceScore: data.confidence_score ?? 0,
    demandScore: data.demand_score ?? 0,
    brandRetentionScore: data.brand_retention_score ?? 0,
    vehicleHealthScore: data.vehicle_health_score ?? 0,
    resaleLiquidityScore: data.resale_liquidity_score ?? 0,
    dealQualityScore: data.deal_quality_score ?? 0,
    urgencyScore: data.urgency_score ?? 0,
    urgencyLabel: data.urgency_label ?? 'Medium',
    positiveFactors: data.positive_factors ?? [],
    negativeFactors: data.negative_factors ?? [],
    quoteMessage: data.quote_message ?? '',
    warnings: data.warnings ?? [],
    shap: data.shap ?? [],
    counterfactuals: buildCounterfactuals(inputs),
    damageBoxes: [],
    models: [
      { name: 'CatBoost Base Market Value', price: baseMarketValue, weight: 65 },
      { name: 'Condition Calibration Layer', price: predictedPrice, weight: 15 },
      { name: 'Quote & Risk Decision Engine', price: recommendedBuyPrice, weight: 20 },
    ],
    modelName: data.model_name || 'CatBoostRegressor',
    isMLPowered: data.is_ml_powered ?? true,
    modelMetrics: data.metrics || {},
    trainMetrics: data.train_metrics || {},
    validationMetrics: data.validation_metrics || {},
    testMetrics: data.test_metrics || {},
    overfittingCheck: data.overfitting_check || {},
    valuationSource: 'CatBoost ML Backend',
    segmentClass: data.segment_class ?? data.brand_class ?? 'economy',
    segmentModelUsed: data.segment_model_used ?? data.class_model_used ?? false,
    routingNote: data.routing_note ?? '',
    
    valuationConfidence:     data.confidence ?? 'Low',
    valuationConfidenceScore: data.confidence_score ?? 0,
    marketSupport:           data.market_support ?? 'Weak',
    comparablesUsed:         data.comparables_used ?? 0,
    averageSimilarity:       data.average_similarity ?? 0,
    ensembleVariance:        data.ensemble_variance ?? 0,
    expectedModelError:      data.expected_model_error ?? 0,
    confidenceCase:          data.confidence_case ?? 'low',
  };
}

async function postJson(path, payload) {
  const response = await fetch(`${getApiBase()}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const message = await response.text().catch(() => '');
    throw new Error(`ML API error ${response.status}${message ? `: ${message}` : ''}`);
  }
  return response.json();
}

export async function fetchBrands() {
  const response = await fetch(`${getApiBase()}/api/brands`);
  if (!response.ok) {
    const message = await response.text().catch(() => '');
    throw new Error(`Brands API error ${response.status}${message ? `: ${message}` : ''}`);
  }
  const data = await response.json();
  return data.brands || {};
}

export async function fetchCatalog(variantId) {
  const url = variantId
    ? `${getApiBase()}/api/catalog?model_variant=${encodeURIComponent(variantId)}`
    : `${getApiBase()}/api/catalog`;
  const response = await fetch(url);
  if (!response.ok) {
    // If variant-specific catalog fails, silently fall back to default
    if (variantId) {
      const fallback = await fetch(`${getApiBase()}/api/catalog`).catch(() => null);
      if (fallback && fallback.ok) {
        const data = await fallback.json();
        return data.catalog || {};
      }
    }
    return {};
  }
  const data = await response.json();
  return data.catalog || {};
}

export async function fetchBrandModels(brand) {
  const response = await fetch(`${getApiBase()}/api/catalog/${encodeURIComponent(brand)}`);
  if (!response.ok) {
    return { brand, models: {} };
  }
  const data = await response.json();
  return data || { brand, models: {} };
}

export async function fetchOptions({ brand, model, variant } = {}) {
  const params = new URLSearchParams();
  if (brand)   params.set('brand',   brand);
  if (model)   params.set('model',   model);
  if (variant) params.set('variant', variant);

  try {
    const response = await fetch(`${getApiBase()}/api/options?${params.toString()}`);
    if (!response.ok) throw new Error('options api failed');
    return await response.json();
  } catch {
    return {
      fuel_types:    ['Petrol', 'Diesel', 'CNG', 'Electric', 'Hybrid'],
      transmissions: ['Manual', 'Automatic', 'AMT', 'CVT', 'DCT', 'IMT'],
      years:         Array.from({ length: 20 }, (_, i) => String(new Date().getFullYear() - i)),
    };
  }
}

export async function runMLValuation(inputs) {
  const data = await postJson('/evaluate', payloadFromInputs(inputs));
  return normalizeApiResult(data, inputs);
}

export async function runMLValuationWithVariant(inputs, variantId) {
  const payload = {
    ...payloadFromInputs(inputs),
    model_variant: variantId,
  };
  const data = await postJson('/evaluate', payload);
  return normalizeApiResult(data, inputs);
}

export async function runS5Valuation(inputs) {
  const payload = payloadFromInputs(inputs);
  const data = await postJson('/evaluate', payload);
  return normalizeApiResult(data, inputs);
}

function normalizeEnhancedResult(data, inputs) {
  const base = normalizeApiResult(data, inputs);
  return {
    ...base,
    disqualifier: data.disqualifier,
    seasonalMultiplier: data.seasonal_multiplier,
    seasonalMonth: data.seasonal_month,
    recon: data.recon,
    wheelrRisk: data.wheelr_risk,
    negotiation: data.negotiation,
    dealHealth: data.deal_health,
    enhancedMaxBuyPrice: data.enhanced_max_buy_price,
    idvAnalysis: data.idv_analysis || null,
  };
}

export function enhancedPayloadFromInputs(inputs, inspection) {
  return {
    ...payloadFromInputs(inputs),
    accident_history: inspection.accidentHistory || 'none',
    registration_state: inspection.registrationState || '',
    sale_state: inspection.saleState || inputs.city || '',
    loan_outstanding: Boolean(inspection.loanOutstanding),
    seller_reason: inspection.sellerReason || 'upgrading',
    engine_grade: inspection.engineGrade || 'good',
    tyre_grade: inspection.tyreGrade || 'good',
    body_grade: inspection.bodyGrade || 'clean',
    interior_grade: inspection.interiorGrade || 'clean',
    electrical_grade: inspection.electricalGrade || 'all_good',
    vendor_type: inspection.vendorType || {
      engine: 'vendor',
      tyre: 'vendor',
      body: 'vendor',
      interior: 'vendor',
      electrical: 'vendor',
    },
    rc_transfer_cost: Math.max(0, Number(inspection.rcTransferCost) || 3500),
    idv_value: Math.max(0, Number(inspection.idvValue) || 0),
  };
}

export async function runEnhancedEvaluation(inputs, inspection) {
  const data = await postJson('/evaluate-enhanced', enhancedPayloadFromInputs(inputs, inspection));
  return normalizeEnhancedResult(data, inputs);
}

export async function runReverseCalculate(payload) {
  const data = await postJson('/reverse-calculate', payload);
  return {
    expectedSellPrice: data.expected_sell_price,
    recon: data.recon,
    profitTarget: data.profit_target,
    wheelrRisk: data.wheelr_risk,
    maxBuyPrice: data.max_buy_price,
    negotiation: data.negotiation,
    dealHealth: data.deal_health,
    disqualifier: data.disqualifier,
    seasonalMultiplier: data.seasonal_multiplier,
    priceBreakdown: data.price_breakdown,
  };
}

export async function fetchRegistry() {
  const response = await fetch(`${getApiBase()}/api/registry`);
  if (!response.ok) {
    return { default: null, variants: [] };
  }
  return response.json();
}

export async function activateVariant(variantId) {
  const response = await fetch(`${getApiBase()}/api/registry/${encodeURIComponent(variantId)}/activate`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw new Error(`Failed to activate variant ${variantId}`);
  }
  return response.json();
}

