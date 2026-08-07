export const INDIAN_STATES = [
  'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
  'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
  'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
  'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
  'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
  'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
  'Andaman and Nicobar Islands', 'Chandigarh',
  'Dadra and Nagar Haveli and Daman and Diu', 'Delhi',
  'Jammu and Kashmir', 'Ladakh', 'Lakshadweep', 'Puducherry',
];

export const DEFAULT_VENDOR_TYPE = {
  engine: 'vendor',
  tyre: 'vendor',
  body: 'vendor',
  interior: 'vendor',
  electrical: 'vendor',
};

const ENGINE_COSTS = {
  good: { inhouse: 0, vendor: 0 },
  average: { inhouse: 4000, vendor: 8000 },
  poor: { inhouse: 18000, vendor: 35000 },
  critical: { inhouse: 45000, vendor: 80000 },
};

const TYRE_COSTS = {
  good: { inhouse: 0, vendor: 0 },
  two_bad: { inhouse: 4000, vendor: 6000 },
  all_bad: { inhouse: 8000, vendor: 12000 },
};

const BODY_COSTS = {
  clean: { inhouse: 0, vendor: 0 },
  minor: { inhouse: 3000, vendor: 5000 },
  major: { inhouse: 10000, vendor: 18000 },
  accident: { inhouse: 22000, vendor: 40000 },
};

const INTERIOR_COSTS = {
  clean: { inhouse: 0, vendor: 0 },
  needs_cleaning: { inhouse: 1500, vendor: 3000 },
  full_refurb: { inhouse: 6000, vendor: 10000 },
};

const ELECTRICAL_COSTS = {
  all_good: { inhouse: 0, vendor: 0 },
  ac_fault: { inhouse: 4500, vendor: 8000 },
  multi_fault: { inhouse: 8000, vendor: 15000 },
};

export const GRADE_OPTIONS = {
  engine: [
    { value: 'good', label: 'Good', inhouse: 0, vendor: 0 },
    { value: 'average', label: 'Average', inhouse: 4000, vendor: 8000 },
    { value: 'poor', label: 'Poor', inhouse: 18000, vendor: 35000 },
    { value: 'critical', label: 'Critical', inhouse: 45000, vendor: 80000 },
  ],
  tyre: [
    { value: 'good', label: 'All good', inhouse: 0, vendor: 0 },
    { value: 'two_bad', label: '2 need replacing', inhouse: 4000, vendor: 6000 },
    { value: 'all_bad', label: 'All need replacing', inhouse: 8000, vendor: 12000 },
  ],
  body: [
    { value: 'clean', label: 'Clean', inhouse: 0, vendor: 0 },
    { value: 'minor', label: 'Minor dents', inhouse: 3000, vendor: 5000 },
    { value: 'major', label: 'Major damage', inhouse: 10000, vendor: 18000 },
    { value: 'accident', label: 'Accident damage', inhouse: 22000, vendor: 40000 },
  ],
  interior: [
    { value: 'clean', label: 'Clean', inhouse: 0, vendor: 0 },
    { value: 'needs_cleaning', label: 'Needs cleaning', inhouse: 1500, vendor: 3000 },
    { value: 'full_refurb', label: 'Full refurbish', inhouse: 6000, vendor: 10000 },
  ],
  electrical: [
    { value: 'all_good', label: 'All working', inhouse: 0, vendor: 0 },
    { value: 'ac_fault', label: 'AC fault', inhouse: 4500, vendor: 8000 },
    { value: 'multi_fault', label: 'Multiple faults', inhouse: 8000, vendor: 15000 },
  ],
};

function categoryCost(table, grade, vendorKey) {
  const row = table[grade] || table[Object.keys(table)[0]];
  return row[vendorKey === 'inhouse' ? 'inhouse' : 'vendor'];
}

export function getReconCost(grades, vendorType = DEFAULT_VENDOR_TYPE, rcCost = 3500) {
  const engineCost = categoryCost(ENGINE_COSTS, grades.engine, vendorType.engine);
  const tyreCost = categoryCost(TYRE_COSTS, grades.tyre, vendorType.tyre);
  const bodyCost = categoryCost(BODY_COSTS, grades.body, vendorType.body);
  const interiorCost = categoryCost(INTERIOR_COSTS, grades.interior, vendorType.interior);
  const electricalCost = categoryCost(ELECTRICAL_COSTS, grades.electrical, vendorType.electrical);
  const rcTransferCost = Math.max(0, Number(rcCost) || 3500);
  const fixedCost = rcTransferCost + 2500 + 2000; 
  const total = engineCost + tyreCost + bodyCost + interiorCost + electricalCost + fixedCost;

  return {
    engine_cost: engineCost,
    tyre_cost: tyreCost,
    body_cost: bodyCost,
    interior_cost: interiorCost,
    electrical_cost: electricalCost,
    fixed_cost: fixedCost,
    rc_transfer_cost: rcTransferCost,
    total,
    breakdown: {
      engine: engineCost,
      tyres: tyreCost,
      body_paint: bodyCost,
      interior: interiorCost,
      electricals: electricalCost,
      fixed: fixedCost,
    },
  };
}

export function checkDisqualifier(vehicleAge, odometer, ownerCount, accidentHistory) {
  const accident = (accidentHistory || 'none').toLowerCase();
  if (vehicleAge > 12) {
    return { disqualified: true, reason: 'Vehicle age exceeds 12 years' };
  }
  if (odometer > 150000) {
    return { disqualified: true, reason: 'Odometer reading exceeds 150,000 km' };
  }
  if (ownerCount >= 4 && ['minor', 'major'].includes(accident)) {
    return { disqualified: true, reason: `Multiple owners (${ownerCount}) + accident history detected` };
  }
  return { disqualified: false, reason: 'Passes pre-screening' };
}

export function getGradePreview(category, grade) {
  const opt = GRADE_OPTIONS[category]?.find(o => o.value === grade);
  if (!opt) return { inhouse: 0, vendor: 0 };
  return { inhouse: opt.inhouse, vendor: opt.vendor };
}

export const SELLER_REASONS = [
  { value: 'upgrading', label: 'Upgrading' },
  { value: 'relocating', label: 'Relocating' },
  { value: 'financial', label: 'Financial need' },
  { value: 'unused', label: 'Car unused' },
  { value: 'problem', label: 'Problem car' },
];

export const DEAL_HEALTH_META = {
  green: {
    title: 'Strong Deal — Proceed with confidence',
    reason: 'Healthy margin, low recon cost, clean history',
  },
  yellow: {
    title: 'Moderate Deal — Review carefully',
    reason: 'Margin or recon/risk factors need closer inspection',
  },
  red: {
    title: 'Weak Deal — High risk or low margin',
    reason: 'Low margin or high recon/risk exposure',
  },
};
