export function exportEvaluationsToCSV(evaluations = [], filename = 'vehicle_evaluations.csv') {
  const records = Array.isArray(evaluations) ? evaluations : [evaluations];
  if (records.length === 0) {
    alert('No evaluation data available to export.');
    return;
  }
  const headers = [
    'ID',
    'Date Created',
    'Source',
    'Brand',
    'Model',
    'Variant',
    'Year',
    'Fuel Type',
    'Transmission',
    'Odometer (km)',
    'City',
    'Locality',
    'Owners',
    'Condition',
    'Asking Price (INR)',
    'Predicted Market Value (INR)',
    'Recommended Buy Price (INR)',
    'Recommended Sell Price (INR)',
    'Expected Profit (INR)',
    'Margin (%)',
    'Risk Score',
    'Confidence Score',
    'Deal Quality Score',
    'Action / Recommendation',
    'Model Used',
  ];
  const escapeCSV = (val) => {
    if (val === null || val === undefined) return '""';
    const str = String(val).replace(/"/g, '""');
    return `"${str}"`;
  };
  const rows = records.map(rec => [
    rec.id || '',
    rec.createdAt || new Date().toISOString(),
    rec.source || 'Valuation',
    rec.brand || '',
    rec.model || '',
    rec.variant || '',
    rec.year || '',
    rec.fuel || '',
    rec.transmission || '',
    rec.odometer || rec.mileage || 0,
    rec.city || '',
    rec.locality || '',
    rec.ownerCount || 1,
    rec.condition || '',
    rec.sellerAskingPrice || 0,
    rec.marketValue || rec.predictedPrice || 0,
    rec.buyPrice || rec.recommendedBuyPrice || 0,
    rec.sellPrice || rec.recommendedSellPrice || 0,
    rec.expectedProfit || 0,
    rec.marginPct || 0,
    rec.riskScore || 0,
    rec.confidenceScore || 0,
    rec.dealQualityScore || 0,
    rec.action || '',
    rec.modelName || 'CatBoost',
  ].map(escapeCSV).join(','));
  const csvContent = [headers.map(escapeCSV).join(','), ...rows].join('\r\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
