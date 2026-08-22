/**
 * Exports vehicle valuation evaluations to CSV for data analysis and dealer auditing.
 * Contains no API secrets or model file paths.
 */
export function exportEvaluationsToCSV(evaluations = [], filename = 'priceref_evaluations.csv') {
  const records = Array.isArray(evaluations) ? evaluations : [evaluations];
  if (records.length === 0) {
    alert('No valuation data available to export.');
    return;
  }

  const headers = [
    'evaluation_id',
    'timestamp',
    'brand',
    'model',
    'variant',
    'year',
    'odometer',
    'owner_count',
    'fuel_type',
    'transmission',
    'color',
    'locality',
    'condition',
    'seller_asking_price',
    'predicted_market_value',
    'market_low',
    'market_high',
    'recommended_buy_low',
    'recommended_buy_high',
    'target_retail_price',
    'expected_profit',
    'ROI_pct',
    'deal_score',
    'decision',
  ];

  const escapeCSV = (val) => {
    if (val === null || val === undefined) return '""';
    const str = String(val).replace(/"/g, '""');
    return `"${str}"`;
  };

  const rows = records.map(rec => {
    const marketVal = Number(rec.marketValue ?? rec.predictedPrice ?? 0);
    const buyPrice = Number(rec.buyPrice ?? rec.recommendedBuyPrice ?? 0);
    const sellPrice = Number(rec.sellPrice ?? rec.recommendedSellPrice ?? 0) || Math.round(marketVal * 1.05);
    const profit = Number(rec.expectedProfit ?? (sellPrice - buyPrice - 30500));
    const roi = buyPrice > 0 ? ((profit / buyPrice) * 100).toFixed(1) : '0.0';

    return [
      rec.id || '',
      rec.createdAt || new Date().toISOString(),
      rec.brand || '',
      rec.model || '',
      rec.variant || '',
      rec.year || '',
      rec.odometer || rec.mileage || 0,
      rec.ownerCount || 1,
      rec.fuel || rec.fuel_type || '',
      rec.transmission || '',
      rec.color || 'White',
      rec.locality || 'Bangalore',
      rec.condition || 'Good',
      rec.sellerAskingPrice || 0,
      marketVal,
      rec.priceMin || Math.round(marketVal * 0.94),
      rec.priceMax || Math.round(marketVal * 1.06),
      rec.opening_offer || Math.round((buyPrice * 0.95) / 500) * 500,
      rec.max_offer || Math.round((buyPrice * 1.03) / 500) * 500,
      sellPrice,
      profit,
      roi,
      rec.dealQualityScore ?? rec.dealQuality ?? 75,
      rec.action || 'BUY',
    ].map(escapeCSV).join(',');
  });

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
