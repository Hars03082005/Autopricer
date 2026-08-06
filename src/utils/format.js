export function formatINR(amount) {
  if (!amount && amount !== 0) return '—';
  return '₹' + Number(amount).toLocaleString('en-IN');
}
export function getSeasonalContext(month) {
  const map = {
    1: { label: 'January', note: 'Post-festive slowdown', mult: 0.97 },
    2: { label: 'February', note: 'Quiet market', mult: 0.97 },
    3: { label: 'March', note: 'FY-end bonus spending', mult: 1.04 },
    4: { label: 'April', note: 'Summer slowdown', mult: 0.98 },
    5: { label: 'May', note: 'Summer slowdown', mult: 0.98 },
    6: { label: 'June', note: 'Pre-monsoon SUV spike', mult: 1.06 },
    7: { label: 'July', note: 'Monsoon demand', mult: 1.06 },
    8: { label: 'August', note: 'Monsoon dampening', mult: 0.99 },
    9: { label: 'September', note: 'Pre-festive buildup', mult: 0.99 },
    10: { label: 'October', note: 'Diwali festive season', mult: 1.05 },
    11: { label: 'November', note: 'Post-Diwali tail', mult: 1.05 },
    12: { label: 'December', note: 'Year-end slowdown', mult: 0.96 },
  };
  return map[month] || map[1];
}
