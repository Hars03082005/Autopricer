import { formatINR } from '../utils/format.js';

export function NegotiationPlaybook({ negotiation, variant = 'enhanced', confidenceScore }) {
  const opening  = variant === 'reverse' ? negotiation?.opening  : negotiation?.opening_offer;
  const target   = variant === 'reverse' ? negotiation?.target   : negotiation?.target_offer;
  const walkAway = variant === 'reverse' ? negotiation?.walk_away : negotiation?.walk_away_price;

  const confScore = confidenceScore ?? null;
  const confCls   = confScore >= 75 ? 'good'   : confScore >= 55 ? 'medium'  : 'bad';

  return (
    <div className="negotiation-playbook">
      {/* Section title + confidence badge */}
      <div className="neg-playbook-header">
        <div className="cd-section-label" style={{ marginBottom: 0 }}>
          Price Range &amp; Negotiation Guide
        </div>
        {confScore != null && (
          <span className={`confidence-pill ${confCls}`} style={{ fontSize: 10, padding: '3px 8px' }}>
            🛡 {confScore}% confidence
          </span>
        )}
      </div>

      {/* Intro description */}
      <div className="neg-playbook-intro">
        Predicted acquisition price range based on ML valuation + rule engine adjustments
      </div>

      {/* Two range cards — no middle value */}
      <div className="negotiation-trio" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <div className="negotiation-card opening">
          <div className="trio-label">RANGE FLOOR</div>
          <div className="trio-sublabel">(Open at)</div>
          <div className="trio-price">{formatINR(opening)}</div>
        </div>
        <div className="negotiation-card walk-away">
          <div className="trio-label">RANGE CEILING</div>
          <div className="trio-sublabel">(Max pay)</div>
          <div className="trio-price">{formatINR(walkAway)}</div>
        </div>
      </div>

      {/* One-line confidence summary (Change 3) */}
      {confScore != null && opening != null && walkAway != null && (
        <div className="neg-playbook-summary">
          Model is <strong>{confScore}% confident</strong> the fair buy price falls between{' '}
          <strong>{formatINR(opening)}</strong> and <strong>{formatINR(walkAway)}</strong>
        </div>
      )}
    </div>
  );
}

export function ExpandableBreakdownTable({ title, rows, totalLabel, totalValue }) {
  return (
    <details className="expand-breakdown">
      <summary>{title}</summary>
      <div className="breakdown-table-wrap">
        <table className="breakdown-table">
          <thead>
            <tr>
              {Object.keys(rows[0] || {}).map(h => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {Object.values(row).map((v, j) => (
                  <td key={j}>{typeof v === 'number' ? formatINR(v) : v}</td>
                ))}
              </tr>
            ))}
            {totalLabel && (
              <tr className="breakdown-total-row">
                <td colSpan={Math.max(1, Object.keys(rows[0] || {}).length - 1)}><strong>{totalLabel}</strong></td>
                <td><strong>{formatINR(totalValue)}</strong></td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </details>
  );
}

export function DealHealthBanner({ dealHealth, meta }) {
  const info = meta[dealHealth] || meta.yellow;
  return (
    <div className={`deal-health-banner deal-health-${dealHealth}`}>
      <div className="deal-health-badge">{dealHealth?.toUpperCase()}</div>
      <div>
        <div className="deal-health-title">{info.title}</div>
        <div className="deal-health-reason">{info.reason}</div>
      </div>
    </div>
  );
}
