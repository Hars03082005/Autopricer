import { formatINR } from '../utils/mockData.js';

export default function PriceRangeBar({ min, predicted, max }) {
  const range = max - min;
  const predPct = range > 0 ? ((predicted - min) / range) * 100 : 50;

  return (
    <div className="price-range-bar-wrapper">
      <div className="price-range-labels">
        <span className="pr-label-min">{formatINR(min)}</span>
        <span className="pr-label-mid">Confidence Range</span>
        <span className="pr-label-max">{formatINR(max)}</span>
      </div>
      <div className="price-range-track">
        {}
        <div className="price-range-fill" />
        {}
        <div
          className="price-range-marker"
          style={{ left: `${Math.max(5, Math.min(95, predPct))}%` }}
        >
          <div className="price-range-marker-line" />
          <div className="price-range-marker-label">{formatINR(predicted)}</div>
        </div>
      </div>
      <div className="price-range-ci">
        ± {formatINR(Math.round((max - predicted)))} confidence interval
      </div>
    </div>
  );
}
