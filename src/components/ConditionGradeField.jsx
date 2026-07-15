import { formatINR } from '../utils/format.js';
import { GRADE_OPTIONS, getGradePreview } from '../utils/wheelrCosts.js';

const CATEGORY_LABELS = {
  engine: 'Engine',
  tyre: 'Tyres',
  body: 'Body & Paint',
  interior: 'Interior',
  electrical: 'Electricals',
};

export default function ConditionGradeField({ category, grade, vendorType, onGradeChange, onVendorChange }) {
  const options = GRADE_OPTIONS[category] || [];
  const preview = getGradePreview(category, grade);
  const vendor = vendorType[category] || 'vendor';
  const selectedCost = vendor === 'inhouse' ? preview.inhouse : preview.vendor;

  return (
    <div className="grade-field">
      <label className="spec-label">{CATEGORY_LABELS[category]}</label>
      <div className="grade-pill-row">
        {options.map(opt => (
          <button
            key={opt.value}
            type="button"
            className={`grade-pill ${grade === opt.value ? 'active' : ''}`}
            onClick={() => onGradeChange(category, opt.value)}
          >
            {opt.label}
          </button>
        ))}
      </div>
      <div className="vendor-toggle-row">
        <button
          type="button"
          className={`dep-toggle-btn ${vendor === 'inhouse' ? 'active' : ''}`}
          onClick={() => onVendorChange(category, 'inhouse')}
        >
          In-house
        </button>
        <button
          type="button"
          className={`dep-toggle-btn ${vendor === 'vendor' ? 'active' : ''}`}
          onClick={() => onVendorChange(category, 'vendor')}
        >
          Vendor
        </button>
      </div>
      <div className="live-cost-preview">
        Est. {formatINR(selectedCost)} ({vendor === 'inhouse' ? 'in-house' : 'vendor'}) | {formatINR(preview.inhouse)} (in-house) | {formatINR(preview.vendor)} (vendor)
      </div>
    </div>
  );
}

const GRADE_KEYS = ['engine', 'tyre', 'body', 'interior', 'electrical'];

export function ConditionGradesSection({ grades, vendorType, onGradeChange, onVendorChange }) {
  return (
    <div className="grade-grid">
      {GRADE_KEYS.map(cat => (
        <ConditionGradeField
          key={cat}
          category={cat}
          grade={grades[cat]}
          vendorType={vendorType}
          onGradeChange={onGradeChange}
          onVendorChange={onVendorChange}
        />
      ))}
    </div>
  );
}
