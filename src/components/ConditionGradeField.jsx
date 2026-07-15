import { formatINR } from '../utils/format.js';
import { GRADE_OPTIONS, getGradePreview } from '../utils/wheelrCosts.js';

const CATEGORY_LABELS = {
  engine: 'Engine Block & Mechanicals',
  tyre: 'Tyre Tread & Alignment',
  body: 'Body panels, Paint & Dent Repair',
  interior: 'Cabin Interior & Detailing',
  electrical: 'Electrical Systems & AC',
};

export default function ConditionGradeField({ category, grade, vendorType, onGradeChange, onVendorChange }) {
  const options = GRADE_OPTIONS[category] || [];
  const preview = getGradePreview(category, grade);
  const vendor = vendorType[category] || 'vendor';
  const selectedCost = vendor === 'inhouse' ? preview.inhouse : preview.vendor;

  return (
    <div className="rs2-card" style={{ padding: 18, marginBottom: 14, border: '1px solid #e2e8f0', borderRadius: 12, background: '#ffffff', display: 'flex', flexDirection: 'column', gap: 12 }}>
      
      {/* Category Title */}
      <div style={{ fontSize: 13.5, fontWeight: 750, color: '#0f172a', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
        {CATEGORY_LABELS[category] || category}
      </div>

      {/* Grade Selector Row */}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {options.map(opt => {
          const isSel = grade === opt.value;
          return (
            <button
              key={opt.value}
              type="button"
              onClick={() => onGradeChange(category, opt.value)}
              style={{
                flex: 1,
                minWidth: '70px',
                border: isSel ? '2px solid #ea580c' : '1px solid #cbd5e1',
                background: isSel ? '#ffedd5' : '#ffffff',
                color: isSel ? '#c2410c' : '#334155',
                padding: '6px 10px',
                fontSize: 12,
                fontWeight: 700,
                borderRadius: 6,
                cursor: 'pointer',
                transition: 'all 0.15s'
              }}
            >
              {opt.label}
            </button>
          );
        })}
      </div>

      {/* Vendor Type Selection */}
      <div style={{ display: 'flex', gap: 6, border: '1px solid #e2e8f0', background: '#f8fafc', padding: 4, borderRadius: 8 }}>
        <button
          type="button"
          onClick={() => onVendorChange(category, 'inhouse')}
          style={{
            flex: 1,
            border: 'none',
            padding: '5px 12px',
            fontSize: 11.5,
            fontWeight: 700,
            borderRadius: 6,
            cursor: 'pointer',
            background: vendor === 'inhouse' ? '#ea580c' : 'transparent',
            color: vendor === 'inhouse' ? '#ffffff' : '#475569',
            transition: 'all 0.12s'
          }}
        >
          In-House Repair
        </button>
        <button
          type="button"
          onClick={() => onVendorChange(category, 'vendor')}
          style={{
            flex: 1,
            border: 'none',
            padding: '5px 12px',
            fontSize: 11.5,
            fontWeight: 700,
            borderRadius: 6,
            cursor: 'pointer',
            background: vendor === 'vendor' ? '#ea580c' : 'transparent',
            color: vendor === 'vendor' ? '#ffffff' : '#475569',
            transition: 'all 0.12s'
          }}
        >
          Outsourced Vendor
        </button>
      </div>

      {/* Live Cost Breakdown Grid (No raw text string) */}
      <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: 4 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
          <span style={{ color: '#475569', fontWeight: 600 }}>Active Repair Cost:</span>
          <strong style={{ color: '#ea580c', fontWeight: 800 }}>{formatINR(selectedCost)}</strong>
        </div>
        <div style={{ display: 'flex', gap: 14, color: '#64748b', fontSize: 10.5, borderTop: '1px solid #e2e8f0', paddingTop: 6, marginTop: 2 }}>
          <span>In-House standard: <strong>{formatINR(preview.inhouse)}</strong></span>
          <span>Vendor standard: <strong>{formatINR(preview.vendor)}</strong></span>
        </div>
      </div>

    </div>
  );
}

const GRADE_KEYS = ['engine', 'tyre', 'body', 'interior', 'electrical'];

export function ConditionGradesSection({ grades, vendorType, onGradeChange, onVendorChange }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
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
