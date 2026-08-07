import { useState, useRef, useEffect } from 'react';

export default function SearchableDropdown({
  options = [],
  value,
  onChange,
  placeholder = 'Select…',
  searchPlaceholder = 'Type to search…',
  disabled = false,
  id,
}) {
  const [open, setOpen]       = useState(false);
  const [query, setQuery]     = useState('');
  const [hovered, setHovered] = useState(-1);
  const containerRef          = useRef(null);
  const searchRef             = useRef(null);

  const filtered = query.trim()
    ? options.filter(o => String(o).toLowerCase().includes(query.toLowerCase().trim()))
    : options;

  useEffect(() => {
    const handler = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setOpen(false);
      }
    };
    if (open) document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  useEffect(() => {
    if (!open) { setQuery(''); setHovered(-1); }
    else setTimeout(() => searchRef.current?.focus(), 40);
  }, [open]);

  const handleKeyDown = (e) => {
    if (e.key === 'Escape')    { setOpen(false); return; }
    if (e.key === 'ArrowDown') { e.preventDefault(); setHovered(h => Math.min(h + 1, filtered.length - 1)); }
    if (e.key === 'ArrowUp')   { e.preventDefault(); setHovered(h => Math.max(h - 1, 0)); }
    if (e.key === 'Enter' && hovered >= 0) { e.preventDefault(); select(filtered[hovered]); }
  };

  const select = (opt) => {
    onChange(opt);
    setOpen(false);
  };

  return (
    <div className="sdd" ref={containerRef}>
      {}
      <button
        type="button"
        id={id}
        className={`sdd-trigger${!value ? ' sdd-empty' : ''}${disabled ? ' sdd-disabled' : ''}`}
        onClick={() => !disabled && setOpen(o => !o)}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="sdd-val">{value || placeholder}</span>
        {}
        <svg
          className={`sdd-chevron${open ? ' sdd-chevron-open' : ''}`}
          width="14" height="14" viewBox="0 0 24 24"
          fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {}
      {open && (
        <div className="sdd-panel" role="listbox">
          {}
          <div className="sdd-search-row">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="2.2" strokeLinecap="round">
              <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
            </svg>
            <input
              ref={searchRef}
              className="sdd-search"
              type="text"
              value={query}
              onChange={e => { setQuery(e.target.value); setHovered(-1); }}
              onKeyDown={handleKeyDown}
              placeholder={searchPlaceholder}
              autoComplete="off"
            />
            {query && (
              <button type="button" className="sdd-clear" onClick={() => setQuery('')} tabIndex={-1}>
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <path d="M18 6 6 18M6 6l12 12"/>
                </svg>
              </button>
            )}
          </div>

          {}
          <div className="sdd-list">
            {filtered.length === 0 ? (
              <p className="sdd-empty">No results for "{query}"</p>
            ) : filtered.map((opt, idx) => (
              <button
                key={opt}
                type="button"
                className={`sdd-option${value === opt ? ' sdd-selected' : ''}${hovered === idx ? ' sdd-focused' : ''}`}
                onClick={() => select(opt)}
                onMouseEnter={() => setHovered(idx)}
                role="option"
                aria-selected={value === opt}
              >
                {opt}
                {value === opt && (
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                )}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
