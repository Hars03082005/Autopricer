import { useEffect, useMemo, useRef, useState } from 'react';
import Icon from './Icon.jsx';

export default function SearchableSelect({
  label,
  value,
  options = [],
  placeholder = 'Search…',
  disabled = false,
  onChange,
  icon,          
  sublabel,      
}) {
  const wrapRef    = useRef(null);
  const inputRef   = useRef(null);
  const listRef    = useRef(null);
  const [open,   setOpen]   = useState(false);
  const [query,  setQuery]  = useState('');
  const [cursor, setCursor] = useState(-1);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return options;
    return options.filter(opt => opt.toLowerCase().includes(q));
  }, [options, query]);

  useEffect(() => {
    const handleOutside = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    };
    
    document.addEventListener('mousedown', handleOutside);
    document.addEventListener('touchstart', handleOutside, { passive: true });
    return () => {
      document.removeEventListener('mousedown', handleOutside);
      document.removeEventListener('touchstart', handleOutside);
    };
  }, []);

  useEffect(() => {
    if (!open) {
      setTimeout(() => {
        setQuery('');
        setCursor(-1);
      }, 0);
    } else {
      
      const isTouch = window.matchMedia('(pointer: coarse)').matches;
      if (!isTouch) {
        setTimeout(() => inputRef.current?.focus(), 0);
      }
    }
  }, [open]);

  useEffect(() => {
    if (cursor >= 0 && listRef.current) {
      const el = listRef.current.children[cursor];
      el?.scrollIntoView({ block: 'nearest' });
    }
  }, [cursor]);

  const handleKeyDown = (e) => {
    if (!open) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen(true); } return; }
    if (e.key === 'Escape') { setOpen(false); return; }
    if (e.key === 'ArrowDown') { e.preventDefault(); setCursor(c => Math.min(c + 1, filtered.length - 1)); return; }
    if (e.key === 'ArrowUp')   { e.preventDefault(); setCursor(c => Math.max(c - 1, 0)); return; }
    if (e.key === 'Enter' && cursor >= 0) {
      e.preventDefault();
      if (filtered[cursor]) { onChange(filtered[cursor]); setOpen(false); }
    }
  };

  const handleSelect = (option) => {
    onChange(option);
    setOpen(false);
  };

  const highlight = (text) => {
    const q = query.trim().toLowerCase();
    if (!q) return text;
    const idx = text.toLowerCase().indexOf(q);
    if (idx < 0) return text;
    return (
      <>
        {text.slice(0, idx)}
        <mark className="ss-mark">{text.slice(idx, idx + q.length)}</mark>
        {text.slice(idx + q.length)}
      </>
    );
  };

  return (
    <div className="searchable-select" ref={wrapRef} onKeyDown={handleKeyDown}>
      {label && (
        <label className="spec-label ss-label-row">
          {icon && <span className="ss-icon">{icon}</span>}
          <span>{label}</span>
          {sublabel && <span className="ss-sublabel">{sublabel}</span>}
        </label>
      )}

      {}
      <button
        type="button"
        className={`searchable-select-trigger cd-input ${disabled ? 'disabled' : ''} ${open ? 'open' : ''}`}
        onClick={() => !disabled && setOpen(o => !o)}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className={value ? 'searchable-select-value' : 'searchable-select-placeholder'}>
          {value || placeholder}
        </span>
        <span className={`ss-caret ${open ? 'rotate' : ''}`}>
          <Icon name="chevronDown" size={14} color="#888" strokeWidth={2} />
        </span>
      </button>

      {}
      {open && (
        <div className="searchable-select-menu" role="listbox">
          {}
          <div className="searchable-select-search">
            <Icon name="search" size={14} color="#aaa" strokeWidth={2} />
            <input
              ref={inputRef}
              className="searchable-select-input"
              value={query}
              onChange={e => { setQuery(e.target.value); setCursor(-1); }}
              placeholder={placeholder}
            />
            {query && (
              <button
                type="button"
                className="ss-clear-btn"
                onClick={() => { setQuery(''); setCursor(-1); inputRef.current?.focus(); }}
              >×</button>
            )}
            {filtered.length > 0 && (
              <span className="ss-count">{filtered.length}</span>
            )}
          </div>

          {}
          <div className="searchable-select-options" ref={listRef}>
            {filtered.length === 0 && (
              <div className="searchable-select-empty">
                <span>🔍</span> No matches for &ldquo;{query}&rdquo;
              </div>
            )}
            {filtered.map((option, i) => {
              const isActive  = option === value;
              const isCursor  = i === cursor;
              return (
                <button
                  key={option}
                  type="button"
                  className={`searchable-select-option ${isActive ? 'active' : ''} ${isCursor ? 'cursor' : ''}`}
                  onPointerDown={(e) => e.preventDefault()} 
                  onClick={() => handleSelect(option)}
                  onMouseEnter={() => setCursor(i)}
                  role="option"
                  aria-selected={isActive}
                >
                  <span className="ss-opt-text">{highlight(option)}</span>
                  {isActive && <span className="ss-check">✓</span>}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
