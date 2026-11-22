import { useState } from 'react';
import { ChevronDown, ChevronRight, Info, TriangleAlert, LoaderCircle, ServerOff } from 'lucide-react';

export function PageHead({ title, children }) {
  return (
    <header className="page-head">
      <h1>{title}</h1>
      {children && <p>{children}</p>}
      <svg className="ecg" viewBox="0 0 330 48" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M0 28h58l8-6 8 6h26l6 6 6-26 8 38 7-18h16l8-5 9 5h46l7 4 7-20 7 26 6-10h45" />
      </svg>
    </header>
  );
}

export function Note({ children, tone = 'info' }) {
  const Icon = tone === 'caution' ? TriangleAlert : Info;
  return (
    <div className={`note ${tone === 'caution' ? 'caution' : ''}`} role="note">
      <Icon size={18} aria-hidden="true" />
      <p>{children}</p>
    </div>
  );
}

export function Tabs({ tabs, value, onChange }) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((t) => (
        <button key={t.key} role="tab" className="tab" aria-selected={value === t.key} onClick={() => onChange(t.key)}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function Disclosure({ label, children }) {
  const [open, setOpen] = useState(false);
  const Icon = open ? ChevronDown : ChevronRight;
  return (
    <div className="disclosure">
      <button onClick={() => setOpen(!open)} aria-expanded={open}>
        <Icon size={16} aria-hidden="true" />
        {label}
      </button>
      {open && <div>{children}</div>}
    </div>
  );
}

export function Loading() {
  return (
    <div className="state">
      <LoaderCircle size={22} aria-hidden="true" className="muted" />
      <p className="muted">Loading results.</p>
    </div>
  );
}

export function LoadError() {
  return (
    <div className="state">
      <ServerOff size={26} aria-hidden="true" className="muted" />
      <h2>The results service is not reachable</h2>
      <p>Start the API from the project root with <code>uvicorn backend.main:app --port 8000</code>. If the results folder is empty, run <code>python scripts/run_experiments.py</code> first, then reload this page.</p>
    </div>
  );
}

export function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((i) => (
        <span key={i.label}><span className="dot" style={{ background: i.color }} />{i.label}</span>
      ))}
    </div>
  );
}
