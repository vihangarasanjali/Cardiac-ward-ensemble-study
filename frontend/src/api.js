import { useEffect, useState } from 'react';

const cache = new Map();
export const clearApiCache = () => cache.clear();

export function useApi(path) {
  const [state, setState] = useState({ data: cache.get(path) ?? null, error: null, loading: !cache.has(path) });
  useEffect(() => {
    if (cache.has(path)) return;
    let live = true;
    fetch(path)
      .then((r) => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then((data) => { cache.set(path, data); if (live) setState({ data, error: null, loading: false }); })
      .catch((error) => { if (live) setState({ data: null, error, loading: false }); });
    return () => { live = false; };
  }, [path]);
  return state;
}

export const FAMILY_COLORS = { Baseline: 'var(--baseline)', Bagging: 'var(--bagging)', Boosting: 'var(--boosting)', Stacking: 'var(--stacking)' };

export const METRICS = [
  { key: 'AUC', label: 'ROC-AUC' },
  { key: 'PR_AUC', label: 'PR-AUC' },
  { key: 'Recall', label: 'Recall (deaths caught)' },
  { key: 'Precision', label: 'Precision' },
  { key: 'F1', label: 'F1 score' },
  { key: 'BalAcc', label: 'Balanced accuracy' },
  { key: 'Brier', label: 'Brier score (lower is better)' },
];

export const fmt = (x, d = 3) => (x == null || Number.isNaN(x) ? '-' : Number(x).toFixed(d));

export function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export async function getJson(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

export async function postJson(path, body) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const detail = Array.isArray(data.detail) ? 'Check that every field holds a number.' : data.detail;
    throw new Error(detail || `Request failed (${r.status})`);
  }
  return data;
}
