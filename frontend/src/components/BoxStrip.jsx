import { quantile } from '../api';

// Horizontal box plot with every observation drawn as a point (one row per group).
export default function BoxStrip({ rows, xLabel, rowHeight = 34, labelWidth = 190, domain }) {
  const W = 760, right = 24, top = 8, axisH = 44;
  const plotW = W - labelWidth - right;
  const all = rows.flatMap((r) => r.values);
  const lo0 = domain ? domain[0] : Math.min(...all), hi0 = domain ? domain[1] : Math.max(...all);
  const pad = domain ? 0 : (hi0 - lo0) * 0.04 || 1;
  const lo = lo0 - pad, hi = hi0 + pad;
  const x = (v) => labelWidth + ((v - lo) / (hi - lo)) * plotW;
  const H = top + rows.length * rowHeight + axisH;
  const ticks = Array.from({ length: 6 }, (_, i) => lo + ((hi - lo) * i) / 5);
  const digits = hi - lo > 20 ? 0 : hi - lo > 2 ? 1 : 2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`Distribution of ${xLabel} by group`} style={{ display: 'block' }}>
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={x(t)} x2={x(t)} y1={top} y2={H - axisH + 4} stroke="var(--line-soft)" />
          <text x={x(t)} y={H - axisH + 20} textAnchor="middle" fontSize="12" fill="var(--muted)">{t.toFixed(digits)}</text>
        </g>
      ))}
      <text x={labelWidth + plotW / 2} y={H - 6} textAnchor="middle" fontSize="12.5" fill="var(--text)">{xLabel}</text>
      {rows.map((r, i) => {
        const v = [...r.values].sort((a, b) => a - b);
        const cy = top + i * rowHeight + rowHeight / 2;
        const q1 = quantile(v, 0.25), q2 = quantile(v, 0.5), q3 = quantile(v, 0.75);
        return (
          <g key={r.label}>
            <text x={labelWidth - 12} y={cy + 4} textAnchor="end" fontSize="13" fill="var(--ink)">{r.label}</text>
            <line x1={x(v[0])} x2={x(v[v.length - 1])} y1={cy} y2={cy} stroke={r.color} strokeWidth="1.4" />
            <rect x={x(q1)} y={cy - 8} width={Math.max(1, x(q3) - x(q1))} height="16" fill={r.color} fillOpacity="0.16" stroke={r.color} strokeWidth="1.2" />
            <line x1={x(q2)} x2={x(q2)} y1={cy - 8} y2={cy + 8} stroke={r.color} strokeWidth="2.6" />
            {v.map((val, k) => (
              <circle key={k} cx={x(val)} cy={cy + (((k * 37) % 11) - 5) * 1.35} r="2.1" fill={r.color} fillOpacity="0.5" />
            ))}
          </g>
        );
      })}
    </svg>
  );
}
