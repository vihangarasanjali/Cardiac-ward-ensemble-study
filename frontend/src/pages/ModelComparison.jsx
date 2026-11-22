import { useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useApi, fmt, FAMILY_COLORS, METRICS } from '../api';
import { PageHead, Note, Tabs, Loading, LoadError, Legend } from '../components/ui';
import BoxStrip from '../components/BoxStrip';

const FAMILIES = ['Baseline', 'Bagging', 'Boosting', 'Stacking'];
const SIG_METRICS = METRICS.filter((m) => ['AUC', 'PR_AUC', 'Recall', 'F1'].includes(m.key));

function RankTip({ active, payload, metricLabel }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{ background: '#fff', border: '1px solid var(--line)', padding: '8px 11px', fontSize: 13, borderRadius: 4 }}>
      <b>{d.model}</b><br />{metricLabel}: {fmt(d.mean)} ± {fmt(d.sd)}
    </div>
  );
}

export default function ModelComparison() {
  const ov = useApi('/api/overview');
  const fo = useApi('/api/folds');
  const sg = useApi('/api/significance');
  const [tab, setTab] = useState('rank');
  const [metric, setMetric] = useState('AUC');
  const [fams, setFams] = useState(FAMILIES);
  const [sigMetric, setSigMetric] = useState('AUC');

  const lowerBetter = metric === 'Brier';
  const metricLabel = METRICS.find((m) => m.key === metric).label;

  const rows = useMemo(() => {
    if (!ov.data) return [];
    return ov.data.summary
      .filter((s) => fams.includes(s.Family))
      .map((s) => ({ model: s.Model, family: s.Family, mean: s[`${metric}_mean`], sd: s[`${metric}_std`] }))
      .sort((a, b) => (lowerBetter ? a.mean - b.mean : b.mean - a.mean));
  }, [ov.data, fams, metric, lowerBetter]);

  const spread = useMemo(() => {
    if (!fo.data) return [];
    return rows.map((r) => ({ label: r.model, color: FAMILY_COLORS[r.family], values: fo.data.filter((f) => f.Model === r.model).map((f) => f[metric]) }));
  }, [fo.data, rows, metric]);

  if (ov.loading || fo.loading || sg.loading) return <Loading />;
  if (ov.error || fo.error || sg.error) return <LoadError />;

  const toggle = (f) => setFams(fams.includes(f) ? fams.filter((x) => x !== f) : [...fams, f]);
  const sigRows = sg.data.filter((r) => r.metric === sigMetric).sort((a, b) => a.p - b.p);
  const nSig = sigRows.filter((r) => r.p < 0.05).length;
  const sigLabel = SIG_METRICS.find((m) => m.key === sigMetric).label;
  const domain = lowerBetter ? undefined : [0, 1];

  return (
    <>
      <PageHead title="Model comparison">
        Eleven models in three ensemble families, scored on the same 25 cross-validation splits.
      </PageHead>
      <div className="content">
        <Tabs value={tab} onChange={setTab} tabs={[{ key: 'rank', label: 'Ranking' }, { key: 'spread', label: 'Spread across folds' }, { key: 'sig', label: 'Against Random Forest' }]} />

        {tab !== 'sig' && (
          <div className="toolbar">
            <label className="field">Metric
              <select value={metric} onChange={(e) => setMetric(e.target.value)}>
                {METRICS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
              </select>
            </label>
            {FAMILIES.map((f) => (
              <label key={f} className="check">
                <input type="checkbox" checked={fams.includes(f)} onChange={() => toggle(f)} />
                <span className="dot" style={{ background: FAMILY_COLORS[f], margin: 0 }} />{f}
              </label>
            ))}
          </div>
        )}

        {tab === 'rank' && (
          <section>
            <div className="chart">
              <p className="chart-title">Mean {metricLabel.toLowerCase()} by model</p>
              <p className="chart-sub">Bars show the mean; whiskers show one standard deviation across folds.</p>
              {rows.length === 0 ? <p className="muted">Select at least one model family.</p> : (
                <ResponsiveContainer width="100%" height={rows.length * 38 + 40}>
                  <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 36, left: 4, bottom: 4 }}>
                    <CartesianGrid horizontal={false} stroke="var(--line-soft)" />
                    <XAxis type="number" domain={lowerBetter ? [0, 'auto'] : [0, 1]} tick={{ fontSize: 12, fill: 'var(--muted)' }} tickLine={false} />
                    <YAxis type="category" dataKey="model" width={200} tick={{ fontSize: 13, fill: 'var(--ink)' }} tickLine={false} axisLine={false} />
                    <Tooltip cursor={{ fill: 'var(--line-soft)' }} content={<RankTip metricLabel={metricLabel} />} />
                    <Bar dataKey="mean" barSize={15} radius={[0, 2, 2, 0]}>
                      {rows.map((r) => <Cell key={r.model} fill={FAMILY_COLORS[r.family]} />)}
                      <ErrorBar dataKey="sd" direction="x" stroke="var(--ink)" width={5} strokeWidth={1.3} />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
              <Legend items={FAMILIES.map((f) => ({ label: f, color: FAMILY_COLORS[f] }))} />
            </div>
            <Note>AUC and PR-AUC measure how well a model ranks patients by risk and do not depend on a cutoff. Recall, precision and F1 use a 0.5 cutoff, so they mostly show how each model's probabilities behave when only one patient in five dies.</Note>

            <h2 style={{ marginTop: 34, marginBottom: 12 }}>All metrics</h2>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Model</th>{METRICS.map((m) => <th key={m.key} className="num">{m.label.replace(' (lower is better)', '')}</th>)}</tr></thead>
                <tbody>
                  {ov.data.summary.map((s) => (
                    <tr key={s.Model}>
                      <td><span className="dot" style={{ background: FAMILY_COLORS[s.Family] }} />{s.Model}</td>
                      {METRICS.map((m) => <td key={m.key} className="num">{fmt(s[`${m.key}_mean`])} ± {fmt(s[`${m.key}_std`], 2)}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {tab === 'spread' && (
          <section>
            <div className="chart">
              <p className="chart-title">{metricLabel} in each of the 25 folds</p>
              <p className="chart-sub">Each dot is one fold. Boxes span the middle half and the thick line is the median.</p>
              {spread.length ? <BoxStrip rows={spread} xLabel={metricLabel} domain={domain} /> : <p className="muted">Select at least one model family.</p>}
            </div>
            <Note>Each test fold holds about four deaths, so one fold can swing a score a long way. Boxes that overlap heavily mean the data cannot tell those models apart.</Note>
          </section>
        )}

        {tab === 'sig' && (
          <section>
            <p className="lede muted">Random Forest is the reference. Folds share training data, so the p-values use the Nadeau-Bengio corrected resampled t-test, which is stricter than a plain paired test.</p>
            <div className="toolbar">
              <label className="field">Metric
                <select value={sigMetric} onChange={(e) => setSigMetric(e.target.value)}>
                  {SIG_METRICS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
                </select>
              </label>
            </div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Compared with Random Forest</th><th className="num">Difference in mean (RF minus other)</th><th className="num">p-value</th><th>Result</th></tr></thead>
                <tbody>
                  {sigRows.map((r) => (
                    <tr key={r.other}>
                      <td>{r.other}</td>
                      <td className="num">{r.mean_diff > 0 ? '+' : ''}{fmt(r.mean_diff)}</td>
                      <td className="num">{fmt(r.p)}</td>
                      <td className={r.p < 0.05 ? 'sig-yes' : 'muted'}>{r.p < 0.05 ? 'Different at the 5% level' : 'Not distinguishable'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Note tone="caution">On {sigLabel}, {nSig} of {sigRows.length} comparisons reach p below 0.05. With 112 patients and 19 deaths, a non-significant result means the data cannot separate the models. It does not show that they perform equally.</Note>
          </section>
        )}
      </div>
    </>
  );
}
