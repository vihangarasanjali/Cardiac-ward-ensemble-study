import { useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useApi, fmt } from '../api';
import { PageHead, Note, Loading, LoadError, Disclosure, Legend } from '../components/ui';
import BoxStrip from '../components/BoxStrip';

const SURVIVE = '#4d8da1', DEATH = '#b42318';

function histogram(report) {
  const labels = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', 'over 100'];
  const bins = labels.map((l) => ({ bin: l, Survived: 0, Died: 0 }));
  report.readings_per_patient.forEach((n, i) => {
    const k = Math.min(Math.floor((n - 1) / 10), 10);
    bins[k][report.readings_outcome[i] === 1 ? 'Survived' : 'Died'] += 1;
  });
  return bins;
}

export default function DataOverview() {
  const ov = useApi('/api/overview');
  const pt = useApi('/api/patients');
  const [feature, setFeature] = useState('Ceratinine');
  const report = ov.data?.report;
  const bins = useMemo(() => (report ? histogram(report) : []), [report]);
  const features = useMemo(() => (pt.data ? Object.keys(pt.data[0]).filter((k) => !['ID', 'Death', 'n_window'].includes(k)) : []), [pt.data]);

  if (ov.loading || pt.loading) return <Loading />;
  if (ov.error || pt.error) return <LoadError />;

  const total = report.survived + report.died;
  const removed = Object.entries(report.impossible_values_removed);
  const maxRemoved = Math.max(...removed.map(([, n]) => n), 1);
  const died = report.died_with_at_least;
  const groups = [
    { label: 'Survived', color: SURVIVE, values: pt.data.filter((p) => p.Death === 0 && p[feature] != null).map((p) => p[feature]) },
    { label: 'Died', color: DEATH, values: pt.data.filter((p) => p.Death === 1 && p[feature] != null).map((p) => p[feature]) },
  ].filter((g) => g.values.length);

  return (
    <>
      <PageHead title="Data overview">
        What the raw file contains, what was removed, and how it became one row per patient.
      </PageHead>
      <div className="content">
        <section className="section">
          <h2>The cohort</h2>
          <p className="lede muted">{report.raw_rows.toLocaleString()} hourly rows from {report.patients} patients. The outcome never changes within a patient, so every patient is one observation.</p>
          <div className="balance" role="img" aria-label={`${report.survived} survived, ${report.died} died`}>
            <div className="a" style={{ width: `${(report.survived / total) * 100}%` }}>{report.survived} survived</div>
            <div className="b" style={{ width: `${(report.died / total) * 100}%` }}>{report.died}</div>
          </div>
          <div className="balance-legend"><span>About five survivors for every death</span><span>{report.died} deaths ({Math.round((report.died / total) * 100)}%)</span></div>
          <Note>With this imbalance, a model that predicts survival for everyone would be {Math.round((report.survived / total) * 100)}% accurate and catch no deaths. That is why the comparison leans on AUC, PR-AUC and recall.</Note>
        </section>

        <section className="section">
          <h2>Length of stay differs by outcome</h2>
          <p className="lede muted">Median stay was {report.median_readings_survived} hourly readings for survivors and {report.median_readings_died} for patients who died.</p>
          <div className="chart">
            <p className="chart-title">Patients by number of hourly readings</p>
            <Legend items={[{ label: 'Survived', color: SURVIVE }, { label: 'Died', color: DEATH }]} />
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={bins} margin={{ top: 4, right: 8, left: -8, bottom: 4 }}>
                <CartesianGrid vertical={false} stroke="var(--line-soft)" />
                <XAxis dataKey="bin" tick={{ fontSize: 12, fill: 'var(--muted)' }} tickLine={false} />
                <YAxis allowDecimals={false} tick={{ fontSize: 12, fill: 'var(--muted)' }} tickLine={false} axisLine={false} />
                <Tooltip cursor={{ fill: 'var(--line-soft)' }} />
                <Bar dataKey="Survived" fill={SURVIVE} radius={[2, 2, 0, 0]} />
                <Bar dataKey="Died" fill={DEATH} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Note>A patient who dies leaves fewer readings behind, so stay length would tell a model the answer. It is kept out of the predictors.</Note>
        </section>

        <section className="section">
          <h2>Why the first {report.window} readings</h2>
          <div className="split">
            <div className="table-wrap">
              <table>
                <thead><tr><th>Readings required</th><th className="num">Deaths kept</th><th className="num">Share of {report.died}</th></tr></thead>
                <tbody>
                  {[6, 12, 24].map((h) => (
                    <tr key={h}><td>At least {h}</td><td className="num">{died[h]}</td><td className="num">{Math.round((died[h] / report.died) * 100)}%</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div>
              <p>Most deaths happen early. Requiring 24 readings would leave only {died[24]} of the {report.died} deaths to learn from.</p>
              <p>Patients with fewer than {report.window} readings stay in the data and are summarised from whatever they have.</p>
            </div>
          </div>
        </section>

        <section className="section">
          <h2>Cleaning</h2>
          <p className="lede muted">Readings outside physiological limits were set to missing before any summary was calculated.</p>
          <div className="split">
            <div className="table-wrap">
              <table className="inline-bars">
                <thead><tr><th>Vital sign</th><th className="num">Values removed</th><th></th></tr></thead>
                <tbody>
                  {removed.map(([k, n]) => (
                    <tr key={k}><td>{k}</td><td className="num">{n}</td><td><div className="bar-track"><div className="bar-fill" style={{ width: `${(n / maxRemoved) * 100}%` }} /></div></td></tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p>Glasgow Coma Scale accounts for most removals: the scale tops out at 15, but the file contains values up to 100. Diastolic pressure of 999 and respiratory rates above 60 are other examples. Body temperature is stored in whole degrees Fahrenheit.</p>
          </div>
        </section>

        <section className="section">
          <h2>Explore a predictor</h2>
          <p className="lede muted">Each dot is one patient. The box covers the middle half of each group and the thick line marks the median.</p>
          <div className="toolbar">
            <label className="field">Predictor
              <select value={feature} onChange={(e) => setFeature(e.target.value)}>
                {features.map((f) => <option key={f} value={f}>{f}</option>)}
              </select>
            </label>
          </div>
          <div className="chart"><BoxStrip rows={groups} xLabel={feature} rowHeight={64} labelWidth={110} /></div>
          <Disclosure label="Missing values in the raw file">
            <div className="table-wrap" style={{ maxWidth: 420 }}>
              <table><thead><tr><th>Column</th><th className="num">Missing rows</th></tr></thead>
                <tbody>{Object.entries(report.raw_missing_per_column).map(([k, n]) => <tr key={k}><td>{k}</td><td className="num">{n}</td></tr>)}</tbody>
              </table>
            </div>
          </Disclosure>
          <Disclosure label="Patient-level table">
            <div className="table-wrap scroll-y">
              <table>
                <thead><tr>{Object.keys(pt.data[0]).map((c) => <th key={c}>{c}</th>)}</tr></thead>
                <tbody>{pt.data.map((r) => <tr key={r.ID}>{Object.keys(r).map((c) => <td key={c}>{typeof r[c] === 'number' && !Number.isInteger(r[c]) ? fmt(r[c], 2) : r[c] ?? '-'}</td>)}</tr>)}</tbody>
              </table>
            </div>
          </Disclosure>
        </section>
      </div>
    </>
  );
}
