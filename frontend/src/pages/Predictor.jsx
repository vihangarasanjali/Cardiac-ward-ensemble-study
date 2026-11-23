import { useState } from 'react';
import { CircleAlert, Eraser, LoaderCircle, Play, UserRound, UserRoundX } from 'lucide-react';
import { FAMILY_COLORS, fmt, getJson, postJson } from '../api';
import { Note, PageHead } from '../components/ui';

const VITALS = [
  { key: 'SBP', label: 'SBP', unit: 'mmHg' },
  { key: 'DBP', label: 'DBP', unit: 'mmHg' },
  { key: 'HR', label: 'Heart rate', unit: '/min' },
  { key: 'RR', label: 'Resp. rate', unit: '/min' },
  { key: 'BT', label: 'Temp.', unit: '°F' },
  { key: 'SpO2', label: 'SpO2', unit: '%' },
  { key: 'GCS', label: 'GCS', unit: '3 to 15' },
];
const LABS = [
  { key: 'Na', label: 'Sodium', unit: 'mmol/L' },
  { key: 'K', label: 'Potassium', unit: 'mmol/L' },
  { key: 'Cl', label: 'Chloride', unit: 'mmol/L' },
  { key: 'Urea', label: 'Urea', unit: 'as in source file' },
  { key: 'Ceratinine', label: 'Creatinine', unit: 'as in source file' },
];
const YESNO = [['', 'Unknown'], ['1', 'Yes'], ['0', 'No']];
const N_ROWS = 6;

const emptyReadings = () => Array.from({ length: N_ROWS }, () => Object.fromEntries(VITALS.map((v) => [v.key, ''])));
const emptyAdmission = () => ({ Age: '', Gender: '', TriageScore: '', Alcoholic: '', Smoke: '', FHCD: '', Na: '', K: '', Cl: '', Urea: '', Ceratinine: '' });
const toNum = (v) => (v === '' || v == null ? null : Number(v));
const toStr = (v) => (v == null ? '' : String(v));

export default function Predictor() {
  const [adm, setAdm] = useState(emptyAdmission());
  const [rows, setRows] = useState(emptyReadings());
  const [result, setResult] = useState(null);
  const [example, setExample] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [counters, setCounters] = useState({ survived: 0, died: 0 });

  const setA = (k, v) => setAdm((a) => ({ ...a, [k]: v }));
  const setR = (i, k, v) => setRows((rs) => rs.map((r, j) => (j === i ? { ...r, [k]: v } : r)));

  const clear = () => { setAdm(emptyAdmission()); setRows(emptyReadings()); setResult(null); setExample(null); setError(''); };

  async function loadExample(outcome) {
    setError(''); setResult(null);
    try {
      const ex = await getJson(`/api/example?outcome=${outcome}&i=${counters[outcome]}`);
      setCounters((c) => ({ ...c, [outcome]: c[outcome] + 1 }));
      setAdm(Object.fromEntries(Object.keys(emptyAdmission()).map((k) => [k, toStr(ex.admission[k])])));
      setRows(Array.from({ length: N_ROWS }, (_, i) => Object.fromEntries(VITALS.map((v) => [v.key, toStr(ex.readings[i]?.[v.key])]))));
      setExample(ex);
    } catch {
      setError('Could not load an example. Check that the results service is running.');
    }
  }

  async function submit(e) {
    e.preventDefault();
    setError(''); setBusy(true);
    const readings = rows.map((r) => Object.fromEntries(VITALS.map((v) => [v.key, toNum(r[v.key])])))
      .filter((r) => Object.values(r).some((x) => x != null));
    try {
      const out = await postJson('/api/predict', { readings, admission: Object.fromEntries(Object.entries(adm).map(([k, v]) => [k, toNum(v)])) });
      setResult(out);
    } catch (err) {
      setResult(null);
      setError(err.message === 'Failed to fetch' ? 'The results service is not reachable. Start the API and try again.' : err.message);
    } finally { setBusy(false); }
  }

  const above = result ? result.models.filter((m) => m.score >= 0.5).length : 0;

  return (
    <>
      <PageHead title="Risk predictor">
        Score one patient with the four saved ensemble models (bagging, boosting, stacking and the RF + ANN hybrid) from their first hourly readings and admission details.
      </PageHead>
      <div className="content">
        <div className="examples">
          Try a real patient from the dataset:
          <button type="button" className="btn ghost" onClick={() => loadExample('survived')}><UserRound size={16} aria-hidden="true" />Load a patient who survived</button>
          <button type="button" className="btn ghost" onClick={() => loadExample('died')}><UserRoundX size={16} aria-hidden="true" />Load a patient who died</button>
        </div>

        <form className="predictor" onSubmit={submit}>
          <div>
            <section className="form-block">
              <h2>Admission details</h2>
              <p className="lede">Leave a field empty if it was not recorded.</p>
              <div className="form-grid">
                <label className="fld"><span><b>Age</b> (years)</span><input type="number" min="0" max="120" value={adm.Age} onChange={(e) => setA('Age', e.target.value)} /></label>
                <label className="fld"><span><b>Sex</b></span>
                  <select value={adm.Gender} onChange={(e) => setA('Gender', e.target.value)}>{[['', 'Unknown'], ['1', 'Male'], ['0', 'Female']].map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>
                <label className="fld"><span><b>Triage score</b> (1 to 3)</span>
                  <select value={adm.TriageScore} onChange={(e) => setA('TriageScore', e.target.value)}>{[['', 'Unknown'], ['1', '1'], ['2', '2'], ['3', '3']].map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>
                <label className="fld"><span><b>Alcohol use</b></span>
                  <select value={adm.Alcoholic} onChange={(e) => setA('Alcoholic', e.target.value)}>{YESNO.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>
                <label className="fld"><span><b>Smoker</b></span>
                  <select value={adm.Smoke} onChange={(e) => setA('Smoke', e.target.value)}>{YESNO.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>
                <label className="fld"><span><b>Family history</b> of heart disease</span>
                  <select value={adm.FHCD} onChange={(e) => setA('FHCD', e.target.value)}>{YESNO.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>
                {LABS.map((l) => (
                  <label key={l.key} className="fld"><span><b>{l.label}</b> ({l.unit})</span>
                    <input type="number" step="any" value={adm[l.key]} onChange={(e) => setA(l.key, e.target.value)} /></label>
                ))}
              </div>
            </section>

            <section className="form-block">
              <h2>Hourly readings</h2>
              <p className="lede">Up to the first {N_ROWS} readings, one row per hour. Fill in as many as were taken; one reading is enough to run.</p>
              <div className="table-wrap">
                <table className="readings">
                  <thead><tr><th>Reading</th>{VITALS.map((v) => <th key={v.key}>{v.label}<div className="muted" style={{ fontWeight: 400 }}>{v.unit}</div></th>)}</tr></thead>
                  <tbody>
                    {rows.map((r, i) => (
                      <tr key={i}>
                        <td>Hour {i + 1}</td>
                        {VITALS.map((v) => (
                          <td key={v.key} className="rd"><input type="number" step="any" aria-label={`${v.label} at hour ${i + 1}`} value={r[v.key]} onChange={(e) => setR(i, v.key, e.target.value)} /></td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <div className="actions">
              <button type="submit" className="btn" disabled={busy}>
                {busy ? <LoaderCircle size={17} aria-hidden="true" /> : <Play size={17} aria-hidden="true" />}Predict
              </button>
              <button type="button" className="btn ghost" onClick={clear}><Eraser size={16} aria-hidden="true" />Clear form</button>
            </div>
            {error && <div className="err" role="alert"><CircleAlert size={18} aria-hidden="true" /><p>{error}</p></div>}
          </div>

          <aside className="result" aria-live="polite">
            <h2>Risk scores</h2>
            {!result && <p className="empty">Enter at least one reading and select Predict. Missing values are filled with the median of the 112 patients.</p>}
            {result && (
              <>
                <p className="meter-sub" style={{ marginTop: 4 }}>Score for in-hospital death, from 0 to 1. The vertical mark is the 0.5 cutoff used in the evaluation.</p>
                {result.models.map((m) => (
                  <div className="meter-row" key={m.model}>
                    <div className="meter-head">
                      <span><span className="dot" style={{ background: FAMILY_COLORS[m.family] }} />{m.model}</span>
                      <b>{fmt(m.score, 2)}</b>
                    </div>
                    <div className="meter" role="img" aria-label={`${m.model} score ${fmt(m.score, 2)}`}>
                      <i className={m.score >= 0.5 ? 'high' : ''} style={{ width: `${Math.max(1, m.score * 100)}%` }} /><u />
                    </div>
                    <div className="meter-sub">Higher than {Math.round(m.percentile * 100)}% of the 112 patients' out-of-fold scores. Typical score: {fmt(m.cohort_median_survived, 2)} for survivors, {fmt(m.cohort_median_died, 2)} for deaths.</div>
                  </div>
                ))}
                <p className="verdict"><b>{above} of {result.models.length}</b> models score this patient at or above 0.5.</p>
                <p className="meter-sub">{result.n_missing} of {result.n_features} predictors were missing and filled with cohort medians. {result.readings_used} reading{result.readings_used === 1 ? '' : 's'} used.</p>
                {result.ignored.length > 0 && (
                  <Note tone="caution">Ignored as outside physiological limits: {result.ignored.map((x) => `${x.field} ${x.value} (hour ${x.reading})`).join(', ')}.</Note>
                )}
                {example && (
                  <>
                    <Note tone="caution">Patient {example.id} ({example.outcome}) was part of the data these models were trained on, so the scores above are optimistic. The honest scores below come from models that never saw this patient.</Note>
                    <div className="table-wrap oof-table">
                      <table><thead><tr><th>Model</th><th className="num">Out-of-fold score</th></tr></thead>
                        <tbody>{Object.entries(example.oof).map(([k, v]) => <tr key={k}><td>{k}</td><td className="num">{fmt(v, 2)}</td></tr>)}</tbody></table>
                    </div>
                  </>
                )}
              </>
            )}
            <Note>These are risk scores, not percentages. Training used class weights to cope with few deaths, which pushes scores upward, so use them to compare patients, not as a chance of dying. Research prototype only; not for clinical decisions.</Note>
          </aside>
        </form>
      </div>
    </>
  );
}
