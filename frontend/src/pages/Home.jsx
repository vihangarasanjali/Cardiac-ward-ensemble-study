import { Users, Layers, ShieldCheck } from 'lucide-react';
import { useApi, fmt, FAMILY_COLORS } from '../api';
import { PageHead, Note, Loading, LoadError } from '../components/ui';

export default function Home() {
  const { data, error, loading } = useApi('/api/overview');
  if (loading) return <Loading />;
  if (error) return <LoadError />;
  const { report, run, summary } = data;
  const best = summary[0];
  const top = summary.slice(0, 5);

  return (
    <>
      <PageHead title="Which ensemble method predicts death on a cardiac ward best?">
        Bagging, boosting and stacking compared on {report.patients} patients from a Sri Lankan teaching hospital, using only what is recorded in the first hours of admission.
      </PageHead>
      <div className="content">
        <section className="section">
          <div className="strip">
            <div><div className="v">{report.patients}</div><div className="l">Patients</div></div>
            <div className="death"><div className="v">{report.died}</div><div className="l">Deaths</div><div className="s">{Math.round((report.died / report.patients) * 100)}% of patients</div></div>
            <div><div className="v">{report.n_features}</div><div className="l">Predictors</div><div className="s">from the first {report.window} readings</div></div>
            <div><div className="v">{summary.length}</div><div className="l">Models compared</div></div>
            <div><div className="v">{fmt(best.AUC_mean)}</div><div className="l">Best mean AUC</div><div className="s">{best.Model}</div></div>
          </div>
        </section>

        <section className="section">
          <h2>How the comparison was done</h2>
          <p className="lede muted">Every choice below is aimed at keeping a small dataset from producing flattering numbers.</p>
          <ul className="facts">
            <li>
              <Users size={20} aria-hidden="true" />
              <div><b>One row per patient.</b><p>The raw file has {report.raw_rows.toLocaleString()} hourly rows. Each patient is summarised by their first {report.window} readings, so the same patient never appears in both the training and test sets.</p></div>
            </li>
            <li>
              <Layers size={20} aria-hidden="true" />
              <div><b>Three ensemble families and a hybrid.</b><p>Bagging (Random Forest, Extra Trees, bagged trees), boosting (XGBoost, LightGBM, CatBoost, AdaBoost, Gradient Boosting), a stacked ensemble, and a hybrid that blends a Random Forest with a neural network. Logistic regression, a single decision tree and the neural network alone serve as baselines.</p></div>
            </li>
            <li>
              <ShieldCheck size={20} aria-hidden="true" />
              <div><b>Repeated cross-validation.</b><p>{run.cv} stratified folds, with missing values filled inside each fold. Class weights handle the imbalance instead of synthetic patients.</p></div>
            </li>
          </ul>
        </section>

        <section className="section">
          <h2>Result</h2>
          <p className="lede muted">The five best models by mean AUC. Values are the mean and standard deviation across all folds.</p>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Model</th><th>Family</th><th className="num">AUC</th><th className="num">PR-AUC</th><th className="num">Recall</th></tr></thead>
              <tbody>
                {top.map((m) => (
                  <tr key={m.Model}>
                    <td>{m.Model}</td>
                    <td><span className="dot" style={{ background: FAMILY_COLORS[m.Family] }} />{m.Family}</td>
                    <td className="num">{fmt(m.AUC_mean)} ± {fmt(m.AUC_std)}</td>
                    <td className="num">{fmt(m.PR_AUC_mean)} ± {fmt(m.PR_AUC_std)}</td>
                    <td className="num">{fmt(m.Recall_mean)} ± {fmt(m.Recall_std)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Note tone="caution">The leading ensembles are statistically tied. With only {report.died} deaths, AUC moves by about 0.14 between folds, so a gap of a few hundredths does not show that one model is better. The Model comparison page shows the tests.</Note>
        </section>

        <p className="muted" style={{ marginTop: 40, fontSize: '0.88rem' }}>Research prototype built on one hospital unit. It is not a clinical decision tool.</p>
      </div>
    </>
  );
}
