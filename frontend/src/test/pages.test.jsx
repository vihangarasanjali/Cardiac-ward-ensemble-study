import { render, screen, waitFor, fireEvent, cleanup } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import App from '../App';
import fixtures from './fixtures.json';
import { clearApiCache } from '../api';

beforeEach(() => {
  clearApiCache();
  vi.stubGlobal('fetch', vi.fn((url) => Promise.resolve({ ok: url in fixtures, json: () => Promise.resolve(fixtures[url]) })));
});
afterEach(cleanup);

const open = (path) => render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>);

test('summary page shows cohort numbers and top models', async () => {
  open('/');
  await waitFor(() => expect(screen.getByText('Which ensemble method predicts death on a cardiac ward best?')).toBeTruthy());
  expect(screen.getAllByText('112').length).toBeGreaterThan(0);
  expect(screen.getAllByText('XGBoost').length).toBeGreaterThan(0);
  expect(screen.getByText(/statistically tied/)).toBeTruthy();
});

test('data overview renders cohort, cleaning and feature explorer', async () => {
  open('/data');
  await waitFor(() => expect(screen.getByText('The cohort')).toBeTruthy());
  expect(screen.getByText('93 survived')).toBeTruthy();
  expect(screen.getByText('Cleaning')).toBeTruthy();
  const select = screen.getByLabelText('Predictor');
  fireEvent.change(select, { target: { value: 'HR_mean' } });
  expect(screen.getByRole('img', { name: /Distribution of HR_mean/ })).toBeTruthy();
});

test('model comparison tabs and filters work', async () => {
  open('/models');
  await waitFor(() => expect(screen.getByText('All metrics')).toBeTruthy());
  fireEvent.change(screen.getByLabelText('Metric'), { target: { value: 'Brier' } });
  fireEvent.click(screen.getByRole('tab', { name: 'Spread across folds' }));
  expect(screen.getByRole('img', { name: /Distribution of Brier/ })).toBeTruthy();
  fireEvent.click(screen.getByRole('tab', { name: 'Against Random Forest' }));
  expect(screen.getByText('Decision Tree')).toBeTruthy();
  expect(screen.getByText(/comparisons reach p below 0.05/)).toBeTruthy();
});

test('shows a helpful message when the API is down', async () => {
  vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('down'))));
  open('/models?x=1');
  await waitFor(() => expect(screen.getByText('The results service is not reachable')).toBeTruthy());
});

test('risk predictor loads an example, predicts and shows honest out-of-fold scores', async () => {
  const example = { id: 94, outcome: 'died', admission: { Age: 61, Gender: 1, Na: 138, K: 4, Cl: 104, Urea: 40, Ceratinine: 91, Alcoholic: 0, Smoke: 1, FHCD: 0, TriageScore: 2 },
    readings: [{ SBP: 110, DBP: 70, HR: 90, RR: 20, BT: 98, SpO2: 96, GCS: 15 }], oof: { 'Random Forest': 0.57, XGBoost: 0.96, 'Stacking (RF+XGB+LGBM -> LR)': 0.89 } };
  const pred = { n_missing: 3, n_features: 46, readings_used: 1, ignored: [{ reading: 1, field: 'HR', value: 999, allowed: [20, 250] }],
    models: ['Random Forest', 'XGBoost', 'Stacking (RF+XGB+LGBM -> LR)'].map((m, i) => ({ model: m, family: ['Bagging', 'Boosting', 'Stacking'][i], score: [0.83, 1, 0.94][i], percentile: 0.9, cohort_median_survived: 0.05, cohort_median_died: 0.6 })) };
  const calls = [];
  vi.stubGlobal('fetch', vi.fn((url, opts) => {
    calls.push([url, opts?.method, opts?.body]);
    const body = url.startsWith('/api/example') ? example : pred;
    return Promise.resolve({ ok: true, json: () => Promise.resolve(body) });
  }));
  open('/predict');
  fireEvent.click(await screen.findByRole('button', { name: /Load a patient who died/ }));
  await waitFor(() => expect(screen.getByLabelText('Heart rate at hour 1').value).toBe('90'));
  fireEvent.click(screen.getByRole('button', { name: 'Predict' }));
  await waitFor(() => expect(screen.getByText('3 of 3')).toBeTruthy());
  expect(screen.queryByText('Logistic Regression')).toBeNull();
  expect(screen.getByText(/Ignored as outside physiological limits: HR 999/)).toBeTruthy();
  expect(screen.getByText(/optimistic/)).toBeTruthy();
  const post = calls.find((c) => c[1] === 'POST');
  const sent = JSON.parse(post[2]);
  expect(sent.readings).toHaveLength(1);
  expect(sent.admission.Age).toBe(61);
  expect(sent.admission.FHCD).toBe(0);
});

test('risk predictor shows the server message when prediction is rejected', async () => {
  vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ ok: false, status: 422, json: () => Promise.resolve({ detail: 'Enter at least one valid reading before predicting.' }) })));
  open('/predict');
  fireEvent.click(await screen.findByRole('button', { name: 'Predict' }));
  await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('Enter at least one valid reading'));
});
