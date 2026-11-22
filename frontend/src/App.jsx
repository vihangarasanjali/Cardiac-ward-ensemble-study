import { Route, Routes } from 'react-router-dom';
import Shell from './components/Shell';
import Home from './pages/Home';
import DataOverview from './pages/DataOverview';
import ModelComparison from './pages/ModelComparison';
import Predictor from './pages/Predictor';

export default function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route index element={<Home />} />
        <Route path="data" element={<DataOverview />} />
        <Route path="models" element={<ModelComparison />} />
        <Route path="predict" element={<Predictor />} />
        <Route path="*" element={<Home />} />
      </Route>
    </Routes>
  );
}
