import { NavLink, Outlet } from 'react-router-dom';
import { HeartPulse, LayoutDashboard, Database, ChartColumn, Stethoscope } from 'lucide-react';

const links = [
  { to: '/', label: 'Summary', icon: LayoutDashboard, end: true },
  { to: '/data', label: 'Data overview', icon: Database },
  { to: '/models', label: 'Model comparison', icon: ChartColumn },
  { to: '/predict', label: 'Risk predictor', icon: Stethoscope },
];

export default function Shell() {
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <HeartPulse size={26} aria-hidden="true" />
          <div className="brand-name">Cardiac ward<br />ensemble study</div>
        </div>
        <nav className="nav" aria-label="Pages">
          {links.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end}>
              <Icon size={18} aria-hidden="true" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <b>Source data</b><br />
          Teaching Hospital Karapitiya, Galle. Rajapaksha et al., BioMedInformatics 2024 (CC BY 4.0).
        </div>
      </aside>
      <main className="main"><Outlet /></main>
    </div>
  );
}
