import React from 'react';
import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom';
import HomePage from './pages/Home.jsx';
import MapPage from './pages/Map.jsx';
import AnalyticsPage from './pages/Analytics.jsx';
import MLPage from './pages/ML.jsx';
export default function App() {
  return (
    <BrowserRouter>
      <header>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h1>Drone Crop Health Dashboard</h1>
          <nav className="tabs">
            <NavLink to="/" end className={({ isActive }) => isActive ? 'tab active' : 'tab'}>Home</NavLink>
            <NavLink to="/map" className={({ isActive }) => isActive ? 'tab active' : 'tab'}>Map</NavLink>
            <NavLink to="/analytics" className={({ isActive }) => isActive ? 'tab active' : 'tab'}>Analytics</NavLink>
            <NavLink to="/ml" className={({ isActive }) => isActive ? 'tab active' : 'tab'}>ML</NavLink>
          </nav>
        </div>
      </header>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/map" element={<MapPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/ml" element={<MLPage />} />
      </Routes>
    </BrowserRouter>
  );
}


