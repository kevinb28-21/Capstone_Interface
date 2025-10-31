import React, { useEffect, useState } from 'react';
import DashboardMap from '../components/DashboardMap.jsx';

const api = {
  getTelemetry: async () => (await fetch('/api/telemetry')).json()
};

export default function MapPage() {
  const [telemetry, setTelemetry] = useState({ position: null, route: [], geofence: [] });
  const [drawMode, setDrawMode] = useState(false);
  const [draftGeofence, setDraftGeofence] = useState([]);

  useEffect(() => {
    let mounted = true;
    const fetchTel = async () => {
      try {
        const tel = await api.getTelemetry();
        if (mounted) setTelemetry(tel);
      } catch {}
    };
    fetchTel();
    const id = setInterval(fetchTel, 3000);
    return () => { mounted = false; clearInterval(id); };
  }, []);

  return (
    <div className="container">
      <div className="card" style={{ gridColumn: '1 / -1' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div className="section-title">Live Map</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button onClick={() => setDrawMode(v => !v)} style={{ background: drawMode ? '#059669' : undefined }}>
              {drawMode ? 'Drawing: Square' : 'Draw Square Geofence'}
            </button>
            <button onClick={() => setDraftGeofence([])} disabled={draftGeofence.length === 0}>Reset</button>
            <button
              onClick={async () => {
                if (draftGeofence.length < 3) return;
                await fetch('/api/telemetry', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ geofence: draftGeofence }) });
                const tel = await api.getTelemetry();
                setTelemetry(tel);
                setDrawMode(false);
                setDraftGeofence([]);
              }}
              disabled={draftGeofence.length < 3}
            >
              Save Geofence
            </button>
          </div>
        </div>
        <div className="map-wrapper">
          <DashboardMap telemetry={telemetry} drawMode={drawMode} draftGeofence={draftGeofence} onDraftChange={setDraftGeofence} />
        </div>
      </div>
    </div>
  );
}


