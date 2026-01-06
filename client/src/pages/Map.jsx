import React, { useEffect, useState } from 'react';
import DashboardMap from '../components/DashboardMap.jsx';
import { api } from '../utils/api.js';

export default function MapPage() {
  const [telemetry, setTelemetry] = useState({ position: null, route: [], geofence: [] });
  const [drawMode, setDrawMode] = useState(false);
  const [draftGeofence, setDraftGeofence] = useState([]);
  const [rotation, setRotation] = useState(0);

  useEffect(() => {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Map.jsx:11',message:'useEffect mounted',data:{hidden:document.hidden},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'D'})}).catch(()=>{});
    // #endregion
    
    let mounted = true;
    let intervalId = null;
    let isFetching = false;
    
    const fetchTel = async () => {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Map.jsx:15',message:'fetchTel called',data:{hidden:document.hidden,mounted,isFetching},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
      // #endregion
      
      // Skip if tab is hidden, already fetching, or unmounted
      if (document.hidden || isFetching || !mounted) {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Map.jsx:19',message:'fetchTel skipped',data:{hidden:document.hidden,isFetching,mounted},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
        // #endregion
        return;
      }
      
      isFetching = true;
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Map.jsx:23',message:'fetchTel starting API call',data:{mounted},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
      // #endregion
      
      try {
        const tel = await api.get('/api/telemetry').catch((e) => {
          console.error('Failed to fetch telemetry:', e);
          return null;
        });
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Map.jsx:38',message:'Telemetry response received',data:{hasTelemetry:!!tel,telemetryKeys:tel?Object.keys(tel):[],hasPosition:!!tel?.position,hasRoute:Array.isArray(tel?.route),hasGeofence:Array.isArray(tel?.geofence),routeLength:tel?.route?.length||0,geofenceLength:tel?.geofence?.length||0},timestamp:Date.now(),sessionId:'debug-session',runId:'website-fix',hypothesisId:'D'})}).catch(()=>{});
        // #endregion
        const telemetryData = tel || { position: null, route: [], geofence: [] };
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Map.jsx:44',message:'Setting telemetry',data:{hasPosition:!!telemetryData.position,routeLength:telemetryData.route?.length||0,geofenceLength:telemetryData.geofence?.length||0,mounted},timestamp:Date.now(),sessionId:'debug-session',runId:'website-fix',hypothesisId:'D'})}).catch(()=>{});
        // #endregion
        if (mounted) setTelemetry(telemetryData);
      } catch (e) {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Map.jsx:32',message:'fetchTel error',data:{error:e?.message||'unknown',mounted},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'E'})}).catch(()=>{});
        // #endregion
        console.error('Error fetching telemetry:', e);
      } finally {
        isFetching = false;
      }
    };
    
    // Initial fetch
    fetchTel();
    
    // Poll every 30 seconds (reduced from 3 seconds to save Netlify bandwidth)
    const startPolling = () => {
      if (!document.hidden && !intervalId && mounted) {
        intervalId = setInterval(() => {
          if (!document.hidden && mounted) {
            fetchTel();
          }
        }, 30000);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Map.jsx:47',message:'interval created',data:{intervalId,intervalMs:30000},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'A'})}).catch(()=>{});
        // #endregion
      }
    };
    
    const stopPolling = () => {
      if (intervalId) {
        clearInterval(intervalId);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Map.jsx:55',message:'interval cleared',data:{intervalId},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'A'})}).catch(()=>{});
        // #endregion
        intervalId = null;
      }
    };
    
    // Start polling if tab is visible
    startPolling();
    
    // Refresh immediately when tab becomes visible, and manage polling
    const handleVisibilityChange = () => {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Map.jsx:63',message:'visibility change',data:{hidden:document.hidden},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
      // #endregion
      if (document.hidden) {
        stopPolling();
      } else {
        startPolling();
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Map.jsx:69',message:'visibility change - calling fetchTel',data:{},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
        // #endregion
        fetchTel();
      }
    };
    
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    return () => { 
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Map.jsx:77',message:'useEffect cleanup',data:{intervalId},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'F'})}).catch(()=>{});
      // #endregion
      mounted = false; 
      stopPolling();
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  return (
    <div className="container">
      <div className="card" style={{ gridColumn: '1 / -1' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16, flexWrap: 'wrap', gap: 12 }}>
          <div>
            <div className="section-title" style={{ marginBottom: 4 }}>Drone Telemetry Map</div>
            <div style={{ fontSize: 13, color: '#6b7280' }}>
              View drone location, route, and geofenced areas
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button 
              onClick={() => setDrawMode(v => !v)} 
              style={{ 
                background: drawMode ? '#059669' : '#4cdf20',
                color: '#152111'
              }}
            >
              {drawMode ? '✓ Drawing Mode' : 'Draw Geofence'}
            </button>
            <button 
              onClick={() => setDraftGeofence([])} 
              disabled={draftGeofence.length === 0}
              style={{ background: '#6b7280' }}
            >
              Reset
            </button>
            <button
              onClick={async () => {
                if (draftGeofence.length < 3) return;
                await api.post('/api/telemetry', { geofence: draftGeofence });
                const tel = await api.get('/api/telemetry');
                setTelemetry(tel);
                setDrawMode(false);
                setDraftGeofence([]);
              }}
              disabled={draftGeofence.length < 3}
              style={{ background: '#2563eb', color: 'white' }}
            >
              Save Geofence
            </button>
          </div>
        </div>
        
        {drawMode && (
          <div style={{ 
            padding: 12, 
            background: '#fef3c7', 
            border: '1px solid #fbbf24', 
            borderRadius: 8, 
            marginBottom: 16,
            fontSize: 14,
            color: '#92400e'
          }}>
            <strong>Drawing Mode Active:</strong> Click and drag on the map to draw a rectangular geofence area
          </div>
        )}
        
        <div className="map-wrapper" style={{ position: 'relative' }}>
          <DashboardMap 
            telemetry={telemetry} 
            drawMode={drawMode} 
            draftGeofence={draftGeofence} 
            onDraftChange={setDraftGeofence}
            rotation={rotation}
            onRotationChange={setRotation}
          />
        </div>
        
        {telemetry.position && (
          <div style={{ marginTop: 16, padding: 12, background: '#f9fafb', borderRadius: 8, fontSize: 13 }}>
            <strong>Current Position:</strong> {telemetry.position.lat.toFixed(6)}, {telemetry.position.lng.toFixed(6)}
            {telemetry.geofence.length > 0 && (
              <span style={{ marginLeft: 16 }}>
                <strong>Geofence:</strong> {telemetry.geofence.length} points defined
              </span>
            )}
            {telemetry.route.length > 0 && (
              <span style={{ marginLeft: 16 }}>
                <strong>Route:</strong> {telemetry.route.length} points
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
