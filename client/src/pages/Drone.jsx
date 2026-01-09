import React, { useEffect, useState } from 'react';
import { api, formatDate } from '../utils/api.js';

export default function DronePage() {
  const [telemetry, setTelemetry] = useState(null);
  const [battery, setBattery] = useState(null);
  const [health, setHealth] = useState(null);
  const [schedules, setSchedules] = useState([]);
  const [flightHistory, setFlightHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    let intervalId = null;
    let isFetching = false;

    const fetchData = async () => {
      if (document.hidden || isFetching || !mounted) return;

      isFetching = true;
      try {
        // Fetch telemetry data
        const tel = await api.get('/api/telemetry').catch((e) => {
          console.error('Failed to fetch telemetry:', e);
          return null;
        });

        if (mounted && tel) {
          setTelemetry(tel);
          
          // Generate battery data from telemetry (mock for now)
          setBattery({
            level: 85, // percentage
            voltage: 11.8, // volts
            current: 2.5, // amps
            temperature: 28, // celsius
            cycles: 142,
            health: 'Good',
            estimatedTimeRemaining: 25, // minutes
            charging: false,
            lastCharged: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString() // 2 hours ago
          });

          // Generate health data (mock for now)
          setHealth({
            overall: 'Excellent',
            status: 'Operational',
            uptime: 99.8, // percentage
            lastMaintenance: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(), // 7 days ago
            nextMaintenance: new Date(Date.now() + 23 * 24 * 60 * 60 * 1000).toISOString(), // 23 days from now
            components: {
              motors: { status: 'Good', hours: 245 },
              propellers: { status: 'Good', hours: 89 },
              camera: { status: 'Excellent', hours: 312 },
              gimbal: { status: 'Good', hours: 198 },
              sensors: { status: 'Excellent', hours: 312 }
            },
            errors: [],
            warnings: []
          });

          // Generate flight history (mock for now)
          setFlightHistory([
            {
              id: 1,
              date: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
              duration: 18, // minutes
              distance: 2.4, // km
              images: 156,
              status: 'Completed',
              batteryStart: 100,
              batteryEnd: 72
            },
            {
              id: 2,
              date: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
              duration: 22,
              distance: 3.1,
              images: 203,
              status: 'Completed',
              batteryStart: 100,
              batteryEnd: 68
            },
            {
              id: 3,
              date: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
              duration: 15,
              distance: 1.8,
              images: 124,
              status: 'Completed',
              batteryStart: 95,
              batteryEnd: 75
            }
          ]);
        }
      } catch (e) {
        console.error('Error fetching drone data:', e);
      } finally {
        if (mounted) {
          setIsLoading(false);
          isFetching = false;
        }
      }
    };

    fetchData();

    const startPolling = () => {
      if (!document.hidden && !intervalId && mounted) {
        intervalId = setInterval(() => {
          if (!document.hidden && mounted) {
            fetchData();
          }
        }, 30000); // Poll every 30 seconds
      }
    };

    const stopPolling = () => {
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
    };

    startPolling();

    const handleVisibilityChange = () => {
      if (document.hidden) {
        stopPolling();
      } else {
        startPolling();
        fetchData();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      mounted = false;
      stopPolling();
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  // Mock schedules data
  useEffect(() => {
    setSchedules([
      {
        id: 1,
        name: 'Morning Field Survey',
        time: '08:00',
        date: new Date(Date.now() + 1 * 24 * 60 * 60 * 1000).toISOString(),
        duration: 20,
        area: 'North Field',
        status: 'Scheduled',
        priority: 'High'
      },
      {
        id: 2,
        name: 'Afternoon Health Check',
        time: '14:00',
        date: new Date(Date.now() + 1 * 24 * 60 * 60 * 1000).toISOString(),
        duration: 15,
        area: 'South Field',
        status: 'Scheduled',
        priority: 'Medium'
      },
      {
        id: 3,
        name: 'Weekly Full Survey',
        time: '09:00',
        date: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000).toISOString(),
        duration: 45,
        area: 'All Fields',
        status: 'Scheduled',
        priority: 'High'
      }
    ]);
  }, []);

  const getBatteryColor = (level) => {
    if (level >= 70) return 'var(--color-success)';
    if (level >= 40) return 'var(--color-warning)';
    return 'var(--color-error)';
  };

  const getHealthColor = (status) => {
    if (status === 'Excellent' || status === 'Good') return 'var(--color-success)';
    if (status === 'Fair') return 'var(--color-warning)';
    return 'var(--color-error)';
  };

  if (isLoading) {
    return (
      <div className="container">
        <div className="card">
          <div style={{ textAlign: 'center', padding: 'var(--space-12)' }}>
            <div className="animate-pulse" style={{ fontSize: '2rem', marginBottom: 'var(--space-4)' }}>
              🚁
            </div>
            <div>Loading drone information...</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="container animate-fade-in">
      <div className="container-grid">
        {/* Battery Status */}
        <div className="card card-elevated animate-fade-in-up">
          <h3 className="section-title">Battery Status</h3>
          {battery ? (
            <div>
              <div style={{ marginBottom: 'var(--space-6)' }}>
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center',
                  marginBottom: 'var(--space-2)'
                }}>
                  <span className="metric-label">Battery Level</span>
                  <span style={{ 
                    fontSize: 'var(--font-size-2xl)', 
                    fontWeight: 'var(--font-weight-bold)',
                    color: getBatteryColor(battery.level)
                  }}>
                    {battery.level}%
                  </span>
                </div>
                <div style={{
                  width: '100%',
                  height: '24px',
                  background: 'var(--color-bg-tertiary)',
                  borderRadius: 'var(--radius-full)',
                  overflow: 'hidden',
                  border: '2px solid var(--color-border)'
                }}>
                  <div style={{
                    width: `${battery.level}%`,
                    height: '100%',
                    background: `linear-gradient(90deg, ${getBatteryColor(battery.level)} 0%, ${getBatteryColor(battery.level)}dd 100%)`,
                    transition: 'width 0.3s ease',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'flex-end',
                    paddingRight: 'var(--space-2)',
                    color: 'white',
                    fontSize: 'var(--font-size-xs)',
                    fontWeight: 'var(--font-weight-semibold)'
                  }}>
                    {battery.level >= 10 && `${battery.level}%`}
                  </div>
                </div>
              </div>

              <div className="metrics">
                <div className="metric">
                  <div className="metric-label">Voltage</div>
                  <div className="metric-value" style={{ fontSize: 'var(--font-size-xl)' }}>
                    {battery.voltage}V
                  </div>
                </div>
                <div className="metric">
                  <div className="metric-label">Current</div>
                  <div className="metric-value" style={{ fontSize: 'var(--font-size-xl)' }}>
                    {battery.current}A
                  </div>
                </div>
                <div className="metric">
                  <div className="metric-label">Temperature</div>
                  <div className="metric-value" style={{ fontSize: 'var(--font-size-xl)' }}>
                    {battery.temperature}°C
                  </div>
                </div>
                <div className="metric">
                  <div className="metric-label">Cycles</div>
                  <div className="metric-value" style={{ fontSize: 'var(--font-size-xl)' }}>
                    {battery.cycles}
                  </div>
                </div>
              </div>

              <div style={{ 
                marginTop: 'var(--space-4)',
                padding: 'var(--space-4)',
                background: 'var(--color-bg-tertiary)',
                borderRadius: 'var(--radius-lg)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}>
                <div>
                  <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>
                    Estimated Flight Time
                  </div>
                  <div style={{ fontSize: 'var(--font-size-lg)', fontWeight: 'var(--font-weight-semibold)' }}>
                    {battery.estimatedTimeRemaining} min
                  </div>
                </div>
                <div style={{
                  padding: 'var(--space-2) var(--space-4)',
                  borderRadius: 'var(--radius-full)',
                  background: battery.charging ? 'var(--color-info-bg)' : 'var(--color-bg-secondary)',
                  color: battery.charging ? 'var(--color-info-text)' : 'var(--color-text-secondary)',
                  fontSize: 'var(--font-size-xs)',
                  fontWeight: 'var(--font-weight-medium)'
                }}>
                  {battery.charging ? '⚡ Charging' : '🔋 Standby'}
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">🔋</div>
              <div className="empty-state-title">No Battery Data</div>
            </div>
          )}
        </div>

        {/* Health Analytics */}
        <div className="card card-elevated animate-fade-in-up stagger-1">
          <h3 className="section-title">Health Analytics</h3>
          {health ? (
            <div>
              <div style={{ marginBottom: 'var(--space-6)' }}>
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center',
                  marginBottom: 'var(--space-4)'
                }}>
                  <div>
                    <div className="metric-label">Overall Status</div>
                    <div style={{ 
                      fontSize: 'var(--font-size-2xl)', 
                      fontWeight: 'var(--font-weight-bold)',
                      color: getHealthColor(health.overall)
                    }}>
                      {health.overall}
                    </div>
                  </div>
                  <div style={{
                    padding: 'var(--space-3) var(--space-5)',
                    borderRadius: 'var(--radius-lg)',
                    background: getHealthColor(health.status === 'Operational' ? 'Excellent' : health.status),
                    color: 'white',
                    fontSize: 'var(--font-size-sm)',
                    fontWeight: 'var(--font-weight-semibold)'
                  }}>
                    {health.status}
                  </div>
                </div>

                <div className="metrics">
                  <div className="metric">
                    <div className="metric-label">Uptime</div>
                    <div className="metric-value" style={{ fontSize: 'var(--font-size-xl)' }}>
                      {health.uptime}%
                    </div>
                  </div>
                  <div className="metric">
                    <div className="metric-label">Last Maintenance</div>
                    <div className="metric-value" style={{ fontSize: 'var(--font-size-sm)' }}>
                      {formatDate(health.lastMaintenance, 'date')}
                    </div>
                  </div>
                </div>
              </div>

              <div style={{ marginTop: 'var(--space-4)' }}>
                <div className="metric-label" style={{ marginBottom: 'var(--space-3)' }}>
                  Component Status
                </div>
                <div className="list">
                  {Object.entries(health.components).map(([name, data]) => (
                    <div key={name} className="list-item" style={{ cursor: 'default' }}>
                      <div style={{ flex: 1 }}>
                        <div style={{ 
                          fontSize: 'var(--font-size-sm)', 
                          fontWeight: 'var(--font-weight-semibold)',
                          textTransform: 'capitalize',
                          marginBottom: 'var(--space-1)'
                        }}>
                          {name}
                        </div>
                        <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>
                          {data.hours} hours
                        </div>
                      </div>
                      <span className={`badge badge-${data.status === 'Excellent' ? 'success' : data.status === 'Good' ? 'info' : 'warning'}`}>
                        {data.status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">🏥</div>
              <div className="empty-state-title">No Health Data</div>
            </div>
          )}
        </div>

        {/* Flight Schedule */}
        <div className="card card-elevated animate-fade-in-up stagger-2">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
            <h3 className="section-title" style={{ marginBottom: 0 }}>Flight Schedule</h3>
            <button className="btn btn-primary btn-sm">
              + New Schedule
            </button>
          </div>
          {schedules.length > 0 ? (
            <div className="list">
              {schedules.map((schedule) => (
                <div key={schedule.id} className="list-item">
                  <div style={{ flex: 1 }}>
                    <div style={{ 
                      fontSize: 'var(--font-size-sm)', 
                      fontWeight: 'var(--font-weight-semibold)',
                      marginBottom: 'var(--space-1)'
                    }}>
                      {schedule.name}
                    </div>
                    <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>
                      {formatDate(schedule.date, 'date')} at {schedule.time} • {schedule.duration} min • {schedule.area}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 'var(--space-2)', alignItems: 'center' }}>
                    <span className={`badge badge-${schedule.priority === 'High' ? 'error' : schedule.priority === 'Medium' ? 'warning' : 'info'}`}>
                      {schedule.priority}
                    </span>
                    <span className="badge badge-success">
                      {schedule.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">📅</div>
              <div className="empty-state-title">No Scheduled Flights</div>
              <div className="empty-state-description">
                Create a new flight schedule to get started
              </div>
            </div>
          )}
        </div>

        {/* Flight History */}
        <div className="card card-elevated animate-fade-in-up stagger-3">
          <h3 className="section-title">Recent Flight History</h3>
          {flightHistory.length > 0 ? (
            <div className="list">
              {flightHistory.map((flight) => (
                <div key={flight.id} className="list-item">
                  <div style={{ flex: 1 }}>
                    <div style={{ 
                      fontSize: 'var(--font-size-sm)', 
                      fontWeight: 'var(--font-weight-semibold)',
                      marginBottom: 'var(--space-2)'
                    }}>
                      Flight #{flight.id}
                    </div>
                    <div style={{ 
                      display: 'grid', 
                      gridTemplateColumns: 'repeat(2, 1fr)', 
                      gap: 'var(--space-2)',
                      fontSize: 'var(--font-size-xs)',
                      color: 'var(--color-text-tertiary)'
                    }}>
                      <div>📅 {formatDate(flight.date, 'date')}</div>
                      <div>⏱️ {flight.duration} min</div>
                      <div>📏 {flight.distance} km</div>
                      <div>📸 {flight.images} images</div>
                    </div>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-1)', alignItems: 'flex-end' }}>
                    <span className="badge badge-success">
                      {flight.status}
                    </span>
                    <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>
                      🔋 {flight.batteryStart}% → {flight.batteryEnd}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">📊</div>
              <div className="empty-state-title">No Flight History</div>
            </div>
          )}
        </div>

        {/* Telemetry Summary */}
        {telemetry && (
          <div className="card animate-fade-in-up stagger-4">
            <h3 className="section-title">Current Telemetry</h3>
            <div className="metrics">
              {telemetry.position && (
                <>
                  <div className="metric">
                    <div className="metric-label">Latitude</div>
                    <div className="metric-value" style={{ fontSize: 'var(--font-size-lg)' }}>
                      {telemetry.position.lat.toFixed(6)}
                    </div>
                  </div>
                  <div className="metric">
                    <div className="metric-label">Longitude</div>
                    <div className="metric-value" style={{ fontSize: 'var(--font-size-lg)' }}>
                      {telemetry.position.lng.toFixed(6)}
                    </div>
                  </div>
                  {telemetry.position.altitude && (
                    <div className="metric">
                      <div className="metric-label">Altitude</div>
                      <div className="metric-value" style={{ fontSize: 'var(--font-size-lg)' }}>
                        {telemetry.position.altitude.toFixed(1)}m
                      </div>
                    </div>
                  )}
                </>
              )}
              {telemetry.route && telemetry.route.length > 0 && (
                <div className="metric">
                  <div className="metric-label">Route Points</div>
                  <div className="metric-value" style={{ fontSize: 'var(--font-size-lg)' }}>
                    {telemetry.route.length}
                  </div>
                </div>
              )}
              {telemetry.geofence && telemetry.geofence.length > 0 && (
                <div className="metric">
                  <div className="metric-label">Geofence Points</div>
                  <div className="metric-value" style={{ fontSize: 'var(--font-size-lg)' }}>
                    {telemetry.geofence.length}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}




