import React, { useMemo, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, Polygon, Tooltip, useMapEvents } from 'react-leaflet';
import L from 'leaflet';

const droneIcon = new L.Icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  tooltipAnchor: [16, -28],
  shadowSize: [41, 41]
});

function ClickCapture({ drawMode, onDraftChange }) {
  const startRef = useRef(null);
  useMapEvents({
    click(e) {
      if (!drawMode) return;
      const { lat, lng } = e.latlng;
      if (!startRef.current) {
        startRef.current = { lat, lng };
        return;
      }
      const s = startRef.current;
      const dLat = lat - s.lat;
      const dLng = lng - s.lng;
      const side = Math.max(Math.abs(dLat), Math.abs(dLng));
      const lat2 = s.lat + Math.sign(dLat || 1) * side;
      const lng2 = s.lng + Math.sign(dLng || 1) * side;
      const corners = [
        { lat: s.lat, lng: s.lng },
        { lat: s.lat, lng: lng2 },
        { lat: lat2, lng: lng2 },
        { lat: lat2, lng: s.lng }
      ];
      onDraftChange?.(corners);
      startRef.current = null;
    }
  });
  return null;
}

export default function DashboardMap({ telemetry, drawMode = false, draftGeofence = [], onDraftChange }) {
  const center = useMemo(() => {
    return telemetry?.position ? [telemetry.position.lat, telemetry.position.lng] : [37.7749, -122.4194];
  }, [telemetry]);

  const routeLatLngs = (telemetry?.route || []).map(p => [p.lat, p.lng]);
  const geofenceLatLngs = (telemetry?.geofence || []).map(p => [p.lat, p.lng]);

  return (
    <MapContainer center={center} zoom={15} style={{ height: '100%', width: '100%' }}>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="&copy; OpenStreetMap contributors" />
      <ClickCapture drawMode={drawMode} onDraftChange={onDraftChange} />

      {telemetry?.position && (
        <Marker position={center} icon={droneIcon}>
          <Tooltip direction="top" offset={[0, -10]} opacity={1} permanent>
            Drone
          </Tooltip>
        </Marker>
      )}

      {routeLatLngs.length > 1 && (
        <Polyline pathOptions={{ color: '#2563eb', weight: 3 }} positions={routeLatLngs} />
      )}

      {geofenceLatLngs.length >= 3 && (
        <Polygon pathOptions={{ color: '#f59e0b', weight: 2, fillOpacity: 0.08 }} positions={geofenceLatLngs} />
      )}

      {Array.isArray(draftGeofence) && draftGeofence.length >= 3 && (
        <Polygon pathOptions={{ color: '#10b981', dashArray: '6 6', weight: 2, fillOpacity: 0.06 }} positions={draftGeofence.map(p => [p.lat, p.lng])} />
      )}
    </MapContainer>
  );
}


