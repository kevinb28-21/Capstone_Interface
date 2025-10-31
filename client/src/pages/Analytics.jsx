import React, { useEffect, useMemo, useState } from 'react';
import UploadPanel from '../components/UploadPanel.jsx';

const api = {
  listImages: async () => (await fetch('/api/images')).json()
};

export default function AnalyticsPage() {
  const [images, setImages] = useState([]);
  const [selectedImageId, setSelectedImageId] = useState(null);
  const selectedImage = useMemo(() => images.find(i => i.id === selectedImageId) || images[0], [images, selectedImageId]);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        const imgs = await api.listImages();
        if (mounted) setImages(imgs);
      } catch {}
    };
    load();
    const id = setInterval(load, 3000);
    return () => { mounted = false; clearInterval(id); };
  }, []);

  return (
    <div className="container">
      <div className="card">
        <div className="section-title">Upload & Analyses</div>
        <UploadPanel onUploaded={(item) => setImages(prev => [item, ...prev])} />
        <div style={{ height: 12 }} />
        <div className="list">
          {images.map(img => (
            <div className="list-item" key={img.id}>
              <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                <img src={img.path} alt={img.originalName} style={{ width: 48, height: 48, objectFit: 'cover', borderRadius: 8, border: '1px solid #e5e7eb' }} />
                <div>
                  <div style={{ fontWeight: 600, fontSize: 13 }}>{img.originalName || img.filename}</div>
                  <div style={{ fontSize: 12, color: '#6b7280' }}>{new Date(img.createdAt).toLocaleString()}</div>
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="badge">NDVI {img?.analysis?.ndvi ?? '-'} • {img?.analysis?.summary}</span>
                <button onClick={() => setSelectedImageId(img.id)}>View</button>
              </div>
            </div>
          ))}
          {images.length === 0 && (
            <div style={{ color: '#6b7280', fontSize: 14 }}>No images yet. Upload to start analysis.</div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="section-title">Analysis Details</div>
        {!selectedImage && <div style={{ color: '#6b7280' }}>Select an image to view details.</div>}
        {selectedImage && (
          <>
            <div className="metrics">
              <div className="metric">
                <div style={{ fontSize: 12, color: '#6b7280' }}>NDVI (mock)</div>
                <div style={{ fontSize: 20, fontWeight: 700 }}>{selectedImage.analysis?.ndvi ?? '-'}</div>
              </div>
              <div className="metric">
                <div style={{ fontSize: 12, color: '#6b7280' }}>Summary</div>
                <div style={{ fontSize: 16 }}>{selectedImage.analysis?.summary ?? '-'}</div>
              </div>
            </div>
            <div style={{ height: 12 }} />
            <div className="section-title">Stress Zones (mock 10×10 grid)</div>
            <div className="grid">
              {Array.from({ length: 100 }).map((_, idx) => {
                const x = idx % 10, y = Math.floor(idx / 10);
                const zone = selectedImage.analysis?.stressZones?.find(z => z.x === x && z.y === y);
                const color = zone ? `rgba(220,38,38,${zone.severity})` : '#e5e7eb';
                return <div key={idx} className="grid-cell" style={{ background: color }} />;
              })}
            </div>
          </>
        )}
      </div>
    </div>
  );
}



