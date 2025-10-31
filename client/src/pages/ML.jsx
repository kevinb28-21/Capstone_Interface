import React from 'react';

export default function MLPage() {
  return (
    <div className="container">
      <div className="card" style={{ gridColumn: '1 / -1' }}>
        <div className="section-title">Model Integration (Placeholder)</div>
        <p style={{ marginTop: 6 }}>
          This page will host model configuration and inference controls once a dataset
          and trained model are available. For now, image analysis uses placeholder NDVI
          and stress zones generated on upload.
        </p>
        <ul style={{ margin: 0, paddingLeft: 18 }}>
          <li>Planned: upload model files or endpoint configuration</li>
          <li>Planned: inference job queue and status</li>
          <li>Planned: result overlays on map and analytics comparison</li>
        </ul>
      </div>
    </div>
  );
}



