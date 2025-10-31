import React from 'react';

export default function HomePage() {
  return (
    <div className="container">
      <div className="card">
        <div className="section-title">Welcome</div>
        <p style={{ marginTop: 6 }}>
          Use the tabs to navigate: Map for live drone telemetry, Analytics to upload
          and review crop health analyses, and ML for model integration notes.
        </p>
      </div>
      <div className="card">
        <div className="section-title">Quick Tips</div>
        <ul style={{ margin: 0, paddingLeft: 18 }}>
          <li>Upload images on the Analytics page to see placeholder NDVI and stress zones.</li>
          <li>Post telemetry to /api/telemetry to move the drone and draw routes.</li>
          <li>Replace placeholder analysis with a real model later.</li>
        </ul>
      </div>
    </div>
  );
}



