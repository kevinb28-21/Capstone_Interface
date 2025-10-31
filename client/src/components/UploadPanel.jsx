import React, { useState } from 'react';

export default function UploadPanel({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    try {
      const formData = new FormData();
      formData.append('image', file);
      const res = await fetch('/api/images', { method: 'POST', body: formData });
      if (!res.ok) throw new Error('Upload failed');
      const data = await res.json();
      onUploaded?.(data);
      setFile(null);
    } catch (err) {
      alert(err.message || 'Upload error');
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={onSubmit}>
      <div style={{ display: 'grid', gap: 8 }}>
        <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <button type="submit" disabled={!file || busy}>{busy ? 'Uploading…' : 'Upload & Analyze'}</button>
      </div>
    </form>
  );
}



