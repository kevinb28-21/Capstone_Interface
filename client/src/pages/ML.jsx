import React, { useEffect, useState } from 'react';
import ModelTraining from '../components/ModelTraining.jsx';
import { api, buildImageUrl } from '../utils/api.js';

export default function MLPage() {
  const [images, setImages] = useState([]);
  const [recentPredictions, setRecentPredictions] = useState([]);
  const [selectedImageId, setSelectedImageId] = useState(null);
  const selectedImage = images.find(i => i.id === selectedImageId) || images[0];

  useEffect(() => {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ML.jsx:10',message:'useEffect mounted',data:{hidden:document.hidden,selectedImageId},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'C'})}).catch(()=>{});
    // #endregion
    
    let mounted = true;
    let intervalId = null;
    let isFetching = false;
    
    const load = async () => {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ML.jsx:16',message:'load called',data:{hidden:document.hidden,mounted,selectedImageId,isFetching},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
      // #endregion
      
      // Skip if tab is hidden, already fetching, or unmounted
      if (document.hidden || isFetching || !mounted) {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ML.jsx:20',message:'load skipped',data:{hidden:document.hidden,isFetching,mounted},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
        // #endregion
        return;
      }
      
      isFetching = true;
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ML.jsx:24',message:'load starting API call',data:{mounted,selectedImageId},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
      // #endregion
      
      try {
        // Fetch both images and recent ML predictions
        const [imagesResponse, predictionsResponse] = await Promise.all([
          api.get('/api/images').catch((e) => {
            console.error('Failed to fetch images:', e);
            return [];
          }),
          api.get('/api/ml/recent?limit=10').catch((e) => {
            console.error('Failed to fetch recent predictions:', e);
            return { predictions: [] };
          })
        ]);
        
        const imgs = Array.isArray(imagesResponse) ? imagesResponse : (imagesResponse?.images || []);
        const predictions = predictionsResponse?.predictions || [];
        
        if (mounted) {
          const filtered = imgs.filter(img => img.analysis && img.processingStatus === 'completed');
          setImages(filtered);
          setRecentPredictions(predictions);
          
          if (imgs.length > 0 && !selectedImageId) {
            const firstWithML = imgs.find(img => img.analysis?.confidence);
            if (firstWithML) setSelectedImageId(firstWithML.id);
          }
        }
      } catch (e) {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ML.jsx:38',message:'load error',data:{error:e?.message||'unknown',mounted,selectedImageId},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'E'})}).catch(()=>{});
        // #endregion
        console.error('Error loading images:', e);
      } finally {
        isFetching = false;
      }
    };
    
    // Initial load
    load();
    
    // Poll every 30 seconds (reduced from 5 seconds to save Netlify bandwidth)
    const startPolling = () => {
      if (!document.hidden && !intervalId && mounted) {
        intervalId = setInterval(() => {
          if (!document.hidden && mounted) {
            load();
          }
        }, 30000);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ML.jsx:54',message:'interval created',data:{intervalId,intervalMs:30000,selectedImageId},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'C'})}).catch(()=>{});
        // #endregion
      }
    };
    
    const stopPolling = () => {
      if (intervalId) {
        clearInterval(intervalId);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ML.jsx:62',message:'interval cleared',data:{intervalId,selectedImageId},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'C'})}).catch(()=>{});
        // #endregion
        intervalId = null;
      }
    };
    
    // Start polling if tab is visible
    startPolling();
    
    // Refresh immediately when tab becomes visible, and manage polling
    const handleVisibilityChange = () => {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ML.jsx:70',message:'visibility change',data:{hidden:document.hidden},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
      // #endregion
      if (document.hidden) {
        stopPolling();
      } else {
        startPolling();
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ML.jsx:76',message:'visibility change - calling load',data:{},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
        // #endregion
        load();
      }
    };
    
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    return () => { 
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ML.jsx:84',message:'useEffect cleanup',data:{intervalId,selectedImageId},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'C'})}).catch(()=>{});
      // #endregion
      mounted = false; 
      stopPolling();
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [selectedImageId]);

  // Get model statistics from images
  const modelStats = React.useMemo(() => {
    const mlImages = images.filter(img => img.analysis?.confidence);
    if (mlImages.length === 0) return null;

    const avgConfidence = mlImages.reduce((sum, img) => sum + (img.analysis.confidence || 0), 0) / mlImages.length;
    const modelVersions = [...new Set(mlImages.map(img => img.analysis.modelVersion).filter(Boolean))];
    
    // Count predictions by category
    const categoryCounts = {};
    mlImages.forEach(img => {
      const category = img.analysis?.healthStatus;
      if (category) {
        categoryCounts[category] = (categoryCounts[category] || 0) + 1;
      }
    });

    return {
      totalPredictions: mlImages.length,
      avgConfidence,
      modelVersions,
      categoryCounts
    };
  }, [images]);

  return (
    <div className="container">
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 'var(--space-4)' }}>
        <button 
          className="btn btn-secondary btn-sm" 
          onClick={load}
          disabled={isFetching}
          style={{ textTransform: 'none', gap: 'var(--space-2)' }}
        >
          {isFetching ? '⚡ Syncing...' : '🔄 Refresh Intelligence Data'}
        </button>
      </div>
      <div className="container-grid">
        <div className="card">
          <div className="section-title">Model Training & Status</div>
          <ModelTraining />
        </div>

        <div className="card">
          <div className="section-title">Model Performance Metrics</div>
          {modelStats ? (
            <div>
              <div className="metrics" style={{ marginBottom: 'var(--space-6)' }}>
                <div className="metric">
                  <div className="metric-label">Analytic Predictions</div>
                  <div className="metric-value" style={{ fontSize: 'var(--font-size-2xl)' }}>{modelStats.totalPredictions}</div>
                </div>
                <div className="metric">
                  <div className="metric-label">System Confidence</div>
                  <div className="metric-value" style={{ fontSize: 'var(--font-size-2xl)' }}>
                    {(modelStats.avgConfidence * 100).toFixed(1)}%
                  </div>
                </div>
                {modelStats.modelVersions.length > 0 && (
                  <div className="metric">
                    <div className="metric-label">Deployed Engines</div>
                    <div className="metric-value" style={{ fontSize: 'var(--font-size-2xl)' }}>{modelStats.modelVersions.length}</div>
                  </div>
                )}
              </div>

              {Object.keys(modelStats.categoryCounts).length > 0 && (
                <div style={{ marginBottom: 'var(--space-6)' }}>
                  <div className="metric-label" style={{ marginBottom: 'var(--space-3)' }}>
                    Classification Distribution
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
                    {Object.entries(modelStats.categoryCounts).map(([category, count]) => (
                      <div key={category} style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: 'var(--space-3) var(--space-4)',
                        background: 'var(--color-bg-tertiary)',
                        borderRadius: 'var(--radius-lg)',
                        border: '1px solid var(--color-border)'
                      }}>
                        <span style={{ textTransform: 'capitalize', fontWeight: 'var(--font-weight-semibold)', fontSize: 'var(--font-size-sm)' }}>
                          {category.replace('_', ' ')}
                        </span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)' }}>
                          <div style={{
                            width: 150,
                            height: 6,
                            background: 'var(--color-border-dark)',
                            borderRadius: 'var(--radius-full)',
                            overflow: 'hidden'
                          }}>
                            <div style={{
                              width: `${(count / modelStats.totalPredictions) * 100}%`,
                              height: '100%',
                              background: 'var(--color-primary)',
                              transition: 'width 0.5s cubic-bezier(0.4, 0, 0.2, 1)'
                            }} />
                          </div>
                          <span style={{ fontSize: 'var(--font-size-sm)', fontWeight: 'var(--font-weight-bold)', color: 'var(--color-primary)', minWidth: 24 }}>
                            {count}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {modelStats.modelVersions.length > 0 && (
                <div>
                  <div className="metric-label" style={{ marginBottom: 'var(--space-3)' }}>
                    Active Network Architectures
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
                    {modelStats.modelVersions.map(version => (
                      <div key={version} style={{
                        padding: 'var(--space-3) var(--space-4)',
                        background: 'var(--color-primary-overlay, rgba(10, 93, 44, 0.05))',
                        borderRadius: 'var(--radius-lg)',
                        border: '1px solid var(--color-primary-light)',
                        fontSize: 'var(--font-size-xs)',
                        color: 'var(--color-primary)',
                        fontFamily: 'var(--font-mono)',
                        fontWeight: 'var(--font-weight-medium)'
                      }}>
                        {version}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">📊</div>
              <div className="empty-state-title">Awaiting Analytic Data</div>
              <div className="empty-state-description">
                {images.length === 0 
                  ? 'Initiate field surveys and upload multispectral imagery to generate model performance metrics.'
                  : 'Neural engine is currently processing imagery. Analytic statistics will populate upon completion.'}
              </div>
            </div>
          )}
        </div>

        <div className="card">
          <div className="section-title">Recent Intelligence Predictions</div>
          {recentPredictions.length > 0 ? (
            <div className="list">
              {recentPredictions.map((pred, idx) => {
                // Find corresponding image for display
                const img = images.find(i => i.id === pred.image_id);
                return (
                  <div
                    key={pred.image_id || idx}
                    className="list-item"
                    onClick={() => img && setSelectedImageId(img.id)}
                    style={{
                      cursor: img ? 'pointer' : 'default',
                      border: selectedImageId === pred.image_id ? '2px solid var(--color-primary)' : '1px solid var(--color-border)',
                      background: selectedImageId === pred.image_id ? 'var(--color-bg-hover)' : 'var(--color-bg-secondary)',
                      transform: selectedImageId === pred.image_id ? 'translateX(8px)' : 'none'
                    }}
                  >
                    <div style={{ display: 'flex', gap: 'var(--space-4)', alignItems: 'center', flex: 1 }}>
                      {img && (
                        <div style={{ position: 'relative' }}>
                          <img
                            src={buildImageUrl(img) || '/placeholder.png'}
                            alt={pred.filename}
                            style={{
                              width: 64,
                              height: 64,
                              objectFit: 'cover',
                              borderRadius: 'var(--radius-lg)',
                              border: '1px solid var(--color-border)'
                            }}
                            onError={(e) => {
                              e.target.style.display = 'none';
                            }}
                          />
                          <div style={{
                            position: 'absolute',
                            bottom: -4,
                            right: -4,
                            width: 20,
                            height: 20,
                            borderRadius: '50%',
                            background: 'var(--color-primary)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            border: '2px solid white'
                          }}>
                            <span style={{ fontSize: 10, color: 'white' }}>🤖</span>
                          </div>
                        </div>
                      )}
                      <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 'var(--font-weight-bold)', fontSize: 'var(--font-size-sm)', color: 'var(--color-text-primary)' }}>
                          {pred.filename || `Image ${pred.image_id?.substring(0, 8)}`}
                        </div>
                        <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)', marginTop: 'var(--space-1)', display: 'flex', gap: 'var(--space-2)', alignItems: 'center' }}>
                          {pred.health_status && (
                            <span className="badge badge-neutral" style={{ fontSize: 10, padding: '0 6px', textTransform: 'uppercase' }}>
                              {pred.health_status.replace('_', ' ')}
                            </span>
                          )}
                          {pred.crop_type && (
                            <span>• {pred.crop_type.replace('_', ' ')}</span>
                          )}
                        </div>
                        {pred.processed_at && (
                          <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)', marginTop: 'var(--space-1)', opacity: 0.7 }}>
                            {new Date(pred.processed_at).toLocaleString()}
                          </div>
                        )}
                      </div>
                    </div>
                    {pred.confidence && (
                      <div style={{
                        padding: '4px 10px',
                        borderRadius: 'var(--radius-full)',
                        background: pred.confidence > 0.8 ? 'var(--color-success-bg)' :
                                   pred.confidence > 0.6 ? 'var(--color-warning-bg)' : 'var(--color-error-bg)',
                        color: pred.confidence > 0.8 ? 'var(--color-success-text)' :
                               pred.confidence > 0.6 ? 'var(--color-warning-text)' : 'var(--color-error-text)',
                        fontSize: 11,
                        fontWeight: 'var(--font-weight-bold)',
                        border: '1px solid currentColor',
                        opacity: 0.9
                      }}>
                        {(pred.confidence * 100).toFixed(0)}% CONF
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">🤖</div>
              <div className="empty-state-title">No Intelligence Predictions</div>
              <div className="empty-state-description">
                {images.length > 0 
                  ? 'Neural network is processing the current imagery queue. Predictions will appear here in real-time.'
                  : 'Establish a telemetry link and upload imagery to initiate deep-learning classification.'}
              </div>
            </div>
          )}
        </div>

        {selectedImage && selectedImage.analysis && (
          <div className="card" style={{ border: '2px solid var(--color-primary-glow, #00ff88)' }}>
            <div className="section-title">
              <span style={{ color: 'var(--color-primary)' }}>Detailed Intelligence Report</span>
            </div>
            {buildImageUrl(selectedImage) && (
              <div style={{ marginBottom: 'var(--space-6)', position: 'relative' }}>
                <img
                  src={buildImageUrl(selectedImage)}
                  alt={selectedImage.originalName}
                  style={{
                    width: '100%',
                    borderRadius: 'var(--radius-xl)',
                    border: '1px solid var(--color-border)',
                    boxShadow: 'var(--shadow-lg)'
                  }}
                  onError={(e) => {
                    e.target.style.display = 'none';
                  }}
                />
                <div style={{
                  position: 'absolute',
                  top: 'var(--space-4)',
                  right: 'var(--space-4)',
                  background: 'rgba(10, 31, 18, 0.7)',
                  color: 'white',
                  padding: '4px 12px',
                  borderRadius: 'var(--radius-full)',
                  fontSize: 10,
                  backdropFilter: 'blur(4px)',
                  fontWeight: 'var(--font-weight-bold)',
                  textTransform: 'uppercase',
                  letterSpacing: 'var(--letter-spacing-widest)'
                }}>
                  Live Analysis
                </div>
              </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
              {selectedImage.analysis.confidence && (
                <div className="metric" style={{ background: 'var(--color-bg-tertiary)', border: 'none' }}>
                  <div className="metric-label">Neural Confidence</div>
                  <div className="metric-value" style={{ fontSize: 'var(--font-size-xl)' }}>
                    {(selectedImage.analysis.confidence * 100).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)', marginTop: 4 }}>
                    Model: {selectedImage.analysis.modelVersion || 'Production v1'}
                  </div>
                </div>
              )}

              <div className="metric" style={{ background: 'var(--color-bg-tertiary)', border: 'none' }}>
                <div className="metric-label">Health Classification</div>
                <div className="metric-value" style={{ fontSize: 'var(--font-size-lg)', textTransform: 'capitalize' }}>
                  {selectedImage.analysis.healthStatus?.replace('_', ' ') || 'Calculating...'}
                </div>
                <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)', marginTop: 4 }}>
                  Engine: Heuristic-Fusion
                </div>
              </div>
            </div>

            {selectedImage.analysis.ndvi && (
              <div className="card" style={{ background: 'var(--color-bg-tertiary)', border: 'none', padding: 'var(--space-4)' }}>
                <div className="metric-label" style={{ marginBottom: 'var(--space-3)' }}>
                  Spectral Indices (Mean Values)
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 'var(--space-4)' }}>
                  {selectedImage.analysis.ndvi?.mean !== undefined && (
                    <div style={{ textAlign: 'center', flex: 1 }}>
                      <div style={{ fontSize: 'var(--font-size-lg)', fontWeight: 'var(--font-weight-bold)', color: 'var(--color-primary)' }}>
                        {selectedImage.analysis.ndvi.mean.toFixed(3)}
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)', textTransform: 'uppercase' }}>NDVI</div>
                    </div>
                  )}
                  {selectedImage.analysis.savi?.mean !== undefined && (
                    <div style={{ textAlign: 'center', flex: 1 }}>
                      <div style={{ fontSize: 'var(--font-size-lg)', fontWeight: 'var(--font-weight-bold)', color: 'var(--color-primary)' }}>
                        {selectedImage.analysis.savi.mean.toFixed(3)}
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)', textTransform: 'uppercase' }}>SAVI</div>
                    </div>
                  )}
                  {selectedImage.analysis.gndvi?.mean !== undefined && (
                    <div style={{ textAlign: 'center', flex: 1 }}>
                      <div style={{ fontSize: 'var(--font-size-lg)', fontWeight: 'var(--font-weight-bold)', color: 'var(--color-primary)' }}>
                        {selectedImage.analysis.gndvi.mean.toFixed(3)}
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)', textTransform: 'uppercase' }}>GNDVI</div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
