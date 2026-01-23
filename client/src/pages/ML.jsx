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
      <div className="container-grid">
        <div className="card">
          <div className="section-title">Model Training & Status</div>
          <ModelTraining />
        </div>

        <div className="card">
          <div className="section-title">Model Performance</div>
          {modelStats ? (
            <div>
              <div className="metrics" style={{ marginBottom: 20 }}>
                <div className="metric">
                  <div className="metric-label">Total Predictions</div>
                  <div className="metric-value">{modelStats.totalPredictions}</div>
                </div>
                <div className="metric">
                  <div className="metric-label">Avg Confidence</div>
                  <div className="metric-value">
                    {(modelStats.avgConfidence * 100).toFixed(1)}%
                  </div>
                </div>
                {modelStats.modelVersions.length > 0 && (
                  <div className="metric">
                    <div className="metric-label">Active Models</div>
                    <div className="metric-value">{modelStats.modelVersions.length}</div>
                  </div>
                )}
              </div>

              {Object.keys(modelStats.categoryCounts).length > 0 && (
                <div style={{ marginBottom: 20 }}>
                  <div style={{ fontWeight: 600, marginBottom: 12, color: '#111827' }}>
                    Predictions by Category
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {Object.entries(modelStats.categoryCounts).map(([category, count]) => (
                      <div key={category} style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: 12,
                        background: '#f9fafb',
                        borderRadius: 6,
                        border: '1px solid #e5e7eb'
                      }}>
                        <span style={{ textTransform: 'capitalize', fontWeight: 500 }}>
                          {category.replace('_', ' ')}
                        </span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                          <div style={{
                            width: 200,
                            height: 8,
                            background: '#e5e7eb',
                            borderRadius: 4,
                            overflow: 'hidden'
                          }}>
                            <div style={{
                              width: `${(count / modelStats.totalPredictions) * 100}%`,
                              height: '100%',
                              background: '#3b82f6',
                              transition: 'width 0.3s'
                            }} />
                          </div>
                          <span style={{ fontSize: 14, fontWeight: 600, color: '#111827', minWidth: 40 }}>
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
                  <div style={{ fontWeight: 600, marginBottom: 12, color: '#111827' }}>
                    Active Model Versions
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {modelStats.modelVersions.map(version => (
                      <div key={version} style={{
                        padding: 12,
                        background: '#eff6ff',
                        borderRadius: 6,
                        border: '1px solid #bfdbfe',
                        fontSize: 13,
                        color: '#1e40af'
                      }}>
                        {version}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div style={{ padding: 20, textAlign: 'center', color: '#6b7280' }}>
              <div style={{ fontSize: 16, marginBottom: 8 }}>No ML predictions available yet</div>
              <div style={{ fontSize: 14 }}>
                {images.length === 0 
                  ? 'Upload and process images to see ML performance statistics.'
                  : 'Processing images... Check back soon to see prediction statistics.'}
              </div>
            </div>
          )}
        </div>

        <div className="card">
          <div className="section-title">Recent ML Predictions</div>
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
                      border: selectedImageId === pred.image_id ? '2px solid #3b82f6' : '1px solid #e5e7eb'
                    }}
                  >
                    <div style={{ display: 'flex', gap: 12, alignItems: 'center', flex: 1 }}>
                      {img && (
                        <img
                          src={buildImageUrl(img) || '/placeholder.png'}
                          alt={pred.filename}
                          style={{
                            width: 56,
                            height: 56,
                            objectFit: 'cover',
                            borderRadius: 8,
                            border: '1px solid #e5e7eb'
                          }}
                          onError={(e) => {
                            e.target.style.display = 'none';
                          }}
                        />
                      )}
                      <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 600, fontSize: 14, color: '#111827' }}>
                          {pred.filename || `Image ${pred.image_id?.substring(0, 8)}`}
                        </div>
                        <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>
                          {pred.health_status && (
                            <span style={{ textTransform: 'capitalize' }}>
                              {pred.health_status.replace('_', ' ')}
                            </span>
                          )}
                          {pred.crop_type && (
                            <span style={{ marginLeft: 8, textTransform: 'capitalize' }}>
                              • {pred.crop_type.replace('_', ' ')}
                            </span>
                          )}
                          {pred.confidence && (
                            <span style={{ marginLeft: 8 }}>
                              ({(pred.confidence * 100).toFixed(0)}% confidence)
                            </span>
                          )}
                        </div>
                        {pred.processed_at && (
                          <div style={{ fontSize: 11, color: '#9ca3af', marginTop: 2 }}>
                            {new Date(pred.processed_at).toLocaleString()}
                          </div>
                        )}
                      </div>
                    </div>
                    {pred.confidence && (
                      <div style={{
                        padding: '4px 12px',
                        borderRadius: 12,
                        background: pred.confidence > 0.8 ? '#d1fae5' :
                                   pred.confidence > 0.6 ? '#fef3c7' : '#fee2e2',
                        color: pred.confidence > 0.8 ? '#065f46' :
                               pred.confidence > 0.6 ? '#92400e' : '#991b1b',
                        fontSize: 12,
                        fontWeight: 600
                      }}>
                        {(pred.confidence * 100).toFixed(0)}%
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">🤖</div>
              <div>No ML predictions yet</div>
              <div style={{ fontSize: 14, marginTop: 8, color: '#6b7280', lineHeight: 1.6 }}>
                {images.length > 0 
                  ? 'Images are being processed. The background worker analyzes uploaded images and generates predictions automatically.'
                  : 'Upload images through the Analytics page. The system will process them and display ML predictions here.'}
              </div>
            </div>
          )}
        </div>

        {selectedImage && selectedImage.analysis && (
          <div className="card">
            <div className="section-title">Selected Image Analysis</div>
            {buildImageUrl(selectedImage) && (
              <div style={{ marginBottom: 16 }}>
                <img
                  src={buildImageUrl(selectedImage)}
                  alt={selectedImage.originalName}
                  style={{
                    width: '100%',
                    borderRadius: 8,
                    border: '1px solid #e5e7eb',
                    marginBottom: 16
                  }}
                  onError={(e) => {
                    // #region agent log
                    fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ML.jsx:340',message:'Selected image load error',data:{imageId:selectedImage.id,builtUrl:buildImageUrl(selectedImage)},timestamp:Date.now(),sessionId:'debug-session',runId:'website-fix',hypothesisId:'B'})}).catch(()=>{});
                    // #endregion
                    e.target.style.display = 'none';
                  }}
                />
              </div>
            )}

            {selectedImage.analysis.confidence && (
              <div style={{
                padding: 16,
                background: '#f9fafb',
                borderRadius: 8,
                border: '1px solid #e5e7eb',
                marginBottom: 16
              }}>
                <div style={{ fontWeight: 600, marginBottom: 8, color: '#111827' }}>
                  Prediction Details
                </div>
                <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.8 }}>
                  <div>
                    <strong>Category:</strong>{' '}
                    <span style={{ textTransform: 'capitalize' }}>
                      {selectedImage.analysis.healthStatus?.replace('_', ' ')}
                    </span>
                  </div>
                  <div>
                    <strong>Confidence:</strong> {(selectedImage.analysis.confidence * 100).toFixed(1)}%
                  </div>
                  {selectedImage.analysis.modelVersion && (
                    <div>
                      <strong>Model:</strong> {selectedImage.analysis.modelVersion}
                    </div>
                  )}
                </div>
              </div>
            )}

            {selectedImage.analysis.ndvi && (
              <div style={{
                padding: 16,
                background: '#f9fafb',
                borderRadius: 8,
                border: '1px solid #e5e7eb'
              }}>
                <div style={{ fontWeight: 600, marginBottom: 8, color: '#111827' }}>
                  Feature Values
                </div>
                <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.8 }}>
                  {selectedImage.analysis.ndvi?.mean !== undefined && (
                    <div>NDVI: {selectedImage.analysis.ndvi.mean.toFixed(3)}</div>
                  )}
                  {selectedImage.analysis.savi?.mean !== undefined && (
                    <div>SAVI: {selectedImage.analysis.savi.mean.toFixed(3)}</div>
                  )}
                  {selectedImage.analysis.gndvi?.mean !== undefined && (
                    <div>GNDVI: {selectedImage.analysis.gndvi.mean.toFixed(3)}</div>
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
