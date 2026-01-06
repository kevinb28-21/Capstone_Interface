import React, { useEffect, useMemo, useState } from 'react';
import UploadPanel from '../components/UploadPanel.jsx';
import ProcessingStatus from '../components/ProcessingStatus.jsx';
import VegetationIndexMaps from '../components/VegetationIndexMaps.jsx';
import MLExplanation from '../components/MLExplanation.jsx';
import { api, buildImageUrl, formatDate } from '../utils/api.js';

export default function AnalyticsPage() {
  const [images, setImages] = useState([]);
  const [selectedImageId, setSelectedImageId] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const selectedImage = useMemo(() => images.find(i => i.id === selectedImageId) || images[0], [images, selectedImageId]);

  useEffect(() => {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Analytics.jsx:14',message:'useEffect mounted',data:{hidden:document.hidden},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'D'})}).catch(()=>{});
    // #endregion
    
    let mounted = true;
    let intervalId = null;
    let isFetching = false;
    
    const load = async () => {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Analytics.jsx:18',message:'load called',data:{hidden:document.hidden,mounted,isFetching},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
      // #endregion
      
      // Skip if tab is hidden, already fetching, or unmounted
      if (document.hidden || isFetching || !mounted) {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Analytics.jsx:22',message:'load skipped',data:{hidden:document.hidden,isFetching,mounted},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
        // #endregion
        return;
      }
      
      isFetching = true;
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Analytics.jsx:26',message:'load starting API call',data:{mounted},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
      // #endregion
      
      try {
        const response = await api.get('/api/images').catch((e) => {
          console.error('Failed to fetch images:', e);
          return [];
        });
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Analytics.jsx:41',message:'API response received',data:{responseType:Array.isArray(response)?'array':typeof response,responseKeys:response&&!Array.isArray(response)?Object.keys(response):[],imagesCount:Array.isArray(response)?response.length:(response?.images?.length||0),firstImage:Array.isArray(response)?response[0]?{id:response[0].id,hasAnalysis:!!response[0].analysis}:null:null},timestamp:Date.now(),sessionId:'debug-session',runId:'website-fix',hypothesisId:'A'})}).catch(()=>{});
        // #endregion
        const imgs = Array.isArray(response) ? response : (response?.images || []);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Analytics.jsx:45',message:'load success',data:{imagesCount:imgs.length,mounted},timestamp:Date.now(),sessionId:'debug-session',runId:'website-fix',hypothesisId:'B'})}).catch(()=>{});
        // #endregion
        if (mounted) setImages(imgs);
      } catch (e) {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Analytics.jsx:35',message:'load error',data:{error:e?.message||'unknown',mounted},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'E'})}).catch(()=>{});
        // #endregion
        console.error('Error loading images:', e);
      } finally {
        isFetching = false;
      }
    };
    
    // Initial load
    load();
    
    // Poll every 30 seconds (reduced from 3 seconds to save Netlify bandwidth)
    const startPolling = () => {
      if (!document.hidden && !intervalId && mounted) {
        intervalId = setInterval(() => {
          if (!document.hidden && mounted) {
            load();
          }
        }, 30000);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Analytics.jsx:50',message:'interval created',data:{intervalId,intervalMs:30000},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'A'})}).catch(()=>{});
        // #endregion
      }
    };
    
    const stopPolling = () => {
      if (intervalId) {
        clearInterval(intervalId);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Analytics.jsx:58',message:'interval cleared',data:{intervalId},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'A'})}).catch(()=>{});
        // #endregion
        intervalId = null;
      }
    };
    
    // Start polling if tab is visible
    startPolling();
    
    // Refresh immediately when tab becomes visible, and manage polling
    const handleVisibilityChange = () => {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Analytics.jsx:66',message:'visibility change',data:{hidden:document.hidden},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
      // #endregion
      if (document.hidden) {
        stopPolling();
      } else {
        startPolling();
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Analytics.jsx:72',message:'visibility change - calling load',data:{},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
        // #endregion
        load();
      }
    };
    
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    return () => { 
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Analytics.jsx:80',message:'useEffect cleanup',data:{intervalId},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'F'})}).catch(()=>{});
      // #endregion
      mounted = false; 
      stopPolling();
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  const exportAnalysis = (image) => {
    if (!image || !image.analysis) return;
    
    const exportData = {
      imageId: image.id,
      filename: image.originalName || image.filename,
      uploadedAt: image.createdAt,
      processedAt: image.processedAt,
      processingStatus: image.processingStatus,
      analysis: image.analysis
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis_${image.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="container">
      <div className="container-grid">
        <div className="card">
          <div className="section-title">Upload & Image Analyses</div>
          <UploadPanel onUploaded={(item) => setImages(prev => [item, ...prev])} />
          <div style={{ marginTop: 20 }}>
            <div className="section-title" style={{ marginBottom: 12 }}>Recent Analyses</div>
            <div className="list">
              {images.map(img => (
                <div 
                  className="list-item" 
                  key={img.id}
                  onClick={() => setSelectedImageId(img.id)}
                  style={{ 
                    cursor: 'pointer',
                    border: selectedImageId === img.id ? '2px solid #3b82f6' : '1px solid #e5e7eb'
                  }}
                >
                  <div style={{ display: 'flex', gap: 12, alignItems: 'center', flex: 1 }}>
                    <img 
                      src={buildImageUrl(img) || '/placeholder.png'} 
                      alt={img.originalName} 
                      style={{ 
                        width: 56, 
                        height: 56, 
                        objectFit: 'cover', 
                        borderRadius: 8, 
                        border: '1px solid #e5e7eb' 
                      }} 
                      onError={(e) => {
                        // #region agent log
                        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Analytics.jsx:163',message:'Image load error',data:{imageId:img.id,builtUrl:buildImageUrl(img)},timestamp:Date.now(),sessionId:'debug-session',runId:'website-fix',hypothesisId:'B'})}).catch(()=>{});
                        // #endregion
                        e.target.style.display = 'none';
                      }}
                    />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 600, fontSize: 14, color: '#111827' }}>
                        {img.originalName || img.filename}
                      </div>
                      <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>
                        {formatDate(img.createdAt, 'datetime')}
                      </div>
                      {img.processingStatus && (
                        <div style={{ 
                          fontSize: 10, 
                          marginTop: 4,
                          padding: '2px 6px',
                          borderRadius: 4,
                          display: 'inline-block',
                          background: img.processingStatus === 'completed' ? '#d1fae5' :
                                     img.processingStatus === 'processing' ? '#fef3c7' :
                                     img.processingStatus === 'failed' ? '#fee2e2' : '#f3f4f6',
                          color: img.processingStatus === 'completed' ? '#065f46' :
                                 img.processingStatus === 'processing' ? '#92400e' :
                                 img.processingStatus === 'failed' ? '#991b1b' : '#6b7280'
                        }}>
                          {img.processingStatus}
                        </div>
                      )}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    {img?.analysis?.ndvi?.mean !== undefined && (
                      <span className="badge">
                        NDVI {img.analysis?.ndvi?.mean?.toFixed(2) || 'N/A'}
                      </span>
                    )}
                  </div>
                </div>
              ))}
              {images.length === 0 && (
                <div className="empty-state">
                  <div className="empty-state-icon">📸</div>
                  <div>No images analyzed yet</div>
                  <div style={{ fontSize: 14, marginTop: 8 }}>Upload images to get started</div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
            <div className="section-title">Analysis Details</div>
            {selectedImage && selectedImage.analysis && (
              <button
                onClick={() => exportAnalysis(selectedImage)}
                style={{
                  padding: '6px 12px',
                  borderRadius: 6,
                  border: '1px solid #e5e7eb',
                  background: 'white',
                  color: '#374151',
                  fontSize: 12,
                  cursor: 'pointer'
                }}
              >
                📥 Export JSON
              </button>
            )}
          </div>

          {!selectedImage && (
            <div className="empty-state">
              <div className="empty-state-icon">📊</div>
              <div>Select an image to view analysis</div>
            </div>
          )}

          {selectedImage && (
            <>
              {/* Processing Status */}
              <ProcessingStatus
                status={selectedImage.processingStatus}
                uploadedAt={selectedImage.createdAt}
                processedAt={selectedImage.processedAt}
              />

              {/* Tabs */}
              <div style={{ 
                display: 'flex', 
                gap: 8, 
                marginBottom: 20,
                borderBottom: '1px solid #e5e7eb'
              }}>
                {['overview', 'indices', 'ml', 'details'].map(tab => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    style={{
                      padding: '10px 16px',
                      border: 'none',
                      borderBottom: `2px solid ${activeTab === tab ? '#3b82f6' : 'transparent'}`,
                      background: 'transparent',
                      color: activeTab === tab ? '#3b82f6' : '#6b7280',
                      fontWeight: activeTab === tab ? 600 : 400,
                      cursor: 'pointer',
                      fontSize: 13,
                      textTransform: 'capitalize'
                    }}
                  >
                    {tab === 'indices' ? 'Vegetation Indices' : 
                     tab === 'ml' ? 'ML Analysis' : tab}
                  </button>
                ))}
              </div>

              {/* Tab Content */}
              {activeTab === 'overview' && (
                <div>
                  {selectedImage.analysis ? (
                    <>
                      <div className="metrics" style={{ marginBottom: 20 }}>
                        {selectedImage.analysis.ndvi?.mean !== undefined && (
                          <div className="metric">
                            <div className="metric-label">NDVI</div>
                            <div className="metric-value">
                              {selectedImage.analysis.ndvi.mean.toFixed(3)}
                            </div>
                          </div>
                        )}
                        {selectedImage.analysis.savi?.mean !== undefined && (
                          <div className="metric">
                            <div className="metric-label">SAVI</div>
                            <div className="metric-value">
                              {selectedImage.analysis.savi.mean.toFixed(3)}
                            </div>
                          </div>
                        )}
                        {selectedImage.analysis.gndvi?.mean !== undefined && (
                          <div className="metric">
                            <div className="metric-label">GNDVI</div>
                            <div className="metric-value">
                              {selectedImage.analysis.gndvi.mean.toFixed(3)}
                            </div>
                          </div>
                        )}
                        {selectedImage.analysis.healthStatus && (
                          <div className="metric">
                            <div className="metric-label">Health Status</div>
                            <div className="metric-value" style={{ fontSize: 16, textTransform: 'capitalize' }}>
                              {selectedImage.analysis.healthStatus.replace('_', ' ')}
                            </div>
                          </div>
                        )}
                      </div>

                      {buildImageUrl(selectedImage) && (
                        <div style={{ marginBottom: 20 }}>
                          <img 
                            src={buildImageUrl(selectedImage)} 
                            alt={selectedImage.originalName}
                            style={{ 
                              width: '100%', 
                              borderRadius: 8, 
                              border: '1px solid #e5e7eb'
                            }} 
                            onError={(e) => {
                              // #region agent log
                              fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Analytics.jsx:330',message:'Selected image load error',data:{imageId:selectedImage.id,builtUrl:buildImageUrl(selectedImage)},timestamp:Date.now(),sessionId:'debug-session',runId:'website-fix',hypothesisId:'B'})}).catch(()=>{});
                              // #endregion
                              e.target.style.display = 'none';
                            }}
                          />
                        </div>
                      )}

                      {selectedImage.analysis.summary && (
                        <div style={{
                          padding: 16,
                          background: '#f9fafb',
                          borderRadius: 8,
                          border: '1px solid #e5e7eb'
                        }}>
                          <div style={{ fontWeight: 600, marginBottom: 8, color: '#111827' }}>
                            Summary
                          </div>
                          <div style={{ color: '#374151', lineHeight: 1.6 }}>
                            {selectedImage.analysis.summary}
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <div style={{ padding: 20, textAlign: 'center', color: '#6b7280' }}>
                      Analysis not yet available. Processing...
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'indices' && selectedImage.analysis && (
                <div>
                  {buildImageUrl(selectedImage) && (
                    <VegetationIndexMaps
                      imageUrl={buildImageUrl(selectedImage)}
                      analysis={selectedImage.analysis}
                    />
                  )}
                </div>
              )}

              {activeTab === 'ml' && (
                <div>
                  {selectedImage.analysis ? (
                    <MLExplanation
                      analysis={selectedImage.analysis}
                      image={selectedImage}
                    />
                  ) : (
                    <div style={{ padding: 20, textAlign: 'center', color: '#6b7280' }}>
                      ML analysis not available. Train a model to enable ML-based classification.
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'details' && (
                <div>
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontWeight: 600, marginBottom: 8, color: '#111827' }}>
                      Processing Information
                    </div>
                    <div style={{ fontSize: 13, color: '#374151', lineHeight: 1.8 }}>
                      <div>Status: <strong>{selectedImage.processingStatus || 'uploaded'}</strong></div>
                      {selectedImage.createdAt && (
                        <div>Uploaded: {formatDate(selectedImage.createdAt, 'datetime')}</div>
                      )}
                      {selectedImage.processedAt && (
                        <div>Processed: {formatDate(selectedImage.processedAt, 'datetime')}</div>
                      )}
                      {selectedImage.analysis?.analysisType && (
                        <div>Analysis Type: {selectedImage.analysis.analysisType}</div>
                      )}
                      {selectedImage.analysis?.modelVersion && (
                        <div>Model Version: {selectedImage.analysis.modelVersion}</div>
                      )}
                    </div>
                  </div>

                  {selectedImage.analysis && (
                    <div>
                      <div style={{ fontWeight: 600, marginBottom: 8, color: '#111827' }}>
                        Full Analysis Data
                      </div>
                      <pre style={{
                        padding: 16,
                        background: '#1f2937',
                        color: '#f9fafb',
                        borderRadius: 8,
                        fontSize: 11,
                        overflow: 'auto',
                        maxHeight: '400px'
                      }}>
                        {JSON.stringify(selectedImage.analysis, null, 2)}
                      </pre>
                    </div>
                  )}

                  {selectedImage.analysis?.processedImageUrl && (
                    <div style={{ marginTop: 20 }}>
                      <div style={{ fontWeight: 600, marginBottom: 8, color: '#111827' }}>
                        Processed Image
                      </div>
                      <img
                        src={selectedImage.analysis.processedImageUrl}
                        alt="Processed"
                        style={{
                          width: '100%',
                          borderRadius: 8,
                          border: '1px solid #e5e7eb'
                        }}
                      />
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
