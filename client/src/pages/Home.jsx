import React, { useEffect, useState } from 'react';
import { api, buildImageUrl, formatDate } from '../utils/api.js';

export default function HomePage() {
  const [images, setImages] = useState([]);
  const [telemetry, setTelemetry] = useState(null);

  useEffect(() => {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Home.jsx:8',message:'useEffect mounted',data:{hidden:document.hidden},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'D'})}).catch(()=>{});
    // #endregion
    
    let intervalId = null;
    let isFetching = false;
    
    const fetchData = async () => {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Home.jsx:12',message:'fetchData called',data:{hidden:document.hidden,isFetching},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
      // #endregion
      
      // Skip if tab is hidden or already fetching
      if (document.hidden || isFetching) {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Home.jsx:15',message:'fetchData skipped',data:{hidden:document.hidden,isFetching},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
        // #endregion
        return;
      }
      
      isFetching = true;
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Home.jsx:19',message:'fetchData starting API calls',data:{},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
      // #endregion
      
      try {
        const [imgs, tel] = await Promise.all([
          api.get('/api/images').catch((e) => {
            console.error('Failed to fetch images:', e);
            return [];
          }),
          api.get('/api/telemetry').catch((e) => {
            console.error('Failed to fetch telemetry:', e);
            return null;
          })
        ]);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Home.jsx:34',message:'API response received',data:{imagesType:Array.isArray(imgs)?'array':typeof imgs,imagesCount:Array.isArray(imgs)?imgs.length:0,firstImage:imgs?.[0]?{id:imgs[0].id,hasPath:!!imgs[0].path,hasS3Url:!!imgs[0].s3Url,hasAnalysis:!!imgs[0].analysis}:null,telemetryType:typeof tel,hasTelemetry:!!tel},timestamp:Date.now(),sessionId:'debug-session',runId:'website-fix',hypothesisId:'A'})}).catch(()=>{});
        // #endregion
        const imagesArray = Array.isArray(imgs) ? imgs : (imgs?.images || []);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Home.jsx:48',message:'Setting state',data:{imagesCount:imagesArray.length,telemetrySet:!!tel},timestamp:Date.now(),sessionId:'debug-session',runId:'website-fix',hypothesisId:'A'})}).catch(()=>{});
        // #endregion
        setImages(imagesArray);
        setTelemetry(tel);
      } catch (e) {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Home.jsx:32',message:'fetchData error',data:{error:e?.message||'unknown'},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'E'})}).catch(()=>{});
        // #endregion
        console.error('Error fetching data:', e);
      } finally {
        isFetching = false;
      }
    };
    
    // Initial fetch
    fetchData();
    
    // Poll every 30 seconds (reduced from 5 seconds to save Netlify bandwidth)
    // Only create interval if tab is visible
    const startPolling = () => {
      if (!document.hidden && !intervalId) {
        intervalId = setInterval(() => {
          if (!document.hidden) {
            fetchData();
          }
        }, 30000);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Home.jsx:47',message:'interval created',data:{intervalId,intervalMs:30000},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'A'})}).catch(()=>{});
        // #endregion
      }
    };
    
    const stopPolling = () => {
      if (intervalId) {
        clearInterval(intervalId);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Home.jsx:54',message:'interval cleared',data:{intervalId},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'A'})}).catch(()=>{});
        // #endregion
        intervalId = null;
      }
    };
    
    // Start polling if tab is visible
    startPolling();
    
    // Refresh immediately when tab becomes visible, and manage polling
    const handleVisibilityChange = () => {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Home.jsx:62',message:'visibility change',data:{hidden:document.hidden},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
      // #endregion
      if (document.hidden) {
        stopPolling();
      } else {
        startPolling();
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Home.jsx:68',message:'visibility change - calling fetchData',data:{},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'B'})}).catch(()=>{});
        // #endregion
        fetchData();
      }
    };
    
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    return () => {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Home.jsx:75',message:'useEffect cleanup',data:{intervalId},timestamp:Date.now(),sessionId:'debug-session',runId:'post-fix',hypothesisId:'F'})}).catch(()=>{});
      // #endregion
      stopPolling();
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  // #region agent log
  React.useEffect(() => {
    fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Home.jsx:118',message:'Processing images',data:{totalImages:images.length,imagesWithStatus:images.filter(img=>img.processingStatus).length,imagesWithAnalysis:images.filter(img=>img.analysis).length},timestamp:Date.now(),sessionId:'debug-session',runId:'website-fix',hypothesisId:'C'})}).catch(()=>{});
  }, [images]);
  // #endregion
  
  const processedImages = images.filter(img => img.processingStatus === 'completed' && img.analysis);
  const avgNDVI = processedImages.length > 0
    ? (processedImages.reduce((sum, img) => sum + (img?.analysis?.ndvi?.mean || 0), 0) / processedImages.length).toFixed(3)
    : null;
  const avgSAVI = processedImages.length > 0
    ? (processedImages.reduce((sum, img) => sum + (img?.analysis?.savi?.mean || 0), 0) / processedImages.length).toFixed(3)
    : null;
  const avgGNDVI = processedImages.length > 0
    ? (processedImages.reduce((sum, img) => sum + (img?.analysis?.gndvi?.mean || 0), 0) / processedImages.length).toFixed(3)
    : null;
  const avgHealthScore = processedImages.length > 0
    ? (processedImages.reduce((sum, img) => sum + (img?.analysis?.healthScore || 0), 0) / processedImages.length).toFixed(2)
    : null;
  const avgConfidence = processedImages.filter(img => img.analysis?.confidence).length > 0
    ? (processedImages
        .filter(img => img.analysis?.confidence)
        .reduce((sum, img) => sum + (img.analysis.confidence || 0), 0) / 
       processedImages.filter(img => img.analysis?.confidence).length * 100).toFixed(1)
    : null;
  
  // Count images processed today
  const today = new Date().toDateString();
  const imagesToday = images.filter(img => {
    if (!img.createdAt) return false;
    try {
      const imgDate = new Date(img.createdAt).toDateString();
      return imgDate === today && img.processingStatus === 'completed';
    } catch (e) {
      return false;
    }
  }).length;

  return (
    <div className="container">
      <div className="container-grid">
        <div>
          <div className="card" style={{ marginBottom: 24 }}>
            <div className="section-title">Welcome</div>
            <p style={{ margin: 0, color: '#6b7280', lineHeight: 1.6 }}>
              Drone Crop Health Dashboard for analyzing field imagery and monitoring drone telemetry. 
              Upload images to get crop health analysis with NDVI metrics and stress zone detection.
            </p>
          </div>

          <div className="card">
            <div className="section-title">Quick Stats</div>
            <div className="metrics">
              <div className="metric">
                <div className="metric-label">Images Analyzed</div>
                <div className="metric-value">{processedImages.length}</div>
              </div>
              {imagesToday > 0 && (
                <div className="metric">
                  <div className="metric-label">Processed Today</div>
                  <div className="metric-value">{imagesToday}</div>
                </div>
              )}
              {avgNDVI && (
                <div className="metric">
                  <div className="metric-label">Avg NDVI</div>
                  <div className="metric-value">{avgNDVI}</div>
                </div>
              )}
              {avgSAVI && (
                <div className="metric">
                  <div className="metric-label">Avg SAVI</div>
                  <div className="metric-value">{avgSAVI}</div>
                </div>
              )}
              {avgGNDVI && (
                <div className="metric">
                  <div className="metric-label">Avg GNDVI</div>
                  <div className="metric-value">{avgGNDVI}</div>
                </div>
              )}
              {avgHealthScore && (
                <div className="metric">
                  <div className="metric-label">Avg Health Score</div>
                  <div className="metric-value">{avgHealthScore}</div>
                </div>
              )}
              {avgConfidence && (
                <div className="metric">
                  <div className="metric-label">ML Confidence</div>
                  <div className="metric-value" style={{ fontSize: 18 }}>{avgConfidence}%</div>
                </div>
              )}
            </div>
          </div>

          <div className="card">
            <div className="section-title">Getting Started</div>
            <ul style={{ margin: 0, paddingLeft: 20, color: '#6b7280', lineHeight: 1.8 }}>
              <li>Go to <strong>Analytics</strong> to upload field images for analysis</li>
              <li>View <strong>Map</strong> to see drone location and draw geofenced areas</li>
              <li>Check <strong>ML</strong> for model integration information</li>
            </ul>
          </div>
        </div>

        <div className="card">
          <div className="section-title">Recent Activity</div>
          {images.length === 0 && (
            <div className="empty-state">
              <div className="empty-state-icon">📋</div>
              <div>No recent activity</div>
              <div style={{ fontSize: 14, marginTop: 8 }}>Upload your first image to get started</div>
            </div>
          )}
          {images.length > 0 && (
            <div className="list">
              {images.slice(0, 5).map(img => (
                <div key={img.id} className="list-item" style={{ cursor: 'default' }}>
                  <div style={{ display: 'flex', gap: 12, alignItems: 'center', flex: 1 }}>
                    <img 
                      src={buildImageUrl(img) || '/placeholder.png'} 
                      alt="" 
                      style={{ 
                        width: 40, 
                        height: 40, 
                        objectFit: 'cover', 
                        borderRadius: 6, 
                        border: '1px solid #e5e7eb' 
                      }} 
                      onError={(e) => {
                        // #region agent log
                        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Home.jsx:237',message:'Image load error',data:{imageId:img.id,s3Url:img.s3Url,path:img.path,builtUrl:buildImageUrl(img)},timestamp:Date.now(),sessionId:'debug-session',runId:'website-fix',hypothesisId:'B'})}).catch(()=>{});
                        // #endregion
                        e.target.style.display = 'none';
                      }}
                      onLoad={() => {
                        // #region agent log
                        fetch('http://127.0.0.1:7242/ingest/d3c584d3-d2e8-4033-b813-a5c38caf839a',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Home.jsx:244',message:'Image loaded successfully',data:{imageId:img.id},timestamp:Date.now(),sessionId:'debug-session',runId:'website-fix',hypothesisId:'B'})}).catch(()=>{});
                        // #endregion
                      }}
                    />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 13, fontWeight: 500 }}>{img.originalName || img.filename}</div>
                      <div style={{ fontSize: 11, color: '#6b7280', marginTop: 2 }}>
                        {formatDate(img.createdAt, 'date')}
                      </div>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    {img.processingStatus && (
                      <span style={{
                        fontSize: 10,
                        padding: '2px 6px',
                        borderRadius: 4,
                        background: img.processingStatus === 'completed' ? '#d1fae5' :
                                   img.processingStatus === 'processing' ? '#fef3c7' :
                                   img.processingStatus === 'failed' ? '#fee2e2' : '#f3f4f6',
                        color: img.processingStatus === 'completed' ? '#065f46' :
                               img.processingStatus === 'processing' ? '#92400e' :
                               img.processingStatus === 'failed' ? '#991b1b' : '#6b7280',
                        fontWeight: 500
                      }}>
                        {img.processingStatus}
                      </span>
                    )}
                    {img?.analysis?.ndvi?.mean !== undefined && (
                      <span className="badge" style={{ fontSize: 11 }}>
                        NDVI {img.analysis?.ndvi?.mean?.toFixed(2) || 'N/A'}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
