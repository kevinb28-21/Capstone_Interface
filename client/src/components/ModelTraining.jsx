import React, { useState, useEffect } from 'react';
import { api } from '../utils/api';

/**
 * ModelTraining Component
 * Displays ML model training status and information from backend
 */
export default function ModelTraining() {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchModelStatus = async () => {
    try {
      setLoading(true);
      setError(null);
      const status = await api.get('/api/ml/status');
      setModelInfo(status);
    } catch (err) {
      console.error('Failed to fetch model status:', err);
      setError(err.message || 'Failed to load model status');
      // Set default state on error
      setModelInfo({
        model_available: false,
        model_type: 'none',
        model_path: null,
        model_version: null,
        channels: 3,
        worker_config: {
          USE_MULTI_CROP_MODEL: true,
          MULTI_CROP_MODEL_DIR: './models/multi_crop',
          MULTI_CROP_MODEL_PATH: null,
          MODEL_CHANNELS: '3'
        }
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModelStatus();
  }, []);

  if (loading) {
    return (
      <div style={{ padding: 20, textAlign: 'center' }}>
        <div>Loading model information...</div>
      </div>
    );
  }

  if (error && !modelInfo) {
    return (
      <div style={{ padding: 20, textAlign: 'center', color: '#dc2626' }}>
        <div>Error loading model status: {error}</div>
      </div>
    );
  }

  const modelAvailable = modelInfo?.model_available || false;
  const modelType = modelInfo?.model_type || 'none';
  const workerConfig = modelInfo?.worker_config || {};

  return (
    <div>
      {/* Model Status */}
      <div style={{
        padding: 'var(--space-6)',
        background: modelAvailable ? 'var(--color-success-bg)' : 'var(--color-warning-bg)',
        border: `1px solid ${modelAvailable ? 'var(--color-success)' : 'var(--color-warning)'}`,
        borderRadius: 'var(--radius-2xl)',
        marginBottom: 'var(--space-6)',
        position: 'relative',
        overflow: 'hidden',
        boxShadow: var(--shadow-sm)
      }}>
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: 'var(--space-3)',
          marginBottom: 'var(--space-3)'
        }}>
          <div style={{ 
            width: 40, 
            height: 40, 
            borderRadius: 'var(--radius-full)',
            background: modelAvailable ? 'var(--color-success)' : 'var(--color-warning)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'white',
            fontSize: 20,
            boxShadow: 'var(--shadow-md)'
          }}>
            {modelAvailable ? '✓' : '⚠'}
          </div>
          <div>
            <div style={{ 
              fontWeight: 'var(--font-weight-bold)', 
              color: 'var(--color-text-primary)',
              fontFamily: 'var(--font-display)',
              fontSize: 'var(--font-size-lg)',
              display: 'flex',
              alignItems: 'center',
              gap: 8
            }}>
              Model Status: {modelAvailable ? 'System Ready' : 'Optimization Required'}
              <button 
                onClick={fetchModelStatus} 
                disabled={loading}
                title="Force sync with neural engine"
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: 14,
                  opacity: loading ? 0.5 : 0.7,
                  transition: 'opacity 0.2s',
                  display: 'flex',
                  alignItems: 'center',
                  padding: 0
                }}
              >
                {loading ? '⏳' : '🔄'}
              </button>
            </div>
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)', textTransform: 'uppercase', letterSpacing: 'var(--letter-spacing-wider)' }}>
              {modelAvailable ? 'Neural Network Operational' : 'Offline Mode Active'}
            </div>
          </div>
        </div>
        {modelAvailable ? (
          <div style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)', marginTop: 'var(--space-4)', lineHeight: 'var(--line-height-relaxed)' }}>
            <span style={{ 
              display: 'inline-block', 
              padding: '2px 8px', 
              background: 'var(--color-primary)', 
              color: 'white', 
              borderRadius: 'var(--radius-md)',
              fontSize: 'var(--font-size-xs)',
              marginRight: 8
            }}>
              {modelType === 'multi_crop' ? 'MULTI-CROP ENGINE' : 'SINGLE-CROP ENGINE'}
            </span>
            Neural network v{modelInfo?.model_version || '1.0.0'} is currently analyzing image streams.
            {modelInfo?.channels && (
              <span style={{ marginLeft: 8 }}>
                Processing <strong>{modelInfo.channels} spectral channels</strong>.
              </span>
            )}
            <div style={{ marginTop: 'var(--space-3)', fontSize: 'var(--font-size-xs)', fontStyle: 'italic', opacity: 0.8 }}>
              Using sensor-fusion logic to correlate vegetation indices with morphological health markers.
            </div>
          </div>
        ) : (
          <div style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-warning-text)', marginTop: 'var(--space-4)', lineHeight: 'var(--line-height-relaxed)' }}>
            No deep learning model detected. System is currently operating in <strong>Heuristic Fallback Mode</strong> using direct vegetation index analysis (NDVI/SAVI).
          </div>
        )}
      </div>

      {/* Model Information */}
      {modelAvailable && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: 'var(--space-4)',
          marginBottom: 'var(--space-6)'
        }}>
          <div className="metric" style={{ padding: 'var(--space-4)' }}>
            <div className="metric-label">Model Engine</div>
            <div style={{ fontSize: 'var(--font-size-base)', fontWeight: 'var(--font-weight-bold)', color: 'var(--color-primary)', textTransform: 'capitalize' }}>
              {modelType.replace('_', ' ')}
            </div>
          </div>

          {modelInfo?.model_version && (
            <div className="metric" style={{ padding: 'var(--space-4)' }}>
              <div className="metric-label">Release Version</div>
              <div style={{ fontSize: 'var(--font-size-sm)', fontWeight: 'var(--font-weight-bold)', color: 'var(--color-primary)', wordBreak: 'break-word' }}>
                {modelInfo.model_version}
              </div>
            </div>
          )}

          {modelInfo?.channels && (
            <div className="metric" style={{ padding: 'var(--space-4)' }}>
              <div className="metric-label">Neural Inputs</div>
              <div style={{ fontSize: 'var(--font-size-xl)', fontWeight: 'var(--font-weight-bold)', color: 'var(--color-primary)' }}>
                {modelInfo.channels} Channels
              </div>
            </div>
          )}
        </div>
      )}

      {/* Training Instructions */}
      <div className="card" style={{ background: 'var(--color-bg-tertiary)', border: 'none' }}>
        <div className="section-title" style={{ fontSize: 'var(--font-size-sm)' }}>
          System Configuration Guidelines
        </div>
        {!modelAvailable ? (
          <div>
            <p style={{ fontSize: 13, color: '#374151', marginBottom: 12, lineHeight: 1.6 }}>
              To enable ML-based crop health classification, train a multi-crop model using the training script:
            </p>
            <ol style={{ 
              margin: 0, 
              paddingLeft: 20, 
              color: '#374151',
              lineHeight: 1.8,
              fontSize: 13
            }}>
              <li style={{ marginBottom: 12 }}>
                <strong>Prepare training data:</strong> Organize your training images in a directory structure with crop types and health labels.
              </li>
              <li style={{ marginBottom: 12 }}>
                <strong>Train the multi-crop model:</strong>
                <pre style={{
                  marginTop: 8,
                  padding: 12,
                  background: '#1f2937',
                  color: '#f9fafb',
                  borderRadius: 6,
                  fontSize: 11,
                  overflow: 'auto'
                }}>
{`cd python_processing
python train_multi_crop_model_v2.py \\
  --data-dir ./training_data \\
  --output-dir ./models/multi_crop \\
  --epochs 50`}
                </pre>
              </li>
              <li style={{ marginBottom: 12 }}>
                <strong>Configure the worker:</strong> Set environment variables on EC2:
                <pre style={{
                  marginTop: 8,
                  padding: 12,
                  background: '#1f2937',
                  color: '#f9fafb',
                  borderRadius: 6,
                  fontSize: 11,
                  overflow: 'auto'
                }}>
{`USE_MULTI_CROP_MODEL=true
MULTI_CROP_MODEL_DIR=./models/multi_crop
MODEL_CHANNELS=3`}
                </pre>
              </li>
              <li>
                <strong>Restart the background worker</strong> to load the new model. The trained model will be saved to <code>./models/multi_crop/</code>
              </li>
            </ol>
          </div>
        ) : (
          <div>
            <p style={{ fontSize: 13, color: '#374151', marginBottom: 12, lineHeight: 1.6 }}>
              Model is available and ready for predictions. To update or retrain:
            </p>
            <ol style={{ 
              margin: 0, 
              paddingLeft: 20, 
              color: '#374151',
              lineHeight: 1.8,
              fontSize: 13
            }}>
              <li style={{ marginBottom: 12 }}>
                Follow the training steps above to create a new model
              </li>
              <li>
                Restart the background worker to load the updated model
              </li>
            </ol>
            {modelInfo?.model_path && (
              <div style={{ 
                marginTop: 12, 
                padding: 8, 
                background: '#eff6ff', 
                borderRadius: 4,
                fontSize: 12,
                color: '#1e40af'
              }}>
                <strong>Current model path:</strong> {modelInfo.model_path}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Worker Configuration Info (for debugging) */}
      {workerConfig && Object.keys(workerConfig).length > 0 && (
        <details style={{
          marginTop: 16,
          padding: 12,
          background: '#f9fafb',
          borderRadius: 6,
          border: '1px solid #e5e7eb',
          fontSize: 12
        }}>
          <summary style={{ cursor: 'pointer', fontWeight: 600, color: '#6b7280' }}>
            Worker Configuration
          </summary>
          <div style={{ marginTop: 8, fontFamily: 'monospace', fontSize: 11 }}>
            {Object.entries(workerConfig).map(([key, value]) => (
              <div key={key} style={{ marginBottom: 4 }}>
                <strong>{key}:</strong> {String(value || 'not set')}
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
