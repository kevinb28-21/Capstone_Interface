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

  useEffect(() => {
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
        padding: 16,
        background: modelAvailable ? '#d1fae5' : '#fef3c7',
        border: `1px solid ${modelAvailable ? '#059669' : '#f59e0b'}`,
        borderRadius: 8,
        marginBottom: 20
      }}>
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: 8,
          marginBottom: 8
        }}>
          <span style={{ fontSize: 20 }}>
            {modelAvailable ? '✓' : '⚠'}
          </span>
          <div style={{ fontWeight: 600, color: '#111827' }}>
            Model Status: {modelAvailable ? 'Available' : 'Not Available'}
          </div>
        </div>
        {modelAvailable ? (
          <div style={{ fontSize: 13, color: '#065f46', marginTop: 8, lineHeight: 1.6 }}>
            {modelType === 'multi_crop' ? 'Multi-crop model' : 'Single-crop model'} ready for inference
            {modelInfo?.model_version && ` (v${modelInfo.model_version})`}
            {modelInfo?.channels && ` • ${modelInfo.channels}-channel input`}
            <div style={{ marginTop: 4, fontSize: 12, opacity: 0.9 }}>
              The model analyzes crop health using vegetation indices (NDVI, SAVI, GNDVI)
            </div>
          </div>
        ) : (
          <div style={{ fontSize: 13, color: '#92400e', marginTop: 8, lineHeight: 1.6 }}>
            No trained model detected. The system uses vegetation index analysis for crop health assessment.
          </div>
        )}
      </div>

      {/* Model Information */}
      {modelAvailable && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 12,
          marginBottom: 20
        }}>
          <div style={{
            padding: 16,
            background: '#f9fafb',
            borderRadius: 8,
            border: '1px solid #e5e7eb'
          }}>
            <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 4 }}>
              Model Type
            </div>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#111827', textTransform: 'capitalize' }}>
              {modelType.replace('_', ' ')}
            </div>
          </div>

          {modelInfo?.model_version && (
            <div style={{
              padding: 16,
              background: '#f9fafb',
              borderRadius: 8,
              border: '1px solid #e5e7eb'
            }}>
              <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 4 }}>
                Model Version
              </div>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#111827', wordBreak: 'break-word' }}>
                {modelInfo.model_version}
              </div>
            </div>
          )}

          {modelInfo?.channels && (
            <div style={{
              padding: 16,
              background: '#f9fafb',
              borderRadius: 8,
              border: '1px solid #e5e7eb'
            }}>
              <div style={{ fontSize: 11, color: '#6b7280', marginBottom: 4 }}>
                Channels
              </div>
              <div style={{ fontSize: 18, fontWeight: 600, color: '#111827' }}>
                {modelInfo.channels}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Training Instructions */}
      <div style={{
        padding: 16,
        background: '#f9fafb',
        borderRadius: 8,
        border: '1px solid #e5e7eb'
      }}>
        <div style={{ fontWeight: 600, marginBottom: 12, color: '#111827' }}>
          How to Train the Model
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
