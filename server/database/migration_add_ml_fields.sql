-- Migration: Add ML inference and fusion fields to analyses table
-- Adds fields for model versioning, inference timing, band schema, top-k predictions, and fusion scores

-- Add new columns
ALTER TABLE analyses 
ADD COLUMN IF NOT EXISTS model_version VARCHAR(100),
ADD COLUMN IF NOT EXISTS inference_time_ms INTEGER,
ADD COLUMN IF NOT EXISTS band_schema JSONB,
ADD COLUMN IF NOT EXISTS health_topk JSONB,
ADD COLUMN IF NOT EXISTS crop_topk JSONB,
ADD COLUMN IF NOT EXISTS heuristic_fusion_score DECIMAL(5, 3),
ADD COLUMN IF NOT EXISTS fallback_reason VARCHAR(255);

-- Add indexes
CREATE INDEX IF NOT EXISTS idx_analyses_model_version ON analyses(model_version);
CREATE INDEX IF NOT EXISTS idx_analyses_band_schema ON analyses USING GIN(band_schema);
CREATE INDEX IF NOT EXISTS idx_analyses_fallback_reason ON analyses(fallback_reason);

-- Add comments
COMMENT ON COLUMN analyses.model_version IS 'ML model version identifier (e.g., multi_crop_model_20240101_120000)';
COMMENT ON COLUMN analyses.inference_time_ms IS 'Model inference time in milliseconds';
COMMENT ON COLUMN analyses.band_schema IS 'JSON object with band information: {"bands": ["R","G","B","NIR"], "band_count": 4, "domain": "uav"}';
COMMENT ON COLUMN analyses.health_topk IS 'JSON array of top-k health predictions: [{"class": "healthy", "confidence": 0.95}, ...]';
COMMENT ON COLUMN analyses.crop_topk IS 'JSON array of top-k crop predictions: [{"class": "onion", "confidence": 0.92}, ...]';
COMMENT ON COLUMN analyses.heuristic_fusion_score IS 'Optional heuristic health score combining ML prediction with NDVI (0-1). Not used in model inference.';
COMMENT ON COLUMN analyses.fallback_reason IS 'Reason for RGB fallback path (e.g., band schema validation failed, missing NIR).';
COMMENT ON COLUMN analyses.fallback_reason IS 'Reason for RGB fallback path (e.g., band schema validation failed, missing NIR).';

