-- Migration: Add crop_type support to analyses table
-- Adds crop_type and crop_confidence fields for multi-crop classification

-- Add crop_type column to analyses table
ALTER TABLE analyses 
ADD COLUMN IF NOT EXISTS crop_type VARCHAR(50),
ADD COLUMN IF NOT EXISTS crop_confidence DECIMAL(5, 3);

-- Add index for crop_type queries
CREATE INDEX IF NOT EXISTS idx_analyses_crop_type ON analyses(crop_type);

-- Add comment
COMMENT ON COLUMN analyses.crop_type IS 'Predicted crop type: cherry_tomato, onion, corn, unknown';
COMMENT ON COLUMN analyses.crop_confidence IS 'Confidence score for crop type prediction (0-1)';



