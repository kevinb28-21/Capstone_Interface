import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { v4 as uuidv4 } from 'uuid';
import { nanoid } from 'nanoid';
import { uploadToS3, generateS3Key, isS3Enabled, getBucketName } from './s3-utils.js';
import { 
  getDbPool, 
  testConnection, 
  getAllImages, 
  getImageById, 
  saveImage, 
  getTelemetry, 
  updateTelemetry 
} from './db-utils.js';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT || 5050;
const ORIGIN = process.env.ORIGIN || 'http://localhost:5173,http://localhost:5182';
const UPLOAD_DIR = process.env.UPLOAD_DIR || path.join(__dirname, '..', 'uploads');

// Ensure upload dir exists
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const app = express();

// CORS configuration - support multiple origins and credentials
// Allow Netlify domains and configured origins
const allowedOrigins = (ORIGIN || '').split(',').map(o => o.trim()).filter(Boolean);
if (allowedOrigins.length === 0) {
  allowedOrigins.push('http://localhost:5173');
}

// Netlify domain patterns (allow all Netlify preview and production domains)
const netlifyPattern = /^https?:\/\/[\w-]+\.netlify\.app$/;
const netlifyPreviewPattern = /^https?:\/\/[\w-]+--[\w-]+\.netlify\.app$/;

app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (like mobile apps or curl requests)
    if (!origin) {
      return callback(null, true);
    }
    if (allowedOrigins.includes(origin)) {
      return callback(null, true);
    }
    
    // Allow all Netlify domains (production and preview)
    if (netlifyPattern.test(origin) || netlifyPreviewPattern.test(origin)) {
      return callback(null, true);
    }
    
    // In development, be more permissive
    if (process.env.NODE_ENV === 'development') {
      console.log(`⚠️  Allowing origin in development: ${origin}`);
      return callback(null, true);
    }
    
    // Log for debugging
    console.warn(`CORS blocked origin: ${origin}`);
    return callback(new Error(`Origin ${origin} not allowed by CORS`), false);
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

// Explicitly set credentials header
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Credentials', 'true');
  next();
});

app.use(express.json({ limit: '10mb' }));

// Multer setup
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname) || '.bin';
    const name = `${Date.now()}-${nanoid(8)}${ext}`;
    cb(null, name);
  }
});
const upload = multer({ storage });

/**
 * Note: Images saved to database with status 'uploaded' will be automatically
 * processed by the background worker (python_processing/background_worker.py).
 * No need to forward to Flask API - the worker monitors the database.
 */

// Initialize database connection on startup
let dbConnected = false;
(async () => {
  try {
    dbConnected = await testConnection();
    if (dbConnected) {
      console.log('✓ Database connected');
    } else {
      console.warn('⚠️  Database connection failed - some features may not work');
    }
  } catch (error) {
    console.error('Database initialization error:', error);
  }
})();

// Routes
app.get('/api/health', async (req, res) => {
  const dbStatus = await testConnection();
  const { hasGndviColumns } = await import('./db-utils.js');
  const hasGndvi = await hasGndviColumns();
  res.json({ 
    status: 'ok',
    database: dbStatus ? 'connected' : 'disconnected',
    service: 'nodejs-backend',
    gndviColumns: hasGndvi
  });
});

// Serve uploaded files
app.use('/uploads', express.static(UPLOAD_DIR));

// Upload image and save to database
app.post('/api/images', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image uploaded' });
  }
  
  if (!dbConnected) {
    return res.status(503).json({ error: 'Database not connected' });
  }
  
  try {
    const id = uuidv4();
    
    // Upload to S3 (or use local path if S3 disabled)
    const s3Key = generateS3Key(req.file.filename);
    const s3Url = await uploadToS3(req.file.path, s3Key, req.file.mimetype);
    
    // Use S3 URL if available, otherwise local path
    const imagePath = s3Url || req.file.path;
    const s3Stored = !!s3Url;
    
    // Parse GPS metadata if provided (from Camera X app)
    let gpsData = null;
    if (req.body.gps) {
      try {
        gpsData = typeof req.body.gps === 'string' ? JSON.parse(req.body.gps) : req.body.gps;
        console.log('GPS metadata received:', gpsData);
      } catch (e) {
        console.warn('Failed to parse GPS data:', e);
      }
    }
    
    // Save to database with status 'uploaded'
    // Background worker will automatically process it
    // Always save local file path - worker needs it even if S3 is enabled
    const imageId = await saveImage({
      id,
      filename: req.file.filename,
      originalName: req.file.originalname,
      filePath: req.file.path, // Always store local path for worker processing
      s3Url: s3Url,
      s3Key: s3Stored ? s3Key : null,
      s3Stored: s3Stored,
      fileSize: req.file.size,
      mimeType: req.file.mimetype,
      gps: gpsData
    });
    
    console.log(`✓ Image ${imageId} saved to database (status: uploaded)`);
    console.log(`  Background worker will process it automatically`);
    
    // Get the saved image record
    const fileRecord = await getImageById(imageId);
    
    if (!fileRecord) {
      return res.status(500).json({ error: 'Failed to retrieve saved image' });
    }
    
    res.json(fileRecord);
  } catch (error) {
    console.error('Error uploading image:', error);
    res.status(500).json({ error: 'Failed to process image', details: error.message });
  }
});

// List images (backward compatible - also supports new paginated endpoint)
app.get('/api/images', async (req, res) => {
  try {
    if (!dbConnected) {
      return res.status(503).json({ error: 'Database not connected' });
    }
    
    // Check if using new pagination format
    if (req.query.page || req.query.offset || req.query.status || req.query.hasAnalysis) {
      // Use enhanced pagination if available
      try {
        const { getImagesPaginated } = await import('./db-utils-enhanced.js');
        const { validatePagination } = await import('./middleware/validator.js');
        
        // Create a mock request object for validation
        const mockReq = { ...req, pagination: { 
          limit: parseInt(req.query.limit || '50', 10),
          offset: parseInt(req.query.offset || req.query.page ? (parseInt(req.query.page || '1', 10) - 1) * parseInt(req.query.limit || '50', 10) : '0', 10),
          page: parseInt(req.query.page || '1', 10)
        }};
        
        const options = {
          limit: mockReq.pagination.limit,
          offset: mockReq.pagination.offset,
          status: req.query.status,
          hasAnalysis: req.query.hasAnalysis === 'true' ? true : req.query.hasAnalysis === 'false' ? false : undefined,
          minHealthScore: req.query.minHealthScore ? parseFloat(req.query.minHealthScore) : undefined,
          maxHealthScore: req.query.maxHealthScore ? parseFloat(req.query.maxHealthScore) : undefined,
          healthStatus: req.query.healthStatus,
          sortBy: req.query.sortBy || 'uploaded_at',
          sortOrder: req.query.sortOrder || 'DESC',
          search: req.query.search
        };
        
        const result = await getImagesPaginated(options);
        return res.json(result);
      } catch (e) {
        // Fall back to old method if enhanced not available
        console.warn('Enhanced pagination not available, using legacy method');
      }
    }
    
    // Legacy method
    const limit = parseInt(req.query.limit || '100', 10);
    const images = await getAllImages(limit);
    res.json(images);
  } catch (error) {
    console.error('Error fetching images:', error);
    res.status(500).json({ error: 'Failed to fetch images', details: error.message });
  }
});

// Get one image
app.get('/api/images/:id', async (req, res) => {
  try {
    if (!dbConnected) {
      return res.status(503).json({ error: 'Database not connected' });
    }
    
    const image = await getImageById(req.params.id);
    if (!image) {
      return res.status(404).json({ error: 'Image not found' });
    }
    res.json(image);
  } catch (error) {
    console.error('Error fetching image:', error);
    res.status(500).json({ error: 'Failed to fetch image', details: error.message });
  }
});

// Telemetry
app.get('/api/telemetry', async (req, res) => {
  try {
    if (!dbConnected) {
      // Return default telemetry if DB not connected
      return res.json({
        position: { lat: 43.6532, lng: -79.3832 },
        route: [],
        geofence: [
          { lat: 43.6555, lng: -79.391 },
          { lat: 43.6505, lng: -79.391 },
          { lat: 43.6505, lng: -79.3755 },
          { lat: 43.6555, lng: -79.3755 }
        ]
      });
    }
    
    const telemetryData = await getTelemetry();
    res.json(telemetryData);
  } catch (error) {
    console.error('Error fetching telemetry:', error);
    res.status(500).json({ error: 'Failed to fetch telemetry', details: error.message });
  }
});

app.post('/api/telemetry', async (req, res) => {
  try {
    if (!dbConnected) {
      return res.status(503).json({ error: 'Database not connected' });
    }
    
    const { position, route, geofence } = req.body || {};
    
    // Validate and prepare telemetry data
    const telemetryData = {};
    
    if (position && typeof position.lat === 'number' && typeof position.lng === 'number') {
      telemetryData.position = position;
    }
    
    if (Array.isArray(route)) {
      telemetryData.route = route;
    } else if (position) {
      // Append current position to route if not provided
      const currentTelemetry = await getTelemetry();
      currentTelemetry.route.push(position);
      if (currentTelemetry.route.length > 5000) {
        currentTelemetry.route.shift();
      }
      telemetryData.route = currentTelemetry.route;
    }
    
    if (Array.isArray(geofence)) {
      telemetryData.geofence = geofence;
    }
    
    // Update database
    if (Object.keys(telemetryData).length > 0) {
      await updateTelemetry(telemetryData);
    }
    
    // Return updated telemetry
    const updatedTelemetry = await getTelemetry();
    res.json(updatedTelemetry);
  } catch (error) {
    console.error('Error updating telemetry:', error);
    res.status(500).json({ error: 'Failed to update telemetry', details: error.message });
  }
});

// ML Status endpoint - Check if model is available on EC2 worker
app.get('/api/ml/status', async (req, res) => {
  try {
    // Get worker directory (python_processing) - resolve paths robustly
    const serverDir = path.dirname(__dirname);
    const projectRoot = path.dirname(serverDir);
    const pythonProcessingDir = path.resolve(projectRoot, 'python_processing');
    const modelsBaseDir = path.resolve(pythonProcessingDir, 'models');
    
    console.log('[ML STATUS] Checking models in:', modelsBaseDir);
    console.log('[ML STATUS] Python processing dir:', pythonProcessingDir);
    
    // Check environment variables (from worker's perspective)
    const useMultiCrop = process.env.USE_MULTI_CROP_MODEL || 'true';
    const multiCropModelPath = process.env.MULTI_CROP_MODEL_PATH;
    const multiCropModelDir = process.env.MULTI_CROP_MODEL_DIR || path.resolve(modelsBaseDir, 'multi_crop');
    const singleCropModelPath = process.env.ONION_MODEL_PATH || path.resolve(modelsBaseDir, 'onion_crop_health_model.h5');
    const modelChannels = process.env.MODEL_CHANNELS || '3';
    
    let modelAvailable = false;
    let modelType = 'none';
    let modelPath = null;
    let modelVersion = null;
    
    // Check for multi-crop model first (preferred)
    if (useMultiCrop.toLowerCase() === 'true') {
      let multiCropPath = multiCropModelPath;
      
      // If path not specified, try to find latest model in directory
      if (!multiCropPath || !fs.existsSync(multiCropPath)) {
        const resolvedMultiCropDir = path.resolve(multiCropModelDir);
        console.log('[ML STATUS] Checking multi-crop directory:', resolvedMultiCropDir);
        
        if (fs.existsSync(resolvedMultiCropDir)) {
          try {
            const files = fs.readdirSync(resolvedMultiCropDir);
            const modelFiles = files.filter(f => f.endsWith('_final.h5'));
            console.log('[ML STATUS] Found model files:', modelFiles);
            
            if (modelFiles.length > 0) {
              // Get most recently modified
              const modelPaths = modelFiles.map(f => path.join(resolvedMultiCropDir, f));
              multiCropPath = modelPaths.reduce((latest, current) => {
                const latestTime = fs.statSync(latest).mtime;
                const currentTime = fs.statSync(current).mtime;
                return currentTime > latestTime ? current : latest;
              });
              console.log('[ML STATUS] Selected model:', multiCropPath);
            }
          } catch (e) {
            console.warn('[ML STATUS] Error reading multi-crop model directory:', e.message);
          }
        } else {
          console.warn('[ML STATUS] Multi-crop model directory does not exist:', resolvedMultiCropDir);
        }
      }
      
      if (multiCropPath && fs.existsSync(multiCropPath)) {
        modelAvailable = true;
        modelType = 'multi_crop';
        modelPath = multiCropPath;
        
        // Try to extract version from metadata
        try {
          const modelDir = path.dirname(multiCropPath);
          const metadataFiles = fs.readdirSync(modelDir).filter(f => f.endsWith('_metadata.json'));
          if (metadataFiles.length > 0) {
            const metadataPath = path.join(modelDir, metadataFiles[0]);
            const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
            modelVersion = metadata.model_version || metadata.training_date || path.basename(multiCropPath);
          } else {
            modelVersion = path.basename(multiCropPath).replace('_final.h5', '');
          }
        } catch (e) {
          modelVersion = path.basename(multiCropPath).replace('_final.h5', '');
        }
        console.log('[ML STATUS] Multi-crop model detected:', modelPath, 'version:', modelVersion);
      }
    }
    
    // Fallback to single-crop model
    if (!modelAvailable) {
      const resolvedSingleCropPath = path.resolve(singleCropModelPath);
      console.log('[ML STATUS] Checking single-crop model:', resolvedSingleCropPath);
      
      if (fs.existsSync(resolvedSingleCropPath)) {
        modelAvailable = true;
        modelType = 'single_crop';
        modelPath = resolvedSingleCropPath;
        modelVersion = path.basename(singleCropModelPath);
        console.log('[ML STATUS] Single-crop model detected:', modelPath);
      } else {
        console.warn('[ML STATUS] Single-crop model not found:', resolvedSingleCropPath);
      }
    }
    
    const response = {
      model_available: modelAvailable,
      model_type: modelType,
      model_path: modelPath,
      model_version: modelVersion,
      channels: parseInt(modelChannels, 10),
      worker_config: {
        USE_MULTI_CROP_MODEL: useMultiCrop.toLowerCase() === 'true',
        MULTI_CROP_MODEL_DIR: multiCropModelDir,
        MULTI_CROP_MODEL_PATH: multiCropModelPath || null,
        MODEL_CHANNELS: modelChannels,
        ONION_MODEL_PATH: singleCropModelPath
      }
    };
    
    console.log('[ML STATUS] Response:', JSON.stringify(response, null, 2));
    res.json(response);
  } catch (error) {
    console.error('[ML STATUS] Error checking model status:', error);
    res.status(500).json({ 
      error: 'Failed to check model status', 
      details: error.message,
      model_available: false,
      model_type: 'none',
      model_path: null,
      model_version: null
    });
  }
});

// ML Recent Predictions endpoint - Get recent ML predictions from database
app.get('/api/ml/recent', async (req, res) => {
  try {
    if (!dbConnected) {
      return res.status(503).json({ error: 'Database not connected' });
    }
    
    const limit = parseInt(req.query.limit || '10', 10);
    const pool = getDbPool();
    
    // Check which columns exist (backward compatible)
    const hasCropType = await columnExists('analyses', 'crop_type');
    const hasCropConfidence = await columnExists('analyses', 'crop_confidence');
    const hasHeuristicFusion = await columnExists('analyses', 'heuristic_fusion_score');
    const hasBandSchema = await columnExists('analyses', 'band_schema');
    const hasGndvi = await hasGndviColumns();
    
    // Build query with available fields
    const cropFields = hasCropType && hasCropConfidence 
      ? 'a.crop_type, a.crop_confidence,'
      : '';
    const fusionField = hasHeuristicFusion ? 'a.heuristic_fusion_score,' : '';
    const bandSchemaField = hasBandSchema ? 'a.band_schema,' : '';
    const gndviFields = hasGndvi 
      ? 'a.gndvi_mean, a.gndvi_std, a.gndvi_min, a.gndvi_max,'
      : '';
    
    const query = `
      SELECT 
        i.id as image_id,
        i.filename,
        i.original_name,
        a.processed_at,
        a.health_status,
        a.confidence,
        ${cropFields}
        a.model_version,
        a.analysis_type,
        a.health_score,
        ${fusionField}
        ${bandSchemaField}
        a.ndvi_mean,
        a.savi_mean,
        ${gndviFields}
        a.processed_s3_url,
        a.processed_image_path
      FROM analyses a
      INNER JOIN images i ON a.image_id = i.id
      WHERE a.confidence IS NOT NULL
        AND a.health_status IS NOT NULL
        AND i.processing_status = 'completed'
      ORDER BY a.processed_at DESC
      LIMIT $1
    `;
    
    const result = await pool.query(query, [limit]);
    
    const predictions = result.rows.map(row => {
      const prediction = {
        image_id: row.image_id,
        filename: row.filename || row.original_name,
        processed_at: row.processed_at?.toISOString(),
        health_status: row.health_status,
        confidence: row.confidence ? parseFloat(row.confidence) : null,
        model_version: row.model_version,
        analysis_type: row.analysis_type,
        health_score: row.health_score ? parseFloat(row.health_score) : null,
        ndvi_mean: row.ndvi_mean ? parseFloat(row.ndvi_mean) : null,
        savi_mean: row.savi_mean ? parseFloat(row.savi_mean) : null,
        processed_image_url: row.processed_s3_url || row.processed_image_path
      };
      
      if (hasCropType && hasCropConfidence) {
        prediction.crop_type = row.crop_type;
        prediction.crop_confidence = row.crop_confidence ? parseFloat(row.crop_confidence) : null;
      }
      
      if (hasHeuristicFusion) {
        prediction.heuristic_fusion_score = row.heuristic_fusion_score ? parseFloat(row.heuristic_fusion_score) : null;
      }
      
      if (hasBandSchema && row.band_schema) {
        try {
          prediction.band_schema = typeof row.band_schema === 'string' 
            ? JSON.parse(row.band_schema) 
            : row.band_schema;
        } catch (e) {
          prediction.band_schema = null;
        }
      }
      
      if (hasGndvi) {
        prediction.gndvi_mean = row.gndvi_mean ? parseFloat(row.gndvi_mean) : null;
      }
      
      return prediction;
    });
    
    res.json({
      predictions,
      count: predictions.length,
      available_fields: {
        crop_type: hasCropType && hasCropConfidence,
        heuristic_fusion_score: hasHeuristicFusion,
        band_schema: hasBandSchema,
        gndvi: hasGndvi
      }
    });
  } catch (error) {
    console.error('Error fetching recent ML predictions:', error);
    res.status(500).json({ error: 'Failed to fetch recent predictions', details: error.message });
  }
});

// Helper function to check if column exists
async function columnExists(tableName, columnName) {
  try {
    const pool = getDbPool();
    const result = await pool.query(`
      SELECT column_name 
      FROM information_schema.columns 
      WHERE table_name = $1 AND column_name = $2
    `, [tableName, columnName]);
    return result.rows.length > 0;
  } catch (error) {
    console.error(`Error checking column ${tableName}.${columnName}:`, error);
    return false;
  }
}

app.listen(PORT, '0.0.0.0', async () => {
  console.log(`Server listening on http://0.0.0.0:${PORT} (accessible from external connections)`);
  
  // Check database connection
  const dbStatus = await testConnection();
  if (dbStatus) {
    console.log('✓ Database connected');
  } else {
    console.log('⚠️  Database not connected - some features may not work');
    console.log('   Set DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD in .env');
  }
  
  // Check S3 configuration
  if (isS3Enabled()) {
    console.log(`✓ S3 storage enabled: ${getBucketName()}`);
  } else {
    console.log('⚠️  S3 storage disabled (using local storage)');
    console.log('   Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, and S3_BUCKET_NAME to enable');
  }
  
  console.log('✓ Background worker will process images automatically');
  console.log('   Make sure background_worker.py is running for image analysis');
});



