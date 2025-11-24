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

const PORT = process.env.PORT || 5000;
const ORIGIN = process.env.ORIGIN || 'http://localhost:5173';
const UPLOAD_DIR = process.env.UPLOAD_DIR || path.join(__dirname, '..', 'uploads');

// Ensure upload dir exists
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const app = express();

// CORS configuration - support multiple origins and credentials
const allowedOrigins = (ORIGIN || '').split(',').map(o => o.trim()).filter(Boolean);
if (allowedOrigins.length === 0) {
  allowedOrigins.push('http://localhost:5173');
}

app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (like mobile apps or curl requests)
    if (!origin) {
      return callback(null, true);
    }
    if (allowedOrigins.includes(origin)) {
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
  res.json({ 
    status: 'ok',
    database: dbStatus ? 'connected' : 'disconnected',
    service: 'nodejs-backend'
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
    const imageId = await saveImage({
      id,
      filename: req.file.filename,
      originalName: req.file.originalname,
      filePath: s3Stored ? null : req.file.path, // Store local path for processing if not S3
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

// List images
app.get('/api/images', async (req, res) => {
  try {
    if (!dbConnected) {
      return res.status(503).json({ error: 'Database not connected' });
    }
    
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

app.listen(PORT, async () => {
  console.log(`Server listening on http://localhost:${PORT}`);
  
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



