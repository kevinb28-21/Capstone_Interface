/**
 * Enhanced Express Server
 * Production-ready server with improved error handling, validation, and routing
 */
import express from 'express';
import cors from 'cors';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { testConnection, hasGndviColumns } from './db-utils.js';
import { isS3Enabled, getBucketName } from './s3-utils.js';
import { requestLogger } from './middleware/logger.js';
import { errorHandler, notFoundHandler } from './middleware/errorHandler.js';
import imagesRouter from './routes/images.js';
import telemetryRouter from './routes/telemetry.js';
import statisticsRouter from './routes/statistics.js';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT || 5050;
const ORIGIN = process.env.ORIGIN || 'http://localhost:5173,http://localhost:5182';
const UPLOAD_DIR = process.env.UPLOAD_DIR || path.join(__dirname, '..', 'uploads');

// Ensure upload dir exists
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const app = express();

// ============================================
// Middleware
// ============================================

// CORS configuration - allow Netlify domains and configured origins
const allowedOrigins = (ORIGIN || '').split(',').map(o => o.trim()).filter(Boolean);
if (allowedOrigins.length === 0) {
  allowedOrigins.push('http://localhost:5173');
}

// Netlify domain patterns (allow all Netlify preview and production domains)
const netlifyPattern = /^https?:\/\/[\w-]+\.netlify\.app$/;
const netlifyPreviewPattern = /^https?:\/\/[\w-]+--[\w-]+\.netlify\.app$/;

app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (like mobile apps, Postman, or server-to-server)
    if (!origin) {
      return callback(null, true);
    }
    
    // Check if origin is in allowed list
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
    
    // Log blocked origin for debugging
    console.warn(`CORS blocked origin: ${origin}`);
    return callback(new Error(`Origin ${origin} not allowed by CORS`), false);
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));

app.use((req, res, next) => {
  res.header('Access-Control-Allow-Credentials', 'true');
  next();
});

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Request logging
if (process.env.NODE_ENV !== 'test') {
  app.use(requestLogger);
}

// ============================================
// Health Check
// ============================================

app.get('/api/health', async (req, res) => {
  try {
    const dbStatus = await testConnection();
    const hasGndvi = await hasGndviColumns();
    
    res.json({ 
      status: 'ok',
      database: dbStatus ? 'connected' : 'disconnected',
      service: 'nodejs-backend',
      version: '2.0.0',
      gndviColumns: hasGndvi,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(503).json({
      status: 'error',
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// ============================================
// Static Files
// ============================================

app.use('/uploads', express.static(UPLOAD_DIR));

// ============================================
// API Routes
// ============================================

// Images routes (with pagination, filtering, etc.)
app.use('/api/images', imagesRouter);

// Telemetry routes
app.use('/api/telemetry', telemetryRouter);

// Statistics routes
app.use('/api/statistics', statisticsRouter);

// ============================================
// Error Handling
// ============================================

// 404 handler
app.use(notFoundHandler);

// Error handler (must be last)
app.use(errorHandler);

// ============================================
// Server Startup
// ============================================

app.listen(PORT, async () => {
  console.log('='.repeat(60));
  console.log('🚀 Enhanced Drone Crop Health API Server');
  console.log('='.repeat(60));
  console.log(`📍 Server listening on http://localhost:${PORT}`);
  console.log(`📝 Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log('');
  
  // Check database connection
  try {
    const dbStatus = await testConnection();
    if (dbStatus) {
      console.log('✓ Database connected');
      const hasGndvi = await hasGndviColumns();
      if (hasGndvi) {
        console.log('✓ GNDVI columns available');
      }
    } else {
      console.log('⚠️  Database not connected - some features may not work');
      console.log('   Set DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD in .env');
    }
  } catch (error) {
    console.error('✗ Database connection error:', error.message);
  }
  
  // Check S3 configuration
  if (isS3Enabled()) {
    console.log(`✓ S3 storage enabled: ${getBucketName()}`);
  } else {
    console.log('⚠️  S3 storage disabled (using local storage)');
    console.log('   Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, and S3_BUCKET_NAME to enable');
  }
  
  console.log('');
  console.log('📚 API Endpoints:');
  console.log('   GET    /api/health              - Health check');
  console.log('   GET    /api/images              - List images (with pagination & filtering)');
  console.log('   GET    /api/images/:id          - Get image by ID');
  console.log('   POST   /api/images              - Upload new image');
  console.log('   DELETE /api/images/:id          - Delete image');
  console.log('   GET    /api/telemetry           - Get telemetry data');
  console.log('   POST   /api/telemetry           - Update telemetry');
  console.log('   GET    /api/statistics          - Get statistics');
  console.log('   GET    /api/statistics/health   - Database health');
  console.log('');
  console.log('✓ Background worker will process images automatically');
  console.log('   Make sure background_worker.py is running for image analysis');
  console.log('='.repeat(60));
});

export default app;




