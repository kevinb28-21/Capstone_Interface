/**
 * Image Routes
 * Handles all image-related endpoints
 */
import express from 'express';
import multer from 'multer';
import path from 'path';
import { nanoid } from 'nanoid';
import { v4 as uuidv4 } from 'uuid';
import { uploadToS3, generateS3Key } from '../s3-utils.js';
import { saveImage, getImageById } from '../db-utils.js';
import { getImagesPaginated, deleteImage } from '../db-utils-enhanced.js';
import { validateImageUpload, validateUUID, validatePagination } from '../middleware/validator.js';

const router = express.Router();

// Multer setup
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const UPLOAD_DIR = process.env.UPLOAD_DIR || path.join(process.cwd(), 'uploads');
    cb(null, UPLOAD_DIR);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname) || '.bin';
    const name = `${Date.now()}-${nanoid(8)}${ext}`;
    cb(null, name);
  }
});
const upload = multer({ 
  storage,
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB
});

/**
 * GET /api/images
 * List images with pagination and filtering
 */
router.get('/', validatePagination, async (req, res, next) => {
  try {
    const options = {
      limit: req.pagination.limit,
      offset: req.pagination.offset,
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
    res.json(result);
  } catch (error) {
    next(error);
  }
});

/**
 * GET /api/images/:id
 * Get a single image by ID
 */
router.get('/:id', validateUUID('id'), async (req, res, next) => {
  try {
    const image = await getImageById(req.params.id);
    if (!image) {
      return res.status(404).json({
        error: 'NotFound',
        message: 'Image not found',
        details: { id: req.params.id },
        timestamp: new Date().toISOString(),
        path: req.path
      });
    }
    res.json(image);
  } catch (error) {
    next(error);
  }
});

/**
 * POST /api/images
 * Upload a new image
 */
router.post('/', upload.single('image'), validateImageUpload, async (req, res, next) => {
  try {
    const id = uuidv4();
    
    // Upload to S3 (or use local path if S3 disabled)
    const s3Key = generateS3Key(req.file.filename);
    const s3Url = await uploadToS3(req.file.path, s3Key, req.file.mimetype);
    
    // Parse GPS metadata if provided
    let gpsData = null;
    if (req.body.gps) {
      try {
        gpsData = typeof req.body.gps === 'string' ? JSON.parse(req.body.gps) : req.body.gps;
      } catch (e) {
        // GPS parsing error already handled in validator
      }
    }
    
    // Save to database
    const imageId = await saveImage({
      id,
      filename: req.file.filename,
      originalName: req.file.originalname,
      filePath: req.file.path,
      s3Url: s3Url,
      s3Key: s3Url ? s3Key : null,
      s3Stored: !!s3Url,
      fileSize: req.file.size,
      mimeType: req.file.mimetype,
      gps: gpsData
    });
    
    console.log(`✓ Image ${imageId} saved to database (status: uploaded)`);
    
    // Get the saved image record
    const fileRecord = await getImageById(imageId);
    
    if (!fileRecord) {
      return res.status(500).json({
        error: 'InternalServerError',
        message: 'Failed to retrieve saved image',
        timestamp: new Date().toISOString(),
        path: req.path
      });
    }
    
    res.status(201).json(fileRecord);
  } catch (error) {
    next(error);
  }
});

/**
 * DELETE /api/images/:id
 * Delete an image and its related data
 */
router.delete('/:id', validateUUID('id'), async (req, res, next) => {
  try {
    const deleted = await deleteImage(req.params.id);
    if (!deleted) {
      return res.status(404).json({
        error: 'NotFound',
        message: 'Image not found',
        details: { id: req.params.id },
        timestamp: new Date().toISOString(),
        path: req.path
      });
    }
    res.status(204).send();
  } catch (error) {
    next(error);
  }
});

export default router;




