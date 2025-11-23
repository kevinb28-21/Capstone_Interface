/**
 * Database Utilities for PostgreSQL (Node.js)
 * Handles database connections and queries for the Node.js backend
 */
import pg from 'pg';
import dotenv from 'dotenv';

dotenv.config();

const { Pool } = pg;

// Database connection pool
let pool = null;

/**
 * Get or create database connection pool
 * @returns {pg.Pool}
 */
export function getDbPool() {
  if (!pool) {
    try {
      pool = new Pool({
        host: process.env.DB_HOST || 'localhost',
        port: parseInt(process.env.DB_PORT || '5432', 10),
        database: process.env.DB_NAME || 'drone_analytics',
        user: process.env.DB_USER || 'postgres',
        password: process.env.DB_PASSWORD || '',
        max: 10, // Maximum number of clients in the pool
        idleTimeoutMillis: 30000,
        connectionTimeoutMillis: 2000,
      });
      
      // Handle pool errors
      pool.on('error', (err) => {
        console.error('Unexpected error on idle client', err);
      });
      
      console.log('✓ Database connection pool created');
    } catch (error) {
      console.error('Failed to create database pool:', error);
      throw error;
    }
  }
  return pool;
}

/**
 * Test database connection
 * @returns {Promise<boolean>}
 */
export async function testConnection() {
  try {
    const pool = getDbPool();
    const result = await pool.query('SELECT 1');
    return result.rows.length > 0;
  } catch (error) {
    console.error('Database connection test failed:', error);
    return false;
  }
}

/**
 * Get all images with GPS and analysis data
 * @param {number} limit - Maximum number of images to return
 * @returns {Promise<Array>}
 */
export async function getAllImages(limit = 100) {
  const pool = getDbPool();
  try {
    const result = await pool.query(`
      SELECT 
        i.id, 
        i.filename, 
        i.original_name, 
        i.s3_url, 
        i.file_path,
        i.uploaded_at, 
        i.processing_status,
        i.s3_stored,
        g.latitude, 
        g.longitude,
        g.altitude,
        a.ndvi_mean, 
        a.savi_mean, 
        a.gndvi_mean,
        a.health_status, 
        a.summary,
        a.analysis_type
      FROM images i
      LEFT JOIN image_gps g ON i.id = g.image_id
      LEFT JOIN analyses a ON i.id = a.image_id
      ORDER BY i.uploaded_at DESC
      LIMIT $1
    `, [limit]);
    
    return result.rows.map(row => ({
      id: row.id,
      filename: row.filename,
      originalName: row.original_name,
      path: row.s3_url || row.file_path,
      s3Url: row.s3_url,
      s3Stored: row.s3_stored,
      createdAt: row.uploaded_at?.toISOString(),
      processingStatus: row.processing_status,
      gps: row.latitude && row.longitude ? {
        latitude: parseFloat(row.latitude),
        longitude: parseFloat(row.longitude),
        altitude: row.altitude ? parseFloat(row.altitude) : null
      } : null,
      analysis: row.ndvi_mean !== null ? {
        ndvi: parseFloat(row.ndvi_mean),
        savi: row.savi_mean ? parseFloat(row.savi_mean) : null,
        gndvi: row.gndvi_mean ? parseFloat(row.gndvi_mean) : null,
        healthStatus: row.health_status,
        summary: row.summary,
        analysisType: row.analysis_type
      } : null
    }));
  } catch (error) {
    console.error('Error fetching images:', error);
    throw error;
  }
}

/**
 * Get a single image by ID
 * @param {string} imageId - UUID of the image
 * @returns {Promise<Object|null>}
 */
export async function getImageById(imageId) {
  const pool = getDbPool();
  try {
    const result = await pool.query(`
      SELECT 
        i.*, 
        g.latitude, 
        g.longitude, 
        g.altitude,
        a.ndvi_mean, 
        a.savi_mean, 
        a.gndvi_mean,
        a.health_status, 
        a.summary,
        a.analysis_type
      FROM images i
      LEFT JOIN image_gps g ON i.id = g.image_id
      LEFT JOIN analyses a ON i.id = a.image_id
      WHERE i.id = $1
    `, [imageId]);
    
    if (result.rows.length === 0) {
      return null;
    }
    
    const row = result.rows[0];
    return {
      id: row.id,
      filename: row.filename,
      originalName: row.original_name,
      path: row.s3_url || row.file_path,
      s3Url: row.s3_url,
      s3Key: row.s3_key,
      s3Stored: row.s3_stored,
      createdAt: row.uploaded_at?.toISOString(),
      processingStatus: row.processing_status,
      gps: row.latitude && row.longitude ? {
        latitude: parseFloat(row.latitude),
        longitude: parseFloat(row.longitude),
        altitude: row.altitude ? parseFloat(row.altitude) : null
      } : null,
      analysis: row.ndvi_mean !== null ? {
        ndvi: parseFloat(row.ndvi_mean),
        savi: row.savi_mean ? parseFloat(row.savi_mean) : null,
        gndvi: row.gndvi_mean ? parseFloat(row.gndvi_mean) : null,
        healthStatus: row.health_status,
        summary: row.summary,
        analysisType: row.analysis_type
      } : null
    };
  } catch (error) {
    console.error('Error fetching image:', error);
    throw error;
  }
}

/**
 * Save image to database
 * @param {Object} imageData - Image data to save
 * @returns {Promise<string>} - Image ID (UUID)
 */
export async function saveImage(imageData) {
  const pool = getDbPool();
  const client = await pool.connect();
  
  try {
    await client.query('BEGIN');
    
    // Insert image record
    const imageResult = await client.query(`
      INSERT INTO images (
        id, filename, original_name, file_path, s3_url, s3_key, s3_stored,
        file_size, mime_type, captured_at, uploaded_at, processing_status
      ) VALUES (
        $1, $2, $3, $4, $5, $6, $7,
        $8, $9, $10, CURRENT_TIMESTAMP, 'uploaded'
      )
      RETURNING id, uploaded_at
    `, [
      imageData.id,
      imageData.filename,
      imageData.originalName,
      imageData.filePath || null,
      imageData.s3Url || null,
      imageData.s3Key || null,
      imageData.s3Stored || false,
      imageData.fileSize || null,
      imageData.mimeType || 'image/jpeg',
      imageData.capturedAt ? new Date(imageData.capturedAt) : new Date(),
    ]);
    
    const imageId = imageResult.rows[0].id;
    
    // Insert GPS data if provided
    if (imageData.gps && imageData.gps.latitude && imageData.gps.longitude) {
      await client.query(`
        INSERT INTO image_gps (
          image_id, latitude, longitude, altitude, accuracy,
          heading, ground_speed, speed, captured_at
        ) VALUES (
          $1, $2, $3, $4, $5,
          $6, $7, $8, $9
        )
      `, [
        imageId,
        imageData.gps.latitude,
        imageData.gps.longitude,
        imageData.gps.altitude || null,
        imageData.gps.accuracy || null,
        imageData.gps.bearing || imageData.gps.heading || null,
        imageData.gps.speed || null,
        imageData.gps.speed || null,
        imageData.gps.timestamp ? new Date(imageData.gps.timestamp) : new Date(),
      ]);
    }
    
    await client.query('COMMIT');
    return imageId;
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Error saving image:', error);
    throw error;
  } finally {
    client.release();
  }
}

/**
 * Get telemetry data (current position, route, geofence)
 * @returns {Promise<Object>}
 */
export async function getTelemetry() {
  const pool = getDbPool();
  try {
    // Get current telemetry
    const telemetryResult = await pool.query(`
      SELECT latitude, longitude, altitude, heading, ground_speed, battery_level, status
      FROM telemetry
      ORDER BY timestamp DESC
      LIMIT 1
    `);
    
    // Get route points (last 5000)
    const routeResult = await pool.query(`
      SELECT latitude, longitude, altitude, timestamp
      FROM route_points
      ORDER BY sequence ASC, timestamp ASC
      LIMIT 5000
    `);
    
    // Get active geofence
    const geofenceResult = await pool.query(`
      SELECT gp.latitude, gp.longitude
      FROM geofences g
      JOIN geofence_points gp ON g.id = gp.geofence_id
      WHERE g.is_active = true
      ORDER BY gp.sequence ASC
    `);
    
    const position = telemetryResult.rows.length > 0 ? {
      lat: parseFloat(telemetryResult.rows[0].latitude),
      lng: parseFloat(telemetryResult.rows[0].longitude),
      altitude: telemetryResult.rows[0].altitude ? parseFloat(telemetryResult.rows[0].altitude) : null
    } : { lat: 43.6532, lng: -79.3832 }; // Default Toronto
    
    const route = routeResult.rows.map(row => ({
      lat: parseFloat(row.latitude),
      lng: parseFloat(row.longitude),
      altitude: row.altitude ? parseFloat(row.altitude) : null
    }));
    
    const geofence = geofenceResult.rows.map(row => ({
      lat: parseFloat(row.latitude),
      lng: parseFloat(row.longitude)
    }));
    
    return {
      position,
      route,
      geofence: geofence.length > 0 ? geofence : [
        { lat: 43.6555, lng: -79.391 },
        { lat: 43.6505, lng: -79.391 },
        { lat: 43.6505, lng: -79.3755 },
        { lat: 43.6555, lng: -79.3755 }
      ] // Default geofence
    };
  } catch (error) {
    console.error('Error fetching telemetry:', error);
    // Return default telemetry on error
    return {
      position: { lat: 43.6532, lng: -79.3832 },
      route: [],
      geofence: [
        { lat: 43.6555, lng: -79.391 },
        { lat: 43.6505, lng: -79.391 },
        { lat: 43.6505, lng: -79.3755 },
        { lat: 43.6555, lng: -79.3755 }
      ]
    };
  }
}

/**
 * Update telemetry data
 * @param {Object} telemetryData - Telemetry data to update
 * @returns {Promise<void>}
 */
export async function updateTelemetry(telemetryData) {
  const pool = getDbPool();
  const client = await pool.connect();
  
  try {
    await client.query('BEGIN');
    
    // Update current telemetry
    if (telemetryData.position) {
      await client.query(`
        INSERT INTO telemetry (latitude, longitude, altitude, heading, ground_speed, status, timestamp)
        VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
      `, [
        telemetryData.position.lat,
        telemetryData.position.lng,
        telemetryData.position.altitude || null,
        telemetryData.position.heading || null,
        telemetryData.position.speed || null,
        'flying'
      ]);
    }
    
    // Update route points
    if (telemetryData.route && Array.isArray(telemetryData.route)) {
      // Clear old route points (optional - or keep history)
      // await client.query('DELETE FROM route_points');
      
      // Insert new route points
      for (let i = 0; i < telemetryData.route.length; i++) {
        const point = telemetryData.route[i];
        await client.query(`
          INSERT INTO route_points (latitude, longitude, altitude, sequence, timestamp)
          VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
        `, [
          point.lat,
          point.lng,
          point.altitude || null,
          i
        ]);
      }
    }
    
    // Update geofence
    if (telemetryData.geofence && Array.isArray(telemetryData.geofence)) {
      // Get active geofence ID
      const geofenceResult = await client.query(`
        SELECT id FROM geofences WHERE is_active = true LIMIT 1
      `);
      
      let geofenceId;
      if (geofenceResult.rows.length === 0) {
        // Create new active geofence
        const newGeofenceResult = await client.query(`
          INSERT INTO geofences (name, description, is_active)
          VALUES ('Default Geofence', 'User-defined geofence', true)
          RETURNING id
        `);
        geofenceId = newGeofenceResult.rows[0].id;
      } else {
        geofenceId = geofenceResult.rows[0].id;
        // Clear old points
        await client.query('DELETE FROM geofence_points WHERE geofence_id = $1', [geofenceId]);
      }
      
      // Insert new geofence points
      for (let i = 0; i < telemetryData.geofence.length; i++) {
        const point = telemetryData.geofence[i];
        await client.query(`
          INSERT INTO geofence_points (geofence_id, latitude, longitude, sequence)
          VALUES ($1, $2, $3, $4)
        `, [geofenceId, point.lat, point.lng, i]);
      }
    }
    
    await client.query('COMMIT');
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Error updating telemetry:', error);
    throw error;
  } finally {
    client.release();
  }
}

