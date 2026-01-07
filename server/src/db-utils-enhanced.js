/**
 * Enhanced Database Utilities
 * Provides pagination, filtering, statistics, and advanced queries
 */
import { getDbPool, hasGndviColumns } from './db-utils.js';

/**
 * Get images with pagination, filtering, and sorting
 * @param {Object} options - Query options
 * @returns {Promise<Object>} Paginated result with metadata
 */
export async function getImagesPaginated(options = {}) {
  const pool = getDbPool();
  const {
    limit = 50,
    offset = 0,
    status,
    hasAnalysis,
    minHealthScore,
    maxHealthScore,
    healthStatus,
    sortBy = 'uploaded_at',
    sortOrder = 'DESC',
    search
  } = options;

  try {
    const hasGndvi = await hasGndviColumns();
    const gndviFields = hasGndvi 
      ? 'a.gndvi_mean, a.gndvi_std, a.gndvi_min, a.gndvi_max,'
      : '';

    // Build WHERE clause
    const conditions = [];
    const params = [];
    let paramIndex = 1;

    if (status) {
      conditions.push(`i.processing_status = $${paramIndex++}`);
      params.push(status);
    }

    if (hasAnalysis !== undefined) {
      if (hasAnalysis) {
        conditions.push('a.id IS NOT NULL');
      } else {
        conditions.push('a.id IS NULL');
      }
    }

    if (minHealthScore !== undefined) {
      conditions.push(`a.health_score >= $${paramIndex++}`);
      params.push(minHealthScore);
    }

    if (maxHealthScore !== undefined) {
      conditions.push(`a.health_score <= $${paramIndex++}`);
      params.push(maxHealthScore);
    }

    if (healthStatus) {
      conditions.push(`a.health_status = $${paramIndex++}`);
      params.push(healthStatus);
    }

    if (search) {
      conditions.push(`(
        i.original_name ILIKE $${paramIndex} OR
        i.filename ILIKE $${paramIndex} OR
        a.summary ILIKE $${paramIndex}
      )`);
      params.push(`%${search}%`);
      paramIndex++;
    }

    const whereClause = conditions.length > 0 
      ? `WHERE ${conditions.join(' AND ')}`
      : '';

    // Validate sortBy
    const allowedSortFields = {
      'uploaded_at': 'i.uploaded_at',
      'processed_at': 'i.processed_at',
      'health_score': 'a.health_score',
      'ndvi_mean': 'a.ndvi_mean',
      'filename': 'i.original_name'
    };
    const sortField = allowedSortFields[sortBy] || 'i.uploaded_at';
    const sortDir = sortOrder.toUpperCase() === 'ASC' ? 'ASC' : 'DESC';

    // Count total
    const countQuery = `
      SELECT COUNT(*) as total
      FROM images i
      LEFT JOIN image_gps g ON i.id = g.image_id
      LEFT JOIN analyses a ON i.id = a.image_id
      ${whereClause}
    `;
    const countResult = await pool.query(countQuery, params);
    const total = parseInt(countResult.rows[0].total, 10);

    // Fetch paginated results
    params.push(limit, offset);
    const query = `
      SELECT 
        i.id, 
        i.filename, 
        i.original_name, 
        i.s3_url, 
        i.file_path,
        i.uploaded_at,
        i.processed_at,
        i.processing_status,
        i.s3_stored,
        g.latitude, 
        g.longitude,
        g.altitude,
        a.ndvi_mean, a.ndvi_std, a.ndvi_min, a.ndvi_max,
        a.savi_mean, a.savi_std, a.savi_min, a.savi_max,
        ${gndviFields}
        a.health_score,
        a.health_status, 
        a.summary,
        a.analysis_type,
        a.model_version,
        a.confidence,
        a.processed_s3_url,
        a.processed_image_path
      FROM images i
      LEFT JOIN image_gps g ON i.id = g.image_id
      LEFT JOIN analyses a ON i.id = a.image_id
      ${whereClause}
      ORDER BY ${sortField} ${sortDir}
      LIMIT $${paramIndex++} OFFSET $${paramIndex++}
    `;

    const result = await pool.query(query, params);

    // Transform rows (same as getAllImages)
    const images = result.rows.map(row => ({
      id: row.id,
      filename: row.filename,
      originalName: row.original_name,
      path: row.s3_url || row.file_path,
      s3Url: row.s3_url,
      s3Stored: row.s3_stored,
      createdAt: row.uploaded_at?.toISOString(),
      processedAt: row.processed_at?.toISOString(),
      processingStatus: row.processing_status || 'uploaded',
      gps: row.latitude && row.longitude ? {
        latitude: parseFloat(row.latitude),
        longitude: parseFloat(row.longitude),
        altitude: row.altitude ? parseFloat(row.altitude) : null
      } : null,
      analysis: row.ndvi_mean !== null ? {
        ndvi: {
          mean: parseFloat(row.ndvi_mean),
          std: row.ndvi_std ? parseFloat(row.ndvi_std) : null,
          min: row.ndvi_min ? parseFloat(row.ndvi_min) : null,
          max: row.ndvi_max ? parseFloat(row.ndvi_max) : null
        },
        savi: row.savi_mean !== null ? {
          mean: parseFloat(row.savi_mean),
          std: row.savi_std ? parseFloat(row.savi_std) : null,
          min: row.savi_min ? parseFloat(row.savi_min) : null,
          max: row.savi_max ? parseFloat(row.savi_max) : null
        } : null,
        gndvi: hasGndvi && row.gndvi_mean !== null ? {
          mean: parseFloat(row.gndvi_mean),
          std: row.gndvi_std ? parseFloat(row.gndvi_std) : null,
          min: row.gndvi_min ? parseFloat(row.gndvi_min) : null,
          max: row.gndvi_max ? parseFloat(row.gndvi_max) : null
        } : null,
        healthScore: row.health_score ? parseFloat(row.health_score) : null,
        healthStatus: row.health_status,
        summary: row.summary,
        analysisType: row.analysis_type,
        modelVersion: row.model_version,
        confidence: row.confidence ? parseFloat(row.confidence) : null,
        processedImageUrl: row.processed_s3_url,
        processedImagePath: row.processed_image_path
      } : null
    }));

    const pages = Math.ceil(total / limit);
    const currentPage = Math.floor(offset / limit) + 1;

    return {
      items: images,
      pagination: {
        total,
        limit,
        offset,
        page: currentPage,
        pages,
        hasNext: offset + limit < total,
        hasPrev: offset > 0
      }
    };
  } catch (error) {
    console.error('Error fetching paginated images:', error);
    throw error;
  }
}

/**
 * Get statistics about images and analyses
 * @returns {Promise<Object>} Statistics object
 */
export async function getStatistics() {
  const pool = getDbPool();
  
  try {
    const hasGndvi = await hasGndviColumns();
    
    // Total images
    const totalImagesResult = await pool.query('SELECT COUNT(*) as count FROM images');
    const totalImages = parseInt(totalImagesResult.rows[0].count, 10);

    // Images by status
    const statusResult = await pool.query(`
      SELECT processing_status, COUNT(*) as count
      FROM images
      GROUP BY processing_status
    `);
    const byStatus = {};
    statusResult.rows.forEach(row => {
      byStatus[row.processing_status] = parseInt(row.count, 10);
    });

    // Images with analysis
    const analyzedResult = await pool.query(`
      SELECT COUNT(*) as count
      FROM images i
      INNER JOIN analyses a ON i.id = a.image_id
    `);
    const analyzed = parseInt(analyzedResult.rows[0].count, 10);

    // Average health metrics
    const avgMetricsResult = await pool.query(`
      SELECT 
        AVG(ndvi_mean) as avg_ndvi,
        AVG(savi_mean) as avg_savi,
        ${hasGndvi ? 'AVG(gndvi_mean) as avg_gndvi,' : ''}
        AVG(health_score) as avg_health_score,
        AVG(confidence) as avg_confidence
      FROM analyses
      WHERE ndvi_mean IS NOT NULL
    `);
    const avgMetrics = avgMetricsResult.rows[0];

    // Health status distribution
    const healthStatusResult = await pool.query(`
      SELECT health_status, COUNT(*) as count
      FROM analyses
      WHERE health_status IS NOT NULL
      GROUP BY health_status
    `);
    const healthStatusDistribution = {};
    healthStatusResult.rows.forEach(row => {
      healthStatusDistribution[row.health_status] = parseInt(row.count, 10);
    });

    // Recent activity (last 24 hours)
    const recentActivityResult = await pool.query(`
      SELECT COUNT(*) as count
      FROM images
      WHERE uploaded_at >= NOW() - INTERVAL '24 hours'
    `);
    const recentActivity = parseInt(recentActivityResult.rows[0].count, 10);

    // Images with GPS
    const gpsResult = await pool.query(`
      SELECT COUNT(*) as count
      FROM images i
      INNER JOIN image_gps g ON i.id = g.image_id
    `);
    const withGps = parseInt(gpsResult.rows[0].count, 10);

    return {
      totalImages,
      analyzed,
      byStatus,
      averageMetrics: {
        ndvi: avgMetrics.avg_ndvi ? parseFloat(avgMetrics.avg_ndvi) : null,
        savi: avgMetrics.avg_savi ? parseFloat(avgMetrics.avg_savi) : null,
        gndvi: hasGndvi && avgMetrics.avg_gndvi ? parseFloat(avgMetrics.avg_gndvi) : null,
        healthScore: avgMetrics.avg_health_score ? parseFloat(avgMetrics.avg_health_score) : null,
        confidence: avgMetrics.avg_confidence ? parseFloat(avgMetrics.avg_confidence) : null
      },
      healthStatusDistribution,
      recentActivity,
      withGps,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    console.error('Error fetching statistics:', error);
    throw error;
  }
}

/**
 * Delete an image and its related data
 * @param {string} imageId - UUID of the image
 * @returns {Promise<boolean>} Success status
 */
export async function deleteImage(imageId) {
  const pool = getDbPool();
  const client = await pool.connect();
  
  try {
    await client.query('BEGIN');
    
    // Delete will cascade to related tables (image_gps, analyses, stress_zones)
    const result = await client.query('DELETE FROM images WHERE id = $1 RETURNING id', [imageId]);
    
    if (result.rows.length === 0) {
      await client.query('ROLLBACK');
      return false;
    }
    
    await client.query('COMMIT');
    return true;
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Error deleting image:', error);
    throw error;
  } finally {
    client.release();
  }
}

/**
 * Get database health information
 * @returns {Promise<Object>} Health status
 */
export async function getDatabaseHealth() {
  const pool = getDbPool();
  
  try {
    // Test connection
    await pool.query('SELECT 1');
    
    // Get pool stats
    const poolStats = {
      totalCount: pool.totalCount,
      idleCount: pool.idleCount,
      waitingCount: pool.waitingCount
    };
    
    // Get database size
    const sizeResult = await pool.query(`
      SELECT pg_size_pretty(pg_database_size(current_database())) as size
    `);
    const dbSize = sizeResult.rows[0].size;
    
    // Get table counts
    const tableCounts = {};
    const tables = ['images', 'analyses', 'image_gps', 'telemetry', 'route_points', 'geofences'];
    for (const table of tables) {
      try {
        const countResult = await pool.query(`SELECT COUNT(*) as count FROM ${table}`);
        tableCounts[table] = parseInt(countResult.rows[0].count, 10);
      } catch (e) {
        // Table might not exist
        tableCounts[table] = null;
      }
    }
    
    return {
      connected: true,
      poolStats,
      databaseSize: dbSize,
      tableCounts,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    return {
      connected: false,
      error: error.message,
      timestamp: new Date().toISOString()
    };
  }
}



