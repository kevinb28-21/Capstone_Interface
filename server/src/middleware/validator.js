/**
 * Request Validation Middleware
 * Validates request data before processing
 */

/**
 * Validate image upload request
 */
export function validateImageUpload(req, res, next) {
  if (!req.file) {
    return res.status(400).json({
      error: 'ValidationError',
      message: 'No image file provided',
      details: { field: 'image', required: true },
      timestamp: new Date().toISOString(),
      path: req.path
    });
  }

  // Validate file type
  const allowedMimeTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/tif'];
  if (!allowedMimeTypes.includes(req.file.mimetype)) {
    return res.status(400).json({
      error: 'ValidationError',
      message: 'Invalid file type',
      details: { 
        field: 'image',
        allowedTypes: allowedMimeTypes,
        receivedType: req.file.mimetype
      },
      timestamp: new Date().toISOString(),
      path: req.path
    });
  }

  // Validate file size (10MB max)
  const maxSize = 10 * 1024 * 1024; // 10MB
  if (req.file.size > maxSize) {
    return res.status(413).json({
      error: 'ValidationError',
      message: 'File size exceeds maximum allowed size',
      details: { 
        field: 'image',
        maxSize: '10MB',
        receivedSize: `${(req.file.size / 1024 / 1024).toFixed(2)}MB`
      },
      timestamp: new Date().toISOString(),
      path: req.path
    });
  }

  // Validate GPS data if provided
  if (req.body.gps) {
    try {
      const gps = typeof req.body.gps === 'string' ? JSON.parse(req.body.gps) : req.body.gps;
      
      if (gps.latitude !== undefined) {
        if (typeof gps.latitude !== 'number' || gps.latitude < -90 || gps.latitude > 90) {
          return res.status(400).json({
            error: 'ValidationError',
            message: 'Invalid GPS latitude',
            details: { field: 'gps.latitude', range: '[-90, 90]' },
            timestamp: new Date().toISOString(),
            path: req.path
          });
        }
      }

      if (gps.longitude !== undefined) {
        if (typeof gps.longitude !== 'number' || gps.longitude < -180 || gps.longitude > 180) {
          return res.status(400).json({
            error: 'ValidationError',
            message: 'Invalid GPS longitude',
            details: { field: 'gps.longitude', range: '[-180, 180]' },
            timestamp: new Date().toISOString(),
            path: req.path
          });
        }
      }
    } catch (e) {
      return res.status(400).json({
        error: 'ValidationError',
        message: 'Invalid GPS data format',
        details: { field: 'gps', error: e.message },
        timestamp: new Date().toISOString(),
        path: req.path
      });
    }
  }

  next();
}

/**
 * Validate UUID parameter
 */
export function validateUUID(paramName = 'id') {
  return (req, res, next) => {
    const uuid = req.params[paramName];
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    
    if (!uuid || !uuidRegex.test(uuid)) {
      return res.status(400).json({
        error: 'ValidationError',
        message: 'Invalid UUID format',
        details: { field: paramName, value: uuid },
        timestamp: new Date().toISOString(),
        path: req.path
      });
    }
    
    next();
  };
}

/**
 * Validate pagination parameters
 */
export function validatePagination(req, res, next) {
  const limit = parseInt(req.query.limit || '50', 10);
  const offset = parseInt(req.query.offset || '0', 10);
  const page = parseInt(req.query.page || '1', 10);

  if (limit < 1 || limit > 100) {
    return res.status(400).json({
      error: 'ValidationError',
      message: 'Invalid limit parameter',
      details: { field: 'limit', range: '[1, 100]', received: limit },
      timestamp: new Date().toISOString(),
      path: req.path
    });
  }

  if (offset < 0) {
    return res.status(400).json({
      error: 'ValidationError',
      message: 'Invalid offset parameter',
      details: { field: 'offset', minimum: 0, received: offset },
      timestamp: new Date().toISOString(),
      path: req.path
    });
  }

  if (page < 1) {
    return res.status(400).json({
      error: 'ValidationError',
      message: 'Invalid page parameter',
      details: { field: 'page', minimum: 1, received: page },
      timestamp: new Date().toISOString(),
      path: req.path
    });
  }

  req.pagination = { limit, offset, page };
  next();
}

/**
 * Validate telemetry data
 */
export function validateTelemetry(req, res, next) {
  const { position, route, geofence } = req.body || {};

  if (position) {
    if (typeof position.lat !== 'number' || position.lat < -90 || position.lat > 90) {
      return res.status(400).json({
        error: 'ValidationError',
        message: 'Invalid position latitude',
        details: { field: 'position.lat', range: '[-90, 90]' },
        timestamp: new Date().toISOString(),
        path: req.path
      });
    }

    if (typeof position.lng !== 'number' || position.lng < -180 || position.lng > 180) {
      return res.status(400).json({
        error: 'ValidationError',
        message: 'Invalid position longitude',
        details: { field: 'position.lng', range: '[-180, 180]' },
        timestamp: new Date().toISOString(),
        path: req.path
      });
    }
  }

  if (route && !Array.isArray(route)) {
    return res.status(400).json({
      error: 'ValidationError',
      message: 'Route must be an array',
      details: { field: 'route' },
      timestamp: new Date().toISOString(),
      path: req.path
    });
  }

  if (geofence && !Array.isArray(geofence)) {
    return res.status(400).json({
      error: 'ValidationError',
      message: 'Geofence must be an array',
      details: { field: 'geofence' },
      timestamp: new Date().toISOString(),
      path: req.path
    });
  }

  next();
}




