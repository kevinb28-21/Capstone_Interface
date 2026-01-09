/**
 * Error Handling Middleware
 * Provides consistent error responses across the API
 */

export function errorHandler(err, req, res, next) {
  console.error('Error:', {
    message: err.message,
    stack: err.stack,
    path: req.path,
    method: req.method,
    timestamp: new Date().toISOString()
  });

  // Database connection errors
  if (err.code && err.code.startsWith('ECONNREFUSED')) {
    return res.status(503).json({
      error: 'DatabaseConnectionError',
      message: 'Database connection failed',
      details: { code: err.code },
      timestamp: new Date().toISOString(),
      path: req.path
    });
  }

  // PostgreSQL errors
  if (err.code && err.code.startsWith('23')) {
    return res.status(409).json({
      error: 'DatabaseConstraintError',
      message: 'Database constraint violation',
      details: { code: err.code, detail: err.detail },
      timestamp: new Date().toISOString(),
      path: req.path
    });
  }

  // Validation errors
  if (err.name === 'ValidationError') {
    return res.status(422).json({
      error: 'ValidationError',
      message: 'Request validation failed',
      details: err.details || { message: err.message },
      timestamp: new Date().toISOString(),
      path: req.path
    });
  }

  // Multer errors (file upload)
  if (err.name === 'MulterError') {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(413).json({
        error: 'FileTooLarge',
        message: 'File size exceeds maximum allowed size',
        details: { maxSize: '10MB' },
        timestamp: new Date().toISOString(),
        path: req.path
      });
    }
    return res.status(400).json({
      error: 'FileUploadError',
      message: 'File upload failed',
      details: { code: err.code },
      timestamp: new Date().toISOString(),
      path: req.path
    });
  }

  // Default error response
  const statusCode = err.statusCode || err.status || 500;
  res.status(statusCode).json({
    error: err.name || 'InternalServerError',
    message: err.message || 'An unexpected error occurred',
    details: process.env.NODE_ENV === 'development' ? { stack: err.stack } : undefined,
    timestamp: new Date().toISOString(),
    path: req.path
  });
}

/**
 * 404 Not Found handler
 */
export function notFoundHandler(req, res) {
  res.status(404).json({
    error: 'NotFound',
    message: `Route ${req.method} ${req.path} not found`,
    timestamp: new Date().toISOString(),
    path: req.path
  });
}




