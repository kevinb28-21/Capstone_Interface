/**
 * Request Logging Middleware
 * Logs all API requests for monitoring and debugging
 */

export function requestLogger(req, res, next) {
  const startTime = Date.now();
  const originalSend = res.json;

  // Override res.json to log response
  res.json = function(body) {
    const duration = Date.now() - startTime;
    
    console.log({
      method: req.method,
      path: req.path,
      statusCode: res.statusCode,
      duration: `${duration}ms`,
      timestamp: new Date().toISOString(),
      userAgent: req.get('user-agent'),
      ip: req.ip || req.connection.remoteAddress
    });

    return originalSend.call(this, body);
  };

  next();
}




