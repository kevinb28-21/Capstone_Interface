/**
 * Telemetry Routes
 * Handles drone telemetry endpoints
 */
import express from 'express';
import { getTelemetry, updateTelemetry } from '../db-utils.js';
import { validateTelemetry } from '../middleware/validator.js';

const router = express.Router();

/**
 * GET /api/telemetry
 * Get current telemetry data
 */
router.get('/', async (req, res, next) => {
  try {
    const telemetryData = await getTelemetry();
    res.json(telemetryData);
  } catch (error) {
    next(error);
  }
});

/**
 * POST /api/telemetry
 * Update telemetry data
 */
router.post('/', validateTelemetry, async (req, res, next) => {
  try {
    const { position, route, geofence } = req.body || {};
    
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
    next(error);
  }
});

export default router;



