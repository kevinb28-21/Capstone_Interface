/**
 * Statistics Routes
 * Provides analytics and statistics endpoints
 */
import express from 'express';
import { getStatistics, getDatabaseHealth } from '../db-utils-enhanced.js';
import { testConnection } from '../db-utils.js';

const router = express.Router();

/**
 * GET /api/statistics
 * Get overall statistics about images and analyses
 */
router.get('/', async (req, res, next) => {
  try {
    const stats = await getStatistics();
    res.json(stats);
  } catch (error) {
    next(error);
  }
});

/**
 * GET /api/statistics/health
 * Get database health information
 */
router.get('/health', async (req, res, next) => {
  try {
    const dbStatus = await testConnection();
    const health = await getDatabaseHealth();
    
    res.json({
      ...health,
      connectionTest: dbStatus
    });
  } catch (error) {
    next(error);
  }
});

export default router;



