import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { nanoid } from 'nanoid';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT || 5000;
const ORIGIN = process.env.ORIGIN || 'http://localhost:5173';
const UPLOAD_DIR = process.env.UPLOAD_DIR || path.join(__dirname, '..', 'uploads');

// Ensure upload dir exists
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const app = express();
app.use(cors({ origin: ORIGIN }));
app.use(express.json({ limit: '10mb' }));

// In-memory stores (replace with DB later)
const images = new Map(); // id -> { id, filename, originalName, path, analysis, createdAt }
const telemetry = {
  position: { lat: 37.7749, lng: -122.4194 },
  route: [],
  geofence: [
    { lat: 37.7755, lng: -122.421 },
    { lat: 37.773, lng: -122.421 },
    { lat: 37.773, lng: -122.4175 },
    { lat: 37.7755, lng: -122.4175 }
  ]
};

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

// Placeholder analysis function
function analyzeCropHealthPlaceholder(filePath) {
  // Mock NDVI-like score and stress areas
  const ndvi = Number((Math.random() * 0.5 + 0.25).toFixed(2)); // 0.25 - 0.75
  const stressZones = Array.from({ length: 5 }, () => ({
    x: Math.floor(Math.random() * 10),
    y: Math.floor(Math.random() * 10),
    severity: Number((Math.random() * 0.5 + 0.5).toFixed(2))
  }));
  return {
    ndvi,
    summary: ndvi > 0.5 ? 'Healthy' : 'Attention needed',
    stressZones
  };
}

// Routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Serve uploaded files
app.use('/uploads', express.static(UPLOAD_DIR));

// Upload image and analyze
app.post('/api/images', upload.single('image'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image uploaded' });
  }
  const id = nanoid(12);
  const fileRecord = {
    id,
    filename: req.file.filename,
    originalName: req.file.originalname,
    path: `/uploads/${req.file.filename}`,
    createdAt: new Date().toISOString()
  };
  const analysis = analyzeCropHealthPlaceholder(req.file.path);
  fileRecord.analysis = analysis;
  images.set(id, fileRecord);
  res.json(fileRecord);
});

// List images
app.get('/api/images', (req, res) => {
  res.json(Array.from(images.values()).sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1)));
});

// Get one image
app.get('/api/images/:id', (req, res) => {
  const item = images.get(req.params.id);
  if (!item) return res.status(404).json({ error: 'Not found' });
  res.json(item);
});

// Telemetry
app.get('/api/telemetry', (req, res) => {
  res.json(telemetry);
});

app.post('/api/telemetry', (req, res) => {
  const { position, route, geofence } = req.body || {};
  if (position && typeof position.lat === 'number' && typeof position.lng === 'number') {
    telemetry.position = position;
  }
  if (Array.isArray(route)) {
    telemetry.route = route;
  } else {
    // Append current position to route if not provided
    telemetry.route.push(telemetry.position);
    if (telemetry.route.length > 5000) telemetry.route.shift();
  }
  if (Array.isArray(geofence)) {
    telemetry.geofence = geofence;
  }
  res.json(telemetry);
});

app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});



