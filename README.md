## Capstone Interface – Drone Crop Health Dashboard

This project is a full‑stack web application for monitoring onion crop health using drone-captured images. The platform performs automated analysis using vegetation indices (NDVI, SAVI, GNDVI) and machine learning models.

> 📚 **For complete documentation, see [documentation/README.md](./documentation/README.md)**

### Features
- Image upload API with placeholder crop health analysis (mock NDVI/stress zones)
- Telemetry API for drone position, route, and geofence storage/retrieval
- React dashboard with Leaflet map: live drone marker, route polyline, geofence polygon
- Analysis results displayed per uploaded image

### Tech
- Backend: Node.js, Express, Multer, CORS
- Frontend: React (Vite), React‑Leaflet, Leaflet

### Getting Started

1) Backend
```bash
cd server
npm install
cp .env.example .env
npm run dev
```
The server starts on `http://localhost:5000` by default.

2) Frontend
```bash
cd client
npm install
npm run dev
```
Open the printed local URL (typically `http://localhost:5173`).

### API

- POST `/api/images` (multipart/form-data):
  - field: `image` (file)
  - returns: `{ id, filename, analysis }`

- GET `/api/images`: list uploaded images with analyses

- GET `/api/images/:id`: single image with analysis

- POST `/api/telemetry` (application/json):
  - `{ position: { lat, lng }, route?: [{lat,lng},...], geofence?: [{lat,lng},...] }`
  - returns: current state

- GET `/api/telemetry`: returns `{ position, route, geofence }`

### Notes
- Analysis is placeholder logic (randomized/mock NDVI scores and stress zones) until a real model is integrated.
- Uploaded images are stored under `server/uploads/`.

### Next Steps
- Replace placeholder analysis with a real ML model (Python service or Node binding) once dataset/model is ready.
- Persist telemetry and analyses in a database.



