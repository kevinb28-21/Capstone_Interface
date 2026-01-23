# Netlify Deployment Guide

This guide covers deploying your React frontend to Netlify and connecting it to your backend services.

## Architecture Overview

Since Netlify is primarily for frontend/static sites, here's the recommended setup:

```
┌─────────────────┐
│  Netlify        │ → React Frontend (Static)
└────────┬────────┘
         │
    ┌────┴────┬──────────────┐
    │         │              │
┌───▼───┐ ┌──▼───┐    ┌─────▼─────┐
│Railway│ │Railway│  │   AWS S3   │
│Backend│ │Flask  │  │  Storage   │
└───────┘ └───────┘  └───────────┘
```

**Frontend**: Netlify (free tier)  
**Backend**: Railway/Render ($5/month)  
**Python API**: Railway/Render (same account)  
**Database**: Railway PostgreSQL (included)  
**Storage**: AWS S3 (already configured)

---

## Step 1: Prepare Frontend for Netlify

### 1.1 Update Vite Config for Production

Update `client/vite.config.js` to handle API URLs:

```js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    proxy: {
      '/api': 'http://localhost:5050',
      '/uploads': 'http://localhost:5050'
    }
  },
  // Production build settings
  build: {
    outDir: 'dist',
    sourcemap: false
  },
  // Define environment variables
  define: {
    'import.meta.env.VITE_API_URL': JSON.stringify(
      process.env.VITE_API_URL || 'http://localhost:5050'
    )
  }
});
```

### 1.2 Create API Utility

Create `client/src/utils/api.js`:

```js
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5050';

export const api = {
  get: async (endpoint) => {
    const response = await fetch(`${API_URL}${endpoint}`);
    return response.json();
  },
  post: async (endpoint, data) => {
    const response = await fetch(`${API_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    return response.json();
  },
  upload: async (endpoint, formData) => {
    const response = await fetch(`${API_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
    });
    return response.json();
  },
};
```

### 1.3 Update API Calls in Components

Update components to use the API utility instead of relative paths:

**Example in `client/src/pages/Analytics.jsx`:**
```js
import { api } from '../utils/api';

const apiCalls = {
  listImages: async () => api.get('/api/images'),
  // ... other calls
};
```

---

## Step 2: Deploy Backend First

Before deploying frontend, deploy your backend to Railway or Render:

### Option A: Railway (Recommended)

1. Go to [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Select `server` directory
4. Add PostgreSQL database
5. Set environment variables (see below)
6. Deploy

**You'll get**: `https://your-backend.railway.app`

### Option B: Render

1. Go to [render.com](https://render.com)
2. New Web Service
3. Connect GitHub → Select `server` directory
4. Set build/start commands
5. Add PostgreSQL database
6. Set environment variables
7. Deploy

**You'll get**: `https://your-backend.onrender.com`

---

## Step 3: Deploy Frontend to Netlify

### Method 1: Netlify Dashboard (Easiest)

1. **Sign up/Login** at [netlify.com](https://netlify.com)

2. **Add New Site** → **Import from Git**

3. **Connect to GitHub** and select your repository

4. **Build Settings**:
   - **Base directory**: `client`
   - **Build command**: `npm run build`
   - **Publish directory**: `client/dist`

5. **Environment Variables**:
   Click "Show advanced" → "New variable"
   ```
   VITE_API_URL = https://your-backend.railway.app
   ```
   (Update with your actual backend URL)

6. **Deploy site!**

### Method 2: Netlify CLI

1. **Install Netlify CLI**:
   ```bash
   npm install -g netlify-cli
   ```

2. **Login**:
   ```bash
   netlify login
   ```

3. **Initialize** (in project root):
   ```bash
   cd client
   netlify init
   ```
   - Choose "Create & configure a new site"
   - Site name: (choose a name)
   - Build command: `npm run build`
   - Publish directory: `dist`

4. **Set environment variable**:
   ```bash
   netlify env:set VITE_API_URL https://your-backend.railway.app
   ```

5. **Deploy**:
   ```bash
   netlify deploy --prod
   ```

---

## Step 4: Update netlify.toml

After deploying backend, update `client/netlify.toml`:

```toml
[[redirects]]
  from = "/api/*"
  to = "https://your-actual-backend.railway.app/api/:splat"
  status = 200
  force = true
```

Replace `your-actual-backend.railway.app` with your real backend URL.

---

## Step 5: Configure CORS on Backend

Update your backend `server/src/server.js` to allow Netlify domain:

```js
const ORIGIN = process.env.ORIGIN || 'http://localhost:5173';

app.use(cors({ 
  origin: [
    ORIGIN,
    'https://your-app.netlify.app',  // Add your Netlify URL
    /\.netlify\.app$/  // Allow all Netlify previews
  ],
  credentials: true
}));
```

Or use environment variable:
```env
ORIGIN=https://your-app.netlify.app
```

---

## Step 6: Deploy Python Flask API

Deploy to Railway or Render (same as backend):

1. **Railway**: Add another service → Select `python_processing` directory
2. **Set start command**: `gunicorn -w 4 -b 0.0.0.0:$PORT flask_api:app`
3. **Environment variables**: Same as backend (AWS S3 credentials)

---

## Environment Variables Summary

### Netlify (Frontend)
```
VITE_API_URL=https://your-backend.railway.app
```

### Railway/Render (Backend)
```
PORT=5050
ORIGIN=https://your-app.netlify.app
NODE_ENV=production
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket
DB_HOST=your-db-host
DB_PORT=5432
DB_NAME=drone_analytics
DB_USER=postgres
DB_PASSWORD=your-password
```

### Railway/Render (Python API)
```
FLASK_PORT=5001
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket
```

---

## Testing Deployment

1. **Visit your Netlify site**: `https://your-app.netlify.app`
2. **Check browser console** for errors
3. **Test image upload**
4. **Test map functionality**
5. **Verify API calls** in Network tab

---

## Custom Domain (Optional)

1. In Netlify dashboard → **Domain settings**
2. Click **Add custom domain**
3. Enter your domain
4. Follow DNS configuration instructions
5. Netlify provides free SSL automatically

---

## Continuous Deployment

Netlify automatically deploys when you push to GitHub:

- **Production**: Deploys from `main` branch
- **Preview**: Creates preview for pull requests
- **Branch deploys**: Deploy from any branch

---

## Troubleshooting

### CORS Errors
- Check `ORIGIN` in backend environment variables
- Verify Netlify URL is allowed in CORS config

### API Not Found (404)
- Check `VITE_API_URL` in Netlify environment variables
- Verify backend is running and accessible
- Check `netlify.toml` redirects

### Build Fails
- Check build logs in Netlify dashboard
- Verify Node.js version (Netlify uses Node 18 by default)
- Check for missing dependencies

### Images Not Loading
- Verify S3 bucket permissions
- Check if images are uploaded to S3
- Verify S3 URLs in API responses

---

## Cost

- **Netlify**: Free tier (100GB bandwidth, 300 build minutes/month)
- **Railway**: $5/month (includes database)
- **S3**: Pay per use (~$0.023/GB/month)

**Total**: ~$5-10/month

---

## Next Steps

1. ✅ Deploy backend to Railway/Render
2. ✅ Deploy frontend to Netlify
3. ✅ Set environment variables
4. ✅ Test all functionality
5. ✅ Set up custom domain (optional)
6. ✅ Configure monitoring

Your app will be live at: `https://your-app.netlify.app` 🎉

