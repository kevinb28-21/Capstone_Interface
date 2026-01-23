# Netlify Quick Start Guide

## Prerequisites

1. **Backend deployed** (Railway/Render) - Get your backend URL
2. **GitHub repository** with your code
3. **Netlify account** (free at netlify.com)

---

## Step-by-Step Deployment

### Step 1: Deploy Backend First (if not done)

**Railway** (Recommended):
1. Go to [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Select `server` directory
4. Add PostgreSQL database
5. Set environment variables
6. Deploy → Copy your backend URL

**You'll get**: `https://your-backend.railway.app`

---

### Step 2: Deploy Frontend to Netlify

#### Option A: Via Dashboard (Easiest)

1. **Go to Netlify**: [app.netlify.com](https://app.netlify.com)

2. **Add New Site** → **Import from Git**

3. **Connect GitHub** → Select your repository

4. **Configure Build Settings**:
   ```
   Base directory: client
   Build command: npm run build
   Publish directory: client/dist
   ```

5. **Environment Variables**:
   - Click "Show advanced" → "New variable"
   - Key: `VITE_API_URL`
   - Value: `https://your-backend.railway.app` (your actual backend URL)

6. **Deploy site!**

#### Option B: Via CLI

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login
netlify login

# Navigate to client directory
cd client

# Initialize
netlify init
# Choose: Create & configure a new site
# Build command: npm run build
# Publish directory: dist

# Set environment variable
netlify env:set VITE_API_URL https://your-backend.railway.app

# Deploy
netlify deploy --prod
```

---

### Step 3: Update netlify.toml

Edit `client/netlify.toml` and replace the backend URL:

```toml
[[redirects]]
  from = "/api/*"
  to = "https://YOUR-ACTUAL-BACKEND.railway.app/api/:splat"
  status = 200
  force = true
```

---

### Step 4: Update Backend CORS

In your backend environment variables (Railway/Render), add:

```
ORIGIN=https://your-app.netlify.app
```

Or update `server/src/server.js` to allow Netlify:

```js
app.use(cors({ 
  origin: [
    process.env.ORIGIN,
    /\.netlify\.app$/  // Allow all Netlify domains
  ]
}));
```

---

### Step 5: Redeploy

1. **Commit and push** your changes
2. **Netlify will auto-deploy** (or trigger manually)
3. **Test your site**: `https://your-app.netlify.app`

---

## Testing

1. ✅ Visit your Netlify URL
2. ✅ Check browser console (F12) for errors
3. ✅ Test image upload
4. ✅ Test map functionality
5. ✅ Verify API calls work

---

## Troubleshooting

### CORS Errors
- Check backend `ORIGIN` environment variable includes your Netlify URL
- Verify CORS allows `*.netlify.app`

### API 404 Errors
- Check `netlify.toml` redirects have correct backend URL
- Verify backend is running and accessible
- Check `VITE_API_URL` environment variable in Netlify

### Build Fails
- Check build logs in Netlify dashboard
- Verify all dependencies are in `package.json`
- Check Node.js version (Netlify uses 18 by default)

---

## Your Site is Live! 🎉

**URL**: `https://your-app.netlify.app`

Netlify provides:
- ✅ Free SSL/HTTPS
- ✅ Automatic deployments from GitHub
- ✅ Preview deployments for PRs
- ✅ CDN for fast global access

---

## Next Steps

- Set up custom domain (optional)
- Configure monitoring
- Set up backups
- Review `NETLIFY_DEPLOYMENT.md` for advanced options

