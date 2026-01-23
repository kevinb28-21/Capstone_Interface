# Quick Deployment Guide

## Fastest Way to Deploy (Railway - All-in-One)

### Step 1: Deploy Frontend (Vercel - 5 minutes)

1. Go to [vercel.com](https://vercel.com) and sign up
2. Click "New Project"
3. Import your GitHub repository
4. Set root directory to `client`
5. Build settings:
   - Framework Preset: Vite
   - Build Command: `npm run build`
   - Output Directory: `dist`
6. Add environment variable:
   - `VITE_API_URL` = (leave empty for now, add after backend deploy)
7. Deploy!

**You'll get**: `https://your-app.vercel.app`

---

### Step 2: Deploy Backend + Database (Railway - 10 minutes)

1. Go to [railway.app](https://railway.app) and sign up
2. Click "New Project" → "Deploy from GitHub"
3. Select your repository
4. Add service → Select `server` directory
5. Add PostgreSQL database:
   - Click "New" → "Database" → "PostgreSQL"
6. Set environment variables:
   ```
   PORT=5050
   ORIGIN=https://your-app.vercel.app
   DB_HOST=${{Postgres.PGHOST}}
   DB_PORT=${{Postgres.PGPORT}}
   DB_NAME=${{Postgres.PGDATABASE}}
   DB_USER=${{Postgres.PGUSER}}
   DB_PASSWORD=${{Postgres.PGPASSWORD}}
   AWS_ACCESS_KEY_ID=your-key
   AWS_SECRET_ACCESS_KEY=your-secret
   AWS_REGION=us-east-1
   S3_BUCKET_NAME=your-bucket
   ```
7. Deploy!

**You'll get**: `https://your-backend.railway.app`

---

### Step 3: Deploy Python API (Railway - 5 minutes)

1. In same Railway project, click "New" → "GitHub Repo"
2. Select `python_processing` directory
3. Set start command:
   ```
   pip install gunicorn && gunicorn -w 4 -b 0.0.0.0:$PORT flask_api:app
   ```
4. Add environment variables:
   ```
   AWS_ACCESS_KEY_ID=your-key
   AWS_SECRET_ACCESS_KEY=your-secret
   AWS_REGION=us-east-1
   S3_BUCKET_NAME=your-bucket
   ```
5. Deploy!

**You'll get**: `https://your-python-api.railway.app`

---

### Step 4: Update Frontend API URL

1. Go back to Vercel
2. Settings → Environment Variables
3. Add/Update:
   ```
   VITE_API_URL=https://your-backend.railway.app
   ```
4. Redeploy frontend

---

### Step 5: Run Database Migrations

1. In Railway, go to PostgreSQL service
2. Click "Connect" → Copy connection string
3. Run locally:
   ```bash
   psql "your-connection-string" -f server/database/schema.sql
   ```

Or use Railway's built-in PostgreSQL console.

---

## That's It! 🎉

Your app is now live at: `https://your-app.vercel.app`

---

## Cost

- **Vercel**: Free
- **Railway**: $5/month (includes database)
- **S3**: Pay per use (~$0.023/GB/month)

**Total**: ~$5-10/month

---

## Alternative: Render (Free Tier)

Same process but use [render.com](https://render.com):
- Free tier available (with limitations)
- Slower cold starts
- Good for testing

---

## Need Help?

See `DEPLOYMENT_GUIDE.md` for detailed instructions and troubleshooting.

