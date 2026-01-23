# Deployment Guide - Drone Crop Health Platform

This guide covers deploying your full-stack application to the internet, including all components: React frontend, Node.js backend, Python Flask API, PostgreSQL database, and S3 storage.

## Application Architecture

```
┌─────────────────┐
│  React Frontend │ (Static files)
└────────┬────────┘
         │
┌────────▼────────┐
│ Node.js Backend │ (Express API)
└────────┬────────┘
         │
    ┌────┴────┬──────────────┐
    │         │              │
┌───▼───┐ ┌──▼───┐    ┌─────▼─────┐
│ Flask │ │PostgreSQL│  │   S3      │
│  API  │ │ Database │  │  Storage  │
└───────┘ └─────────┘  └───────────┘
```

## Deployment Options

### Option 1: All-in-One Platform (Recommended for Start)

**Railway** or **Render** - Deploy everything together

#### Pros:
- Simple setup
- Free tier available
- Automatic HTTPS
- Environment variable management
- PostgreSQL included

#### Cons:
- Less control
- Can be expensive at scale
- Limited customization

---

### Option 2: Separate Services (Recommended for Production)

**Frontend**: Vercel / Netlify  
**Backend**: Railway / Render / AWS EC2  
**Python API**: Railway / Render / AWS Lambda  
**Database**: AWS RDS / Railway PostgreSQL / Supabase  
**Storage**: AWS S3 (already configured)

#### Pros:
- Best performance
- Scalable
- Cost-effective
- Industry standard

#### Cons:
- More complex setup
- Multiple services to manage

---

## Detailed Deployment Steps

### Part 1: Frontend Deployment (React)

#### Option A: Vercel (Recommended)

1. **Install Vercel CLI:**
   ```bash
   npm i -g vercel
   ```

2. **Build configuration** (`client/vercel.json`):
   ```json
   {
     "buildCommand": "npm run build",
     "outputDirectory": "dist",
     "rewrites": [
       {
         "source": "/api/(.*)",
         "destination": "https://your-backend-url.com/api/$1"
       }
     ]
   }
   ```

3. **Deploy:**
   ```bash
   cd client
   vercel
   ```

4. **Environment Variables** (in Vercel dashboard):
   - `VITE_API_URL=https://your-backend-url.com`

5. **Update `client/vite.config.js`**:
   ```js
   export default {
     // ... existing config
     define: {
       'import.meta.env.VITE_API_URL': JSON.stringify(process.env.VITE_API_URL || 'http://localhost:5050')
     }
   }
   ```

#### Option B: Netlify

1. **Create `client/netlify.toml`**:
   ```toml
   [build]
     command = "npm run build"
     publish = "dist"

   [[redirects]]
     from = "/api/*"
     to = "https://your-backend-url.com/api/:splat"
     status = 200
   ```

2. **Deploy via Netlify dashboard** or CLI:
   ```bash
   npm install -g netlify-cli
   cd client
   netlify deploy --prod
   ```

---

### Part 2: Backend Deployment (Node.js)

#### Option A: Railway

1. **Connect GitHub repository** to Railway
2. **Add service** → Select `server` directory
3. **Set environment variables:**
   ```env
   PORT=5050
   ORIGIN=https://your-frontend-url.vercel.app
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

4. **Railway auto-detects** Node.js and runs `npm start`

#### Option B: Render

1. **Create Web Service** in Render dashboard
2. **Connect GitHub repository**
3. **Settings:**
   - Build Command: `cd server && npm install`
   - Start Command: `cd server && npm start`
   - Environment: `Node`

4. **Environment Variables** (same as Railway)

#### Option C: AWS EC2

1. **Launch EC2 instance** (Ubuntu 22.04)
2. **SSH into instance:**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

3. **Install Node.js:**
   ```bash
   curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
   sudo apt-get install -y nodejs
   ```

4. **Clone repository:**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo/server
   npm install
   ```

5. **Install PM2:**
   ```bash
   sudo npm install -g pm2
   ```

6. **Create PM2 ecosystem file** (`server/ecosystem.config.js`):
   ```js
   module.exports = {
     apps: [{
       name: 'drone-backend',
       script: 'src/server.js',
       env: {
         NODE_ENV: 'production',
         PORT: 5050
       }
     }]
   }
   ```

7. **Start with PM2:**
   ```bash
   pm2 start ecosystem.config.js
   pm2 save
   pm2 startup
   ```

8. **Configure Nginx** (reverse proxy):
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:5050;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection 'upgrade';
           proxy_set_header Host $host;
           proxy_cache_bypass $http_upgrade;
       }
   }
   ```

9. **Install SSL with Let's Encrypt:**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

---

### Part 3: Python Flask API Deployment

#### Option A: Railway

1. **Add another service** in Railway
2. **Select `python_processing` directory**
3. **Railway auto-detects** Python
4. **Set start command:**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:$PORT flask_api:app
   ```

5. **Add to `python_processing/requirements.txt`:**
   ```
   gunicorn==21.2.0
   ```

6. **Environment variables:**
   ```env
   FLASK_PORT=5001
   AWS_ACCESS_KEY_ID=your-key
   AWS_SECRET_ACCESS_KEY=your-secret
   AWS_REGION=us-east-1
   S3_BUCKET_NAME=your-bucket
   ```

#### Option B: Render

1. **Create Web Service**
2. **Settings:**
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn -w 4 -b 0.0.0.0:$PORT flask_api:app`

#### Option C: AWS Lambda (Serverless)

1. **Use Zappa or Serverless Framework**
2. **More complex but cost-effective** for low traffic

---

### Part 4: Database Deployment

#### Option A: Railway PostgreSQL

1. **Add PostgreSQL service** in Railway
2. **Get connection string** from service
3. **Run migrations:**
   ```bash
   psql $DATABASE_URL -f server/database/schema.sql
   ```

#### Option B: AWS RDS

1. **Create RDS PostgreSQL instance**
2. **Configure security group** (allow your backend IP)
3. **Get connection details** and update environment variables

#### Option C: Supabase (Free tier available)

1. **Create project** at supabase.com
2. **Get connection string**
3. **Run migrations** via Supabase SQL editor

---

### Part 5: Update Frontend API URLs

After deploying backend, update frontend to use production URLs:

**`client/src/main.jsx` or create `client/.env.production`:**
```env
VITE_API_URL=https://your-backend.railway.app
```

**Update API calls** to use environment variable:
```js
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5050';
```

---

## Complete Deployment Checklist

### Pre-Deployment

- [ ] Build frontend locally: `cd client && npm run build`
- [ ] Test all API endpoints
- [ ] Verify S3 integration works
- [ ] Set up PostgreSQL database
- [ ] Run database migrations
- [ ] Configure CORS for production domains
- [ ] Set up environment variables

### Deployment Steps

- [ ] Deploy frontend (Vercel/Netlify)
- [ ] Deploy Node.js backend (Railway/Render/EC2)
- [ ] Deploy Python Flask API (Railway/Render)
- [ ] Set up PostgreSQL database
- [ ] Configure S3 bucket (already done)
- [ ] Update frontend API URLs
- [ ] Test end-to-end flow
- [ ] Set up custom domain (optional)
- [ ] Configure SSL/HTTPS
- [ ] Set up monitoring/logging

### Post-Deployment

- [ ] Test image upload
- [ ] Verify S3 uploads work
- [ ] Test map functionality
- [ ] Check database connections
- [ ] Monitor error logs
- [ ] Set up backups
- [ ] Configure auto-scaling (if needed)

---

## Environment Variables Summary

### Frontend (`client/.env.production`)
```env
VITE_API_URL=https://your-backend-url.com
```

### Backend (`server/.env`)
```env
PORT=5050
ORIGIN=https://your-frontend-url.vercel.app
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

### Python API (`python_processing/.env`)
```env
FLASK_PORT=5001
FLASK_DEBUG=False
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket
```

---

## Cost Estimates

### Free Tier Options:
- **Vercel**: Free for personal projects
- **Netlify**: Free tier available
- **Railway**: $5/month (includes database)
- **Render**: Free tier (with limitations)
- **Supabase**: Free tier (500MB database)
- **AWS S3**: Free tier (5GB storage)

### Production Costs (estimated):
- **Frontend**: $0-20/month
- **Backend**: $7-25/month
- **Python API**: $7-25/month
- **Database**: $0-15/month (free tier available)
- **S3 Storage**: $0.023/GB/month
- **Total**: ~$15-85/month

---

## Recommended Setup for Production

1. **Frontend**: Vercel (free, excellent performance)
2. **Backend**: Railway ($5/month, includes PostgreSQL)
3. **Python API**: Railway (same account, easy)
4. **Database**: Railway PostgreSQL (included)
5. **Storage**: AWS S3 (already configured)

**Total**: ~$5-10/month + S3 storage costs

---

## Security Considerations

1. **Never commit `.env` files** (already in `.gitignore`)
2. **Use environment variables** in deployment platforms
3. **Enable HTTPS** everywhere
4. **Configure CORS** properly (only allow your frontend domain)
5. **Use AWS IAM roles** instead of access keys when possible
6. **Set up rate limiting** on APIs
7. **Enable database backups**
8. **Use secrets management** (AWS Secrets Manager, etc.)

---

## Monitoring & Maintenance

### Set Up Logging:
- **Railway/Render**: Built-in logs
- **AWS CloudWatch**: For EC2 deployments
- **Sentry**: Error tracking (free tier)

### Set Up Monitoring:
- **Uptime monitoring**: UptimeRobot (free)
- **Performance**: Vercel Analytics (free)
- **Database**: Railway/Render dashboards

### Regular Tasks:
- Monitor S3 storage usage
- Check database size
- Review error logs
- Update dependencies
- Backup database regularly

---

## Troubleshooting

### Common Issues:

1. **CORS errors**: Update `ORIGIN` in backend `.env`
2. **Database connection**: Check connection string and security groups
3. **S3 upload fails**: Verify AWS credentials and bucket permissions
4. **Frontend can't reach API**: Check API URL in frontend config
5. **Build fails**: Check Node.js/Python versions match deployment platform

---

## Next Steps

1. Choose your deployment platform(s)
2. Set up accounts
3. Deploy services one by one
4. Test thoroughly
5. Set up monitoring
6. Configure custom domain (optional)

For specific platform help, refer to their documentation or ask for detailed steps for your chosen platform.

