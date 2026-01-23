Deployment Guide - Drone Crop Health Platform

This guide covers deploying your full-stack application to the internet, including all components: React frontend, Node.js backend, Python Flask API, PostgreSQL database, and S3 storage.

Application Architecture

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

Deployment Options

Option 1: All-in-One Platform (Recommended for Start)

Railway or Render - Deploy everything together

Pros:
  - Simple setup
  - Free tier available
  - Automatic HTTPS
  - Environment variable management
  - PostgreSQL included

Cons:
  - Less control
  - Can be expensive at scale
  - Limited customization

Option 2: Separate Services (Recommended for Production)

Frontend: Vercel / Netlify  
Backend: Railway / Render / AWS EC2  
Python API: Railway / Render / AWS Lambda  
Database: AWS RDS / Railway PostgreSQL / Supabase  
Storage: AWS S3 (already configured)

Pros:
  - Best performance
  - Scalable
  - Cost-effective
  - Industry standard

Cons:
  - More complex setup
  - Multiple services to manage

Detailed Deployment Steps

Part 1: Frontend Deployment (React)

Option A: Vercel (Recommended)
  Install Vercel CLI:
      npm i -g vercel
  Build configuration (client/vercel.json):
      {
     "buildCommand": "npm run build",
     "outputDirectory": "dist",
     "rewrites": [
       {
         "source": "/api/(.)",
         "destination": "https://your-backend-url.com/api/$1"
       }
     ]
   }
  Deploy:
      cd client
   vercel
  Environment Variables (in Vercel dashboard):
  - VITEAPIURL=https://your-backend-url.com
  Update client/vite.config.js:
      export default {
     // ... existing config
     define: {
       'import.meta.env.VITEAPIURL': JSON.stringify(process.env.VITEAPIURL || 'http://localhost:5050')
     }
   }
   
Option B: Netlify
  Create client/netlify.toml:
      [build]
     command = "npm run build"
     publish = "dist"

   [[redirects]]
     from = "/api/"
     to = "https://your-backend-url.com/api/:splat"
     status = 200
  Deploy via Netlify dashboard or CLI:
      npm install -g netlify-cli
   cd client
   netlify deploy --prod
   

Part 2: Backend Deployment (Node.js)

Option A: Railway
  Connect GitHub repository to Railway
  Add service → Select server directory
  Set environment variables:
      PORT=5050
   ORIGIN=https://your-frontend-url.vercel.app
   AWSACCESSKEYID=your-key
   AWSSECRETACCESSKEY=your-secret
   AWSREGION=us-east-1
   S3BUCKETNAME=your-bucket
   DBHOST=your-db-host
   DBPORT=5432
   DBNAME=droneanalytics
   DBUSER=postgres
   DBPASSWORD=your-password
  Railway auto-detects Node.js and runs npm start

Option B: Render
  Create Web Service in Render dashboard
  Connect GitHub repository
  Settings:
  - Build Command: cd server && npm install
  - Start Command: cd server && npm start
  - Environment: Node
  Environment Variables (same as Railway)

Option C: AWS EC2
  Launch EC2 instance (Ubuntu 22.04)
  SSH into instance:
      ssh -i your-key.pem ubuntu@your-ec2-ip
  Install Node.js:
      curl -fsSL https://deb.nodesource.com/setup20.x | sudo -E bash -
   sudo apt-get install -y nodejs
  Clone repository:
      git clone https://github.com/your-username/your-repo.git
   cd your-repo/server
   npm install
  Install PM2:
      sudo npm install -g pm2
  Create PM2 ecosystem file (server/ecosystem.config.js):
      module.exports = {
     apps: [{
       name: 'drone-backend',
       script: 'src/server.js',
       env: {
         NODEENV: 'production',
         PORT: 5050
       }
     }]
   }
  Start with PM2:
      pm2 start ecosystem.config.js
   pm2 save
   pm2 startup
  Configure Nginx (reverse proxy):
      server {
       listen 80;
       servername your-domain.com;

       location / {
           proxypass http://localhost:5050;
           proxyhttpversion 1.1;
           proxysetheader Upgrade $httpupgrade;
           proxysetheader Connection 'upgrade';
           proxysetheader Host $host;
           proxycachebypass $httpupgrade;
       }
   }
  Install SSL with Let's Encrypt:
      sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   

Part 3: Python Flask API Deployment

Option A: Railway
  Add another service in Railway
  Select pythonprocessing directory
  Railway auto-detects Python
  Set start command:
      gunicorn -w 4 -b 0.0.0.0:$PORT flaskapi:app
  Add to pythonprocessing/requirements.txt:
      gunicorn==21.2.0
  Environment variables:
      FLASKPORT=5001
   AWSACCESSKEYID=your-key
   AWSSECRETACCESSKEY=your-secret
   AWSREGION=us-east-1
   S3BUCKETNAME=your-bucket
   
Option B: Render
  Create Web Service
  Settings:
  - Environment: Python 3
  - Build Command: pip install -r requirements.txt
  - Start Command: gunicorn -w 4 -b 0.0.0.0:$PORT flaskapi:app

Option C: AWS Lambda (Serverless)
  Use Zappa or Serverless Framework
  More complex but cost-effective for low traffic

Part 4: Database Deployment

Option A: Railway PostgreSQL
  Add PostgreSQL service in Railway
  Get connection string from service
  Run migrations:
      psql $DATABASEURL -f server/database/schema.sql
   
Option B: AWS RDS
  Create RDS PostgreSQL instance
  Configure security group (allow your backend IP)
  Get connection details and update environment variables

Option C: Supabase (Free tier available)
  Create project at supabase.com
  Get connection string
  Run migrations via Supabase SQL editor

Part 5: Update Frontend API URLs

After deploying backend, update frontend to use production URLs:

client/src/main.jsx or create client/.env.production:
VITEAPIURL=https://your-backend.railway.app

Update API calls to use environment variable:
const APIURL = import.meta.env.VITEAPIURL || 'http://localhost:5050';

Complete Deployment Checklist

Pre-Deployment
  - [ ] Build frontend locally: cd client && npm run build
  - [ ] Test all API endpoints
  - [ ] Verify S3 integration works
  - [ ] Set up PostgreSQL database
  - [ ] Run database migrations
  - [ ] Configure CORS for production domains
  - [ ] Set up environment variables

Deployment Steps
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

Post-Deployment
  - [ ] Test image upload
  - [ ] Verify S3 uploads work
  - [ ] Test map functionality
  - [ ] Check database connections
  - [ ] Monitor error logs
  - [ ] Set up backups
  - [ ] Configure auto-scaling (if needed)

Environment Variables Summary

Frontend (client/.env.production)
VITEAPIURL=https://your-backend-url.com

Backend (server/.env)
PORT=5050
ORIGIN=https://your-frontend-url.vercel.app
NODEENV=production
AWSACCESSKEYID=your-key
AWSSECRETACCESSKEY=your-secret
AWSREGION=us-east-1
S3BUCKETNAME=your-bucket
DBHOST=your-db-host
DBPORT=5432
DBNAME=droneanalytics
DBUSER=postgres
DBPASSWORD=your-password

Python API (pythonprocessing/.env)
FLASKPORT=5001
FLASKDEBUG=False
AWSACCESSKEYID=your-key
AWSSECRETACCESSKEY=your-secret
AWSREGION=us-east-1
S3BUCKET_NAME=your-bucket

Cost Estimates

Free Tier Options:
  - Vercel: Free for personal projects
  - Netlify: Free tier available
  - Railway: $5/month (includes database)
  - Render: Free tier (with limitations)
  - Supabase: Free tier (500MB database)
  - AWS S3: Free tier (5GB storage)

Production Costs (estimated):
  - Frontend: $0-20/month
  - Backend: $7-25/month
  - Python API: $7-25/month
  - Database: $0-15/month (free tier available)
  - S3 Storage: $0.023/GB/month
  - Total: ~$15-85/month

Recommended Setup for Production
  Frontend: Vercel (free, excellent performance)
  Backend: Railway ($5/month, includes PostgreSQL)
  Python API: Railway (same account, easy)
  Database: Railway PostgreSQL (included)
  Storage: AWS S3 (already configured)

Total: ~$5-10/month + S3 storage costs

Security Considerations
  Never commit .env files (already in .gitignore)
  Use environment variables in deployment platforms
  Enable HTTPS everywhere
  Configure CORS properly (only allow your frontend domain)
  Use AWS IAM roles instead of access keys when possible
  Set up rate limiting on APIs
  Enable database backups
  Use secrets management (AWS Secrets Manager, etc.)

Monitoring & Maintenance

Set Up Logging:
  - Railway/Render: Built-in logs
  - AWS CloudWatch: For EC2 deployments
  - Sentry: Error tracking (free tier)

Set Up Monitoring:
  - Uptime monitoring: UptimeRobot (free)
  - Performance: Vercel Analytics (free)
  - Database: Railway/Render dashboards

Regular Tasks:
  - Monitor S3 storage usage
  - Check database size
  - Review error logs
  - Update dependencies
  - Backup database regularly

Troubleshooting

Common Issues:
  CORS errors: Update ORIGIN in backend .env
  Database connection: Check connection string and security groups
  S3 upload fails: Verify AWS credentials and bucket permissions
  Frontend can't reach API: Check API URL in frontend config
  Build fails: Check Node.js/Python versions match deployment platform

Next Steps
  Choose your deployment platform(s)
  Set up accounts
  Deploy services one by one
  Test thoroughly
  Set up monitoring
  Configure custom domain (optional)

For specific platform help, refer to their documentation or ask for detailed steps for your chosen platform.