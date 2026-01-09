# Netlify Bandwidth Optimization Guide

## Problem

Your application was making excessive API calls through Netlify's proxy, consuming bandwidth credits:
- **Home page**: Polling every 5 seconds
- **Analytics page**: Polling every 3 seconds  
- **Map page**: Polling every 3 seconds
- **ML page**: Polling every 5 seconds

With multiple tabs open, this created hundreds of requests per minute, all going through Netlify's proxy.

## Solutions Implemented

### ✅ 1. Increased Polling Intervals
All polling intervals have been increased from 3-5 seconds to **30 seconds**:
- Reduces API calls by **83-90%**
- Still provides near real-time updates
- Much more bandwidth-efficient

### ✅ 2. Page Visibility API
Polling now pauses when browser tabs are hidden:
- No API calls when users switch to other tabs
- Automatically resumes when tab becomes visible
- Refreshes data immediately when tab regains focus

### ✅ 3. Direct API Calls (Optional - Recommended)

To completely bypass Netlify proxy and avoid bandwidth charges:

#### Step 1: Set Environment Variable in Netlify

1. Go to **Netlify Dashboard** → Your Site → **Site settings** → **Environment variables**
2. Click **Add variable**
3. Add:
   - **Key**: `VITE_API_URL`
   - **Value**: `http://ec2-18-117-90-212.us-east-2.compute.amazonaws.com`
   - **Scopes**: Select **Production** (and **Deploy previews** if desired)

#### Step 2: Ensure CORS Allows Your Netlify Domain

Make sure your EC2 backend allows requests from your Netlify domain. Check your backend CORS configuration:

```javascript
// Example CORS config (adjust for your backend)
const allowedOrigins = [
  'https://your-app.netlify.app',
  'http://localhost:5173', // for local dev
];
```

#### Step 3: Redeploy

After setting the environment variable:
- Netlify will automatically redeploy, OR
- Manually trigger a redeploy from the Netlify dashboard

#### Step 4: Verify

After redeploy, check browser Network tab:
- API calls should go directly to `ec2-18-117-90-212.us-east-2.compute.amazonaws.com`
- No longer proxied through Netlify
- Netlify bandwidth usage should drop to near zero

## Impact

### Before Optimization:
- **Home page**: 12 requests/minute (2 endpoints × 6 times/min)
- **Analytics**: 20 requests/minute
- **Map**: 20 requests/minute  
- **ML**: 12 requests/minute
- **Total**: ~64 requests/minute = **3,840 requests/hour**
- All proxied through Netlify = **high bandwidth usage**

### After Optimization:
- **All pages**: 2 requests/minute (30-second intervals)
- **With visibility API**: 0 requests when tabs hidden
- **With direct API**: 0 Netlify bandwidth usage
- **Total**: ~2-8 requests/minute = **120-480 requests/hour**
- **Reduction**: **87-97% fewer requests**

## Current Status

✅ Polling intervals increased to 30 seconds  
✅ Page Visibility API implemented  
⏳ Direct API calls: **Set `VITE_API_URL` in Netlify to enable**

## Notes

- The `netlify.toml` redirects will still work as a fallback if `VITE_API_URL` is not set
- Setting `VITE_API_URL` bypasses Netlify proxy completely
- Your EC2 backend must allow CORS from your Netlify domain
- The API utility (`client/src/utils/api.js`) already supports `VITE_API_URL`






