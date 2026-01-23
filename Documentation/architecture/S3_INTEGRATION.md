# Amazon S3 Integration Guide

## Overview

The application now supports Amazon S3 for scalable image storage. Images are uploaded to S3 after processing, and S3 URLs are stored in the database instead of local file paths.

## Data Flow

```
Raspberry Pi → Flask API → Process Image → Upload to S3 → Store URL in PostgreSQL
                                    ↓
                            Node.js Backend → Serve S3 URLs to Frontend
```

## What You Need to Provide

### 1. AWS Credentials

You'll need:
- **AWS Access Key ID**: `AKIA...`
- **AWS Secret Access Key**: `...`
- **AWS Region**: (e.g., `us-east-1`, `us-west-2`, `ca-central-1`)

**How to get credentials:**
1. Log into AWS Console
2. Go to IAM → Users → Your User → Security Credentials
3. Create Access Key
4. Save the Access Key ID and Secret Access Key securely

### 2. S3 Bucket

**Bucket Name**: (e.g., `drone-crop-health-images`)

**Create bucket:**
```bash
aws s3 mb s3://your-bucket-name --region your-region
```

**Or via AWS Console:**
1. Go to S3 → Create bucket
2. Choose a unique name
3. Select your region
4. Configure permissions (recommend private with IAM access)

## Setup Instructions

### Step 1: Configure Environment Variables

#### For Node.js Server (`server/.env`):

```env
# Server Configuration
PORT=5050
ORIGIN=http://localhost:5180
UPLOAD_DIR=./uploads

# AWS S3 Configuration
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket-name
```

#### For Python Flask API (`python_processing/.env`):

```env
# Flask Server Configuration
FLASK_PORT=5001
FLASK_DEBUG=False

# Upload Directories
UPLOAD_FOLDER=./uploads
PROCESSED_FOLDER=./processed

# AWS S3 Configuration
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket-name
```

### Step 2: Set Up S3 Bucket Permissions

Create an IAM policy for your application:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::your-bucket-name/*"
    },
    {
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::your-bucket-name"
    }
  ]
}
```

Attach this policy to your IAM user.

### Step 3: Test the Integration

1. **Start the servers:**
   ```bash
   # Terminal 1: Node.js backend
   cd server && npm run dev
   
   # Terminal 2: Flask API
   cd python_processing && python flask_api.py
   ```

2. **Check S3 status:**
   - Node.js server will show: `✓ S3 storage enabled: your-bucket-name`
   - If not configured: `⚠️  S3 storage disabled (using local storage)`

3. **Upload an image:**
   - Use the web interface or send a POST request
   - Check S3 bucket for uploaded files
   - Verify the response includes `s3Url` field

## How It Works

### Image Upload Flow

1. **Image received** → Saved temporarily to local disk
2. **Image processed** → OpenCV/TensorFlow analysis
3. **Upload to S3** → File uploaded to `images/YYYY/MM/DD/filename.jpg`
4. **Store URL** → S3 URL saved in database (or in-memory store)
5. **Cleanup** → Local file can be deleted (optional)

### S3 Folder Structure

Images are organized by date:
```
your-bucket-name/
  ├── images/
  │   ├── 2024/
  │   │   ├── 11/
  │   │   │   ├── 07/
  │   │   │   │   ├── 1234567890_image1.jpg
  │   │   │   │   └── 1234567891_image2.jpg
  │   │   │   └── 08/
  │   │   │       └── ...
  │   │   └── 12/
  │   └── 2025/
  └── processed/
      └── (processed images with NDVI overlays)
```

### Fallback Behavior

If S3 is **not configured**:
- Images are stored locally in `./uploads`
- URLs use local paths: `/uploads/filename.jpg`
- All functionality works the same

## API Response Format

With S3 enabled:
```json
{
  "id": "abc123",
  "filename": "1234567890_image.jpg",
  "path": "https://bucket.s3.region.amazonaws.com/images/2024/11/07/1234567890_image.jpg",
  "s3Url": "https://bucket.s3.region.amazonaws.com/images/2024/11/07/1234567890_image.jpg",
  "s3Key": "images/2024/11/07/1234567890_image.jpg",
  "s3Stored": true,
  "analysis": { ... }
}
```

Without S3:
```json
{
  "id": "abc123",
  "filename": "1234567890_image.jpg",
  "path": "/uploads/1234567890_image.jpg",
  "s3Url": null,
  "s3Key": null,
  "s3Stored": false,
  "analysis": { ... }
}
```

## Security Best Practices

1. **Never commit credentials** to git
   - Use `.env` files (already in `.gitignore`)
   - Use AWS Secrets Manager for production

2. **Use IAM roles** (for EC2/ECS deployments)
   - More secure than access keys
   - Automatically rotated

3. **Bucket permissions**
   - Keep bucket private
   - Use signed URLs for temporary access if needed
   - Enable versioning for important data

4. **Access key rotation**
   - Rotate keys regularly
   - Use separate keys for dev/prod

## Troubleshooting

### "S3 storage disabled"
- Check that all environment variables are set
- Verify credentials are correct
- Check AWS region matches bucket region

### "Access Denied" errors
- Verify IAM policy allows PutObject/GetObject
- Check bucket policy doesn't block your user
- Ensure credentials have correct permissions

### Images not appearing
- Check S3 bucket for uploaded files
- Verify `s3Url` in API response
- Check browser console for CORS errors (if bucket is private)

### High costs
- Enable S3 lifecycle policies to move old images to Glacier
- Delete old/unused images
- Use S3 Intelligent-Tiering for automatic cost optimization

## Next Steps

1. **Set up environment variables** with your AWS credentials
2. **Create S3 bucket** if not already created
3. **Test upload** to verify integration
4. **Monitor S3 usage** in AWS Console
5. **Set up lifecycle policies** for cost optimization

## Support

For issues:
- Check server logs for S3 errors
- Verify AWS credentials in AWS Console
- Test S3 access with AWS CLI: `aws s3 ls s3://your-bucket-name`

