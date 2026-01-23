# Amazon S3 Integration Setup

## Information Needed

To integrate Amazon S3, please provide:

### 1. AWS Credentials
- **AWS Access Key ID**: `AKIA...`
- **AWS Secret Access Key**: `...`
- **AWS Region**: (e.g., `us-east-1`, `us-west-2`, `ca-central-1`)

### 2. S3 Bucket Details
- **Bucket Name**: (e.g., `drone-crop-health-images`)
- **Bucket Region**: (should match AWS Region above)

### 3. Optional Configuration
- **Bucket Folder Structure**: 
  - Default: `images/{year}/{month}/{day}/` for organization
  - Or flat structure: `images/`
- **Public vs Private**: 
  - Private (recommended): Use signed URLs for access
  - Public: Direct URL access (less secure)

## Setup Steps

Once you provide the above information:

1. **Create S3 Bucket** (if not already created):
   ```bash
   aws s3 mb s3://your-bucket-name --region your-region
   ```

2. **Set Bucket Policy** (for private access):
   - Allow your application to read/write
   - Restrict public access

3. **Configure Environment Variables**:
   - Add to `server/.env`
   - Add to `python_processing/.env`

## Data Flow with S3

```
Raspberry Pi → Flask API → Process Image → Upload to S3 → Store URL in PostgreSQL
                                    ↓
                            Node.js Backend → Serve S3 URLs to Frontend
```

## Files to Update

- `python_processing/flask_api.py` - Upload to S3 after processing
- `server/src/server.js` - Upload to S3, store URLs
- `server/src/s3-utils.js` - S3 upload utility (new)
- `python_processing/s3_utils.py` - S3 upload utility (new)

## Security Notes

- Never commit AWS credentials to git
- Use environment variables or AWS IAM roles
- Consider using AWS Secrets Manager for production

