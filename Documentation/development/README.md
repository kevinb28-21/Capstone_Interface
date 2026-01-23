# Development Guide

This guide covers setting up a development environment and contributing to the Drone Crop Health Platform.

## Development Environment Setup

### Prerequisites

- Node.js v18+ and npm
- Python 3.8+
- PostgreSQL 12+
- Git
- Code editor (VS Code recommended)

### Initial Setup

1. **Clone and Install**

```bash
git clone <repository-url>
cd Capstone_Interface

# Install backend dependencies
cd server
npm install

# Install frontend dependencies
cd ../client
npm install

# Install Python dependencies
cd ../python_processing
pip install -r requirements.txt
```

2. **Environment Configuration**

Create `.env` files in each directory:

**server/.env**
```env
PORT=5000
NODE_ENV=development
```

**python_processing/.env**
```env
FLASK_PORT=5001
FLASK_DEBUG=True
DB_HOST=localhost
DB_PORT=5432
DB_NAME=drone_analytics
DB_USER=postgres
DB_PASSWORD=your_password
UPLOAD_FOLDER=./uploads
PROCESSED_FOLDER=./processed
WORKER_POLL_INTERVAL=10
WORKER_BATCH_SIZE=5
```

3. **Database Setup**

```bash
# Create database
createdb drone_analytics

# Run schema
psql -U postgres -d drone_analytics -f server/database/schema.sql

# Run migrations
psql -U postgres -d drone_analytics -f python_processing/database_migration_add_gndvi.sql
```

## Running Development Servers

### Backend (Node.js)

```bash
cd server
npm run dev
```

Runs on `http://localhost:5000` with hot reload.

### Frontend (React)

```bash
cd client
npm run dev
```

Runs on `http://localhost:5173` with hot reload.

### Python Processing Service

```bash
cd python_processing
python flask_api_db.py
```

Runs on `http://localhost:5001`.

### Background Worker

```bash
cd python_processing
python background_worker.py
```

Monitors database and processes images automatically.

## Project Structure

```
Capstone_Interface/
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   └── utils/         # Utility functions
│   └── package.json
├── server/                 # Node.js backend
│   ├── src/
│   │   ├── server.js      # Express server
│   │   └── s3-utils.js    # S3 utilities
│   ├── database/
│   │   └── schema.sql     # Database schema
│   └── package.json
├── python_processing/      # Python image processing
│   ├── image_processor.py # Core processing functions
│   ├── flask_api_db.py    # Flask API with DB
│   ├── background_worker.py # Background processor
│   ├── db_utils.py         # Database utilities
│   ├── s3_utils.py         # S3 utilities
│   └── requirements.txt
└── documentation/                   # Documentation
```

## Development Workflow

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation if needed

3. **Test your changes**
   - Test locally with all services running
   - Verify database migrations if schema changed
   - Check API endpoints

4. **Commit and push**
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature/your-feature-name
   ```

### Code Style

- **JavaScript/TypeScript**: Follow ESLint configuration
- **Python**: Follow PEP 8 style guide
- **SQL**: Use uppercase for keywords
- **Comments**: Document complex functions and algorithms

## Testing

### Manual Testing

1. **Image Upload**
   - Upload an image via the frontend
   - Verify it appears in the database
   - Check S3 storage (if configured)

2. **Processing**
   - Verify background worker processes images
   - Check analysis results in database
   - Verify vegetation indices are calculated

3. **API Endpoints**
   - Test all endpoints with curl or Postman
   - Verify error handling
   - Check response formats

### Testing Image Processing

```bash
cd python_processing
python image_processor.py path/to/test_image.jpg
```

### Batch Testing

```bash
cd python_processing
python batch_test_ndvi.py /path/to/image/folder
```

## Debugging

### Backend Debugging

- Use `console.log()` for Node.js debugging
- Check server logs in terminal
- Use Node.js debugger: `node --inspect server.js`

### Python Debugging

- Use `print()` or `logging` module
- Enable Flask debug mode: `FLASK_DEBUG=True`
- Use Python debugger: `python -m pdb script.py`

### Database Debugging

```bash
# Connect to database
psql -U postgres -d drone_analytics

# Check recent images
SELECT * FROM images ORDER BY uploaded_at DESC LIMIT 10;

# Check analyses
SELECT * FROM analyses ORDER BY processed_at DESC LIMIT 10;
```

## Common Issues

### Port Conflicts

Change ports in `.env` files or kill processes:
```bash
# Find process using port
lsof -i :5000

# Kill process
kill -9 <PID>
```

### Database Connection Errors

- Verify PostgreSQL is running: `pg_isready`
- Check credentials in `.env`
- Verify database exists: `psql -l`

### Import Errors

- Verify all dependencies are installed
- Check Python/Node versions
- Clear caches: `npm cache clean --force` or `pip cache purge`

## Development Tools

### Recommended VS Code Extensions

- ESLint (JavaScript linting)
- Python (Python support)
- PostgreSQL (Database tools)
- GitLens (Git integration)

### Useful Commands

```bash
# Format Python code
black python_processing/*.py

# Lint JavaScript
cd server && npm run lint

# Check database schema
psql -U postgres -d drone_analytics -c "\d images"
```

## Next Steps

- [API Documentation](../api/README.md)
- [Python Processing Guide](../python-processing/README.md)
- [Database Schema](../database/SCHEMA.md)

