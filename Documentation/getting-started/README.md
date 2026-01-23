# Getting Started

This guide will help you get the Drone Crop Health Platform up and running on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** (v18 or higher) - [Download](https://nodejs.org/)
- **Python** (v3.8 or higher) - [Download](https://www.python.org/downloads/)
- **PostgreSQL** (v12 or higher) - [Download](https://www.postgresql.org/download/)
- **Git** - [Download](https://git-scm.com/downloads)

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Capstone_Interface
```

### 2. Backend Setup (Node.js)

```bash
cd server
npm install
cp .env.example .env  # Edit .env with your configuration
npm run dev
```

The server will start on `http://localhost:5000` by default.

### 3. Frontend Setup (React)

```bash
cd client
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173` (or the URL shown in terminal).

### 4. Python Processing Service

```bash
cd python_processing
pip install -r requirements.txt
cp .env.example .env  # Edit .env with your configuration
python flask_api_db.py
```

The Flask API will start on `http://localhost:5001` by default.

### 5. Database Setup

```bash
# Create database
createdb drone_analytics

# Run schema
psql -U postgres -d drone_analytics -f server/database/schema.sql

# Run migrations (if any)
psql -U postgres -d drone_analytics -f python_processing/database_migration_add_gndvi.sql
```

## Environment Variables

### Backend (.env in `server/`)

```env
PORT=5000
NODE_ENV=development
```

### Python Processing (.env in `python_processing/`)

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
```

## Verify Installation

1. **Backend**: Visit `http://localhost:5000/api/health` - should return `{"status":"ok"}`
2. **Frontend**: Visit `http://localhost:5173` - should show the dashboard
3. **Flask API**: Visit `http://localhost:5001/api/health` - should return status

## Next Steps

- Read the [Development Guide](../development/README.md) for detailed setup
- Check the [API Documentation](../api/README.md) for endpoint details
- Review the [Python Processing Guide](../python-processing/README.md) for image processing

## Troubleshooting

### Port Already in Use

If a port is already in use, you can change it:
- Backend: Set `PORT` in `server/.env`
- Frontend: Modify `vite.config.js` or use `npm run dev -- --port <port>`
- Flask: Set `FLASK_PORT` in `python_processing/.env`

### Database Connection Issues

- Ensure PostgreSQL is running: `pg_isready`
- Check credentials in `.env` files
- Verify database exists: `psql -l`

### Python Dependencies

If you encounter import errors:
```bash
cd python_processing
pip install -r requirements.txt --upgrade
```

## Additional Resources

- [Project Overview](./PROJECT_OVERVIEW.md) - High-level system architecture
- [Deployment Guide](../deployment/README.md) - Production deployment
- [Database Schema](../database/SCHEMA.md) - Database structure

