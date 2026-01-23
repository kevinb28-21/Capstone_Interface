Development Guide

This guide covers setting up a development environment and contributing to the Drone Crop Health Platform.

Development Environment Setup

Prerequisites
  - Node.js v18+ and npm
  - Python 3.8+
  - PostgreSQL 12+
  - Git
  - Code editor (VS Code recommended)

Initial Setup
  Clone and Install

git clone <repository-url>
cd CapstoneInterface

Install backend dependencies
cd server
npm install

Install frontend dependencies
cd ../client
npm install

Install Python dependencies
cd ../pythonprocessing
pip install -r requirements.txt
  Environment Configuration

Create .env files in each directory:

server/.env
PORT=5000
NODEENV=development

pythonprocessing/.env
FLASKPORT=5001
FLASKDEBUG=True
DBHOST=localhost
DBPORT=5432
DBNAME=droneanalytics
DBUSER=postgres
DBPASSWORD=yourpassword
UPLOADFOLDER=./uploads
PROCESSEDFOLDER=./processed
WORKERPOLLINTERVAL=10
WORKERBATCHSIZE=5
  Database Setup

Create database
createdb droneanalytics

Run schema
psql -U postgres -d droneanalytics -f server/database/schema.sql

Run migrations
psql -U postgres -d droneanalytics -f pythonprocessing/databasemigrationaddgndvi.sql

Running Development Servers

Backend (Node.js)

cd server
npm run dev

Runs on http://localhost:5000 with hot reload.

Frontend (React)

cd client
npm run dev

Runs on http://localhost:5173 with hot reload.

Python Processing Service

cd pythonprocessing
python flaskapidb.py

Runs on http://localhost:5001.

Background Worker

cd pythonprocessing
python backgroundworker.py

Monitors database and processes images automatically.

Project Structure

CapstoneInterface/
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
├── pythonprocessing/      # Python image processing
│   ├── imageprocessor.py # Core processing functions
│   ├── flaskapidb.py    # Flask API with DB
│   ├── backgroundworker.py # Background processor
│   ├── dbutils.py         # Database utilities
│   ├── s3utils.py         # S3 utilities
│   └── requirements.txt
└── documentation/                   # Documentation

Development Workflow

Making Changes
  Create a feature branch
      git checkout -b feature/your-feature-name
  Make your changes
  - Follow existing code style
  - Add comments for complex logic
  - Update documentation if needed
  Test your changes
  - Test locally with all services running
  - Verify database migrations if schema changed
  - Check API endpoints
  Commit and push
      git add .
   git commit -m "Description of changes"
   git push origin feature/your-feature-name
   
Code Style
  - JavaScript/TypeScript: Follow ESLint configuration
  - Python: Follow PEP 8 style guide
  - SQL: Use uppercase for keywords
  - Comments: Document complex functions and algorithms

Testing

Manual Testing
  Image Upload
  - Upload an image via the frontend
  - Verify it appears in the database
  - Check S3 storage (if configured)
  Processing
  - Verify background worker processes images
  - Check analysis results in database
  - Verify vegetation indices are calculated
  API Endpoints
  - Test all endpoints with curl or Postman
  - Verify error handling
  - Check response formats

Testing Image Processing

cd pythonprocessing
python imageprocessor.py path/to/testimage.jpg

Batch Testing

cd pythonprocessing
python batchtestndvi.py /path/to/image/folder

Debugging

Backend Debugging
  - Use console.log() for Node.js debugging
  - Check server logs in terminal
  - Use Node.js debugger: node --inspect server.js

Python Debugging
  - Use print() or logging module
  - Enable Flask debug mode: FLASKDEBUG=True
  - Use Python debugger: python -m pdb script.py

Database Debugging

Connect to database
psql -U postgres -d droneanalytics

Check recent images
SELECT  FROM images ORDER BY uploadedat DESC LIMIT 10;

Check analyses
SELECT  FROM analyses ORDER BY processedat DESC LIMIT 10;

Common Issues

Port Conflicts

Change ports in .env files or kill processes:
Find process using port
lsof -i :5000

Kill process
kill -9 <PID>

Database Connection Errors
  - Verify PostgreSQL is running: pgisready
  - Check credentials in .env
  - Verify database exists: psql -l

Import Errors
  - Verify all dependencies are installed
  - Check Python/Node versions
  - Clear caches: npm cache clean --force or pip cache purge

Development Tools

Recommended VS Code Extensions
  - ESLint (JavaScript linting)
  - Python (Python support)
  - PostgreSQL (Database tools)
  - GitLens (Git integration)

Useful Commands

Format Python code
black pythonprocessing/*.py

Lint JavaScript
cd server && npm run lint

Check database schema
psql -U postgres -d droneanalytics -c "\d images"

Next Steps
  - API Documentation
  - Python Processing Guide
  - Database Schema