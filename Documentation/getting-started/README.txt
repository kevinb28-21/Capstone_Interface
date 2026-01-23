Getting Started

This guide will help you get the Drone Crop Health Platform up and running on your local machine.

Prerequisites

Before you begin, ensure you have the following installed:
  - Node.js (v18 or higher) - Download
  - Python (v3.8 or higher) - Download
  - PostgreSQL (v12 or higher) - Download
  - Git - Download

Quick Start
  Clone the Repository

git clone <repository-url>
cd CapstoneInterface
  Backend Setup (Node.js)

cd server
npm install
cp .env.example .env  # Edit .env with your configuration
npm run dev

The server will start on http://localhost:5000 by default.
  Frontend Setup (React)

cd client
npm install
npm run dev

The frontend will be available at http://localhost:5173 (or the URL shown in terminal).
  Python Processing Service

cd pythonprocessing
pip install -r requirements.txt
cp .env.example .env  # Edit .env with your configuration
python flaskapidb.py

The Flask API will start on http://localhost:5001 by default.
  Database Setup

Create database
createdb droneanalytics

Run schema
psql -U postgres -d droneanalytics -f server/database/schema.sql

Run migrations (if any)
psql -U postgres -d droneanalytics -f pythonprocessing/databasemigrationaddgndvi.sql

Environment Variables

Backend (.env in server/)

PORT=5000
NODEENV=development

Python Processing (.env in pythonprocessing/)

FLASKPORT=5001
FLASKDEBUG=True
DBHOST=localhost
DBPORT=5432
DBNAME=droneanalytics
DBUSER=postgres
DBPASSWORD=yourpassword
UPLOADFOLDER=./uploads
PROCESSEDFOLDER=./processed

Verify Installation
  Backend: Visit http://localhost:5000/api/health - should return {"status":"ok"}
  Frontend: Visit http://localhost:5173 - should show the dashboard
  Flask API: Visit http://localhost:5001/api/health - should return status

Next Steps
  - Read the Development Guide for detailed setup
  - Check the API Documentation for endpoint details
  - Review the Python Processing Guide for image processing

Troubleshooting

Port Already in Use

If a port is already in use, you can change it:
  - Backend: Set PORT in server/.env
  - Frontend: Modify vite.config.js or use npm run dev -- --port <port>
  - Flask: Set FLASKPORT in pythonprocessing/.env

Database Connection Issues
  - Ensure PostgreSQL is running: pgisready
  - Check credentials in .env files
  - Verify database exists: psql -l

Python Dependencies

If you encounter import errors:
cd pythonprocessing
pip install -r requirements.txt --upgrade

Additional Resources
  - Project Overview - High-level system architecture
  - Deployment Guide - Production deployment
  - Database Schema - Database structure