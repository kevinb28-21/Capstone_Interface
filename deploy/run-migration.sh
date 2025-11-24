#!/bin/bash
# Run database migration on EC2
# Usage: ./run-migration.sh

set -e

echo "Running GNDVI database migration..."

# Load environment variables
if [ -f ~/Capstone_Interface/python_processing/.env ]; then
    export $(grep -v '^#' ~/Capstone_Interface/python_processing/.env | xargs)
fi

# Check if required variables are set
if [ -z "$DB_NAME" ] || [ -z "$DB_USER" ]; then
    echo "Error: DB_NAME and DB_USER must be set in python_processing/.env"
    exit 1
fi

# Set PGPASSWORD if DB_PASSWORD is set
if [ -n "$DB_PASSWORD" ]; then
    export PGPASSWORD="$DB_PASSWORD"
fi

# Run the migration
MIGRATION_FILE=~/Capstone_Interface/python_processing/database_migration_add_gndvi.sql

if [ ! -f "$MIGRATION_FILE" ]; then
    echo "Error: Migration file not found: $MIGRATION_FILE"
    exit 1
fi

echo "Running migration: $MIGRATION_FILE"
psql -U "$DB_USER" -d "$DB_NAME" -h "${DB_HOST:-localhost}" -f "$MIGRATION_FILE"

echo "✓ Migration completed successfully"

# Verify the migration
echo ""
echo "Verifying migration..."
psql -U "$DB_USER" -d "$DB_NAME" -h "${DB_HOST:-localhost}" -c "
SELECT column_name 
FROM information_schema.columns 
WHERE table_name='analyses' AND column_name LIKE 'gndvi%'
ORDER BY column_name;
"

echo ""
echo "✓ Migration verified"

