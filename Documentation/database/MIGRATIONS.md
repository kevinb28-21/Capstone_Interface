# Database Migrations

Guide for running database migrations and schema updates.

## Available Migrations

### Add GNDVI Support

**File:** `python_processing/database_migration_add_gndvi.sql`

Adds GNDVI (Green Normalized Difference Vegetation Index) columns to support onion crop health analysis.

**Run:**
```bash
psql -U postgres -d drone_analytics -f python_processing/database_migration_add_gndvi.sql
```

**Changes:**
- Adds `gndvi`, `gndvi_mean`, `gndvi_std`, `gndvi_min`, `gndvi_max` to `analyses` table
- Adds `gndvi_value` to `stress_zones` table
- Creates index on `gndvi_mean`

## Running Migrations

### Initial Schema

```bash
psql -U postgres -d drone_analytics -f server/database/schema.sql
```

### Apply Migrations

```bash
# GNDVI migration
psql -U postgres -d drone_analytics -f python_processing/database_migration_add_gndvi.sql
```

## Migration Best Practices

1. **Backup First**
   ```bash
   pg_dump -U postgres drone_analytics > backup.sql
   ```

2. **Test on Development**
   - Run migrations on development database first
   - Verify schema changes
   - Test application functionality

3. **Production Deployment**
   - Schedule maintenance window if needed
   - Run migrations during low-traffic periods
   - Monitor for errors

4. **Rollback Plan**
   - Keep backup of previous schema
   - Document rollback procedures
   - Test rollback on development

## Verifying Migrations

### Check Column Exists

```sql
SELECT column_name 
FROM information_schema.columns 
WHERE table_name='analyses' AND column_name='gndvi_mean';
```

### Check Index Exists

```sql
SELECT indexname 
FROM pg_indexes 
WHERE tablename='analyses' AND indexname='idx_analyses_gndvi_mean';
```

## Related Documentation

- [Database Schema](./SCHEMA.md)
- [Python Processing](../python-processing/README.md)

