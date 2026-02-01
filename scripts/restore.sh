#!/bin/bash

# ==========================================
# Stock Predictor - Database Restore Script
# ==========================================

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
BACKUP_DIR="./backups"

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

DB_USER=${POSTGRES_USER:-stockuser}
DB_NAME=${POSTGRES_DB:-stockprediction}

# Check if backup date provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: No backup date specified${NC}"
    echo "Usage: ./scripts/restore.sh YYYYMMDD_HHMMSS"
    echo ""
    echo "Available backups:"
    ls -1 "$BACKUP_DIR"/postgres_*.sql.gz 2>/dev/null | sed 's/.*postgres_\(.*\)\.sql\.gz/  \1/'
    exit 1
fi

BACKUP_DATE=$1
POSTGRES_BACKUP="$BACKUP_DIR/postgres_${BACKUP_DATE}.sql.gz"

# Check if backup exists
if [ ! -f "$POSTGRES_BACKUP" ]; then
    echo -e "${RED}Error: Backup file not found: ${POSTGRES_BACKUP}${NC}"
    exit 1
fi

echo -e "${YELLOW}⚠️  WARNING: This will overwrite the current database!${NC}"
read -p "Are you sure you want to continue? (yes/no): " -r
echo

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Restore cancelled."
    exit 0
fi

# Stop dependent services
echo -e "${YELLOW}Stopping backend services...${NC}"
docker-compose stop backend celery-worker celery-beat

# Restore PostgreSQL
echo -e "${GREEN}Restoring PostgreSQL database...${NC}"
gunzip -c "$POSTGRES_BACKUP" | docker-compose exec -T postgres psql -U "$DB_USER" "$DB_NAME"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Database restored successfully${NC}"
else
    echo -e "${RED}✗ Database restore failed${NC}"
    exit 1
fi

# Restore Redis (if exists)
REDIS_BACKUP="$BACKUP_DIR/redis_${BACKUP_DATE}.rdb.gz"
if [ -f "$REDIS_BACKUP" ]; then
    echo -e "${GREEN}Restoring Redis...${NC}"
    gunzip -c "$REDIS_BACKUP" > /tmp/dump.rdb
    docker cp /tmp/dump.rdb stockpred-redis:/data/dump.rdb
    docker-compose restart redis
    rm /tmp/dump.rdb
    echo -e "${GREEN}✓ Redis restored${NC}"
fi

# Restore models (if exists)
MODELS_BACKUP="$BACKUP_DIR/models_${BACKUP_DATE}.tar.gz"
if [ -f "$MODELS_BACKUP" ]; then
    echo -e "${GREEN}Restoring ML models...${NC}"
    tar -xzf "$MODELS_BACKUP"
    echo -e "${GREEN}✓ Models restored${NC}"
fi

# Restart services
echo -e "${GREEN}Restarting services...${NC}"
docker-compose start backend celery-worker celery-beat

echo -e "${GREEN}✅ Restore completed successfully!${NC}"
exit 0