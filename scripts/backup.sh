#!/bin/bash

# ==========================================
# Stock Predictor - Database Backup Script
# ==========================================

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
BACKUP_DIR="./backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Database credentials
DB_USER=${POSTGRES_USER:-stockuser}
DB_NAME=${POSTGRES_DB:-stockprediction}

echo -e "${GREEN}📦 Starting backup process...${NC}"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup PostgreSQL
echo -e "${YELLOW}Backing up PostgreSQL database...${NC}"
POSTGRES_BACKUP="$BACKUP_DIR/postgres_${DATE}.sql"
docker-compose exec -T postgres pg_dump -U "$DB_USER" "$DB_NAME" > "$POSTGRES_BACKUP"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PostgreSQL backup created${NC}"
    
    # Compress backup
    gzip "$POSTGRES_BACKUP"
    echo -e "${GREEN}✓ Backup compressed: ${POSTGRES_BACKUP}.gz${NC}"
else
    echo -e "${RED}✗ PostgreSQL backup failed${NC}"
    exit 1
fi

# Backup Redis (if needed)
echo -e "${YELLOW}Backing up Redis...${NC}"
REDIS_BACKUP="$BACKUP_DIR/redis_${DATE}.rdb"
docker-compose exec -T redis redis-cli --rdb /data/dump.rdb SAVE > /dev/null 2>&1
docker cp stockpred-redis:/data/dump.rdb "$REDIS_BACKUP" 2>/dev/null

if [ $? -eq 0 ]; then
    gzip "$REDIS_BACKUP"
    echo -e "${GREEN}✓ Redis backup created: ${REDIS_BACKUP}.gz${NC}"
else
    echo -e "${YELLOW}⚠ Redis backup skipped (optional)${NC}"
fi

# Backup models directory
echo -e "${YELLOW}Backing up ML models...${NC}"
MODELS_BACKUP="$BACKUP_DIR/models_${DATE}.tar.gz"
if [ -d "models" ] && [ "$(ls -A models)" ]; then
    tar -czf "$MODELS_BACKUP" models/
    echo -e "${GREEN}✓ Models backup created: ${MODELS_BACKUP}${NC}"
else
    echo -e "${YELLOW}⚠ No models to backup${NC}"
fi

# Backup configuration files
echo -e "${YELLOW}Backing up configuration...${NC}"
CONFIG_BACKUP="$BACKUP_DIR/config_${DATE}.tar.gz"
tar -czf "$CONFIG_BACKUP" \
    --exclude='*.pem' \
    --exclude='*.key' \
    .env docker-compose.yml nginx/nginx.conf 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Configuration backup created: ${CONFIG_BACKUP}${NC}"
fi

# Calculate backup size
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
echo -e "${GREEN}📊 Total backup size: ${BACKUP_SIZE}${NC}"

# Clean up old backups
echo -e "${YELLOW}Cleaning up old backups (older than ${RETENTION_DAYS} days)...${NC}"
find "$BACKUP_DIR" -name "*.gz" -mtime +${RETENTION_DAYS} -delete
find "$BACKUP_DIR" -name "*.sql" -mtime +${RETENTION_DAYS} -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +${RETENTION_DAYS} -delete

REMAINING=$(ls -1 "$BACKUP_DIR" | wc -l)
echo -e "${GREEN}✓ Cleanup complete. ${REMAINING} backup files remaining${NC}"

# Optional: Upload to cloud storage (AWS S3)
if [ ! -z "$AWS_BACKUP_BUCKET" ]; then
    echo -e "${YELLOW}Uploading to AWS S3...${NC}"
    
    if command -v aws &> /dev/null; then
        aws s3 sync "$BACKUP_DIR" "s3://${AWS_BACKUP_BUCKET}/backups/" \
            --exclude "*" \
            --include "*${DATE}*" \
            --region "${AWS_REGION:-us-east-1}" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Backup uploaded to S3${NC}"
        else
            echo -e "${RED}✗ S3 upload failed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ AWS CLI not installed. Skipping S3 upload${NC}"
    fi
fi

# Create backup manifest
MANIFEST="$BACKUP_DIR/manifest_${DATE}.txt"
cat > "$MANIFEST" << EOF
Backup Date: $(date)
PostgreSQL: postgres_${DATE}.sql.gz
Redis: redis_${DATE}.rdb.gz
Models: models_${DATE}.tar.gz
Config: config_${DATE}.tar.gz
Total Size: ${BACKUP_SIZE}
EOF

echo -e "${GREEN}✅ Backup completed successfully!${NC}"
echo -e "${GREEN}Backup location: ${BACKUP_DIR}${NC}"
echo ""
echo "To restore from this backup, run:"
echo "  ./scripts/restore.sh ${DATE}"

exit 0