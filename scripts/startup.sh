# #!/bin/bash

# # ==========================================
# # Stock Predictor Platform - Startup Script
# # ==========================================

# set -e  # Exit on error

# # Colors for output
# RED='\033[0;31m'
# GREEN='\033[0;32m'
# YELLOW='\033[1;33m'
# BLUE='\033[0;34m'
# NC='\033[0m' # No Color

# # Functions
# print_success() {
#     echo -e "${GREEN}✓ $1${NC}"
# }

# print_error() {
#     echo -e "${RED}✗ $1${NC}"
# }

# print_warning() {
#     echo -e "${YELLOW}⚠ $1${NC}"
# }

# print_info() {
#     echo -e "${BLUE}ℹ $1${NC}"
# }

# print_header() {
#     echo -e "\n${BLUE}================================================${NC}"
#     echo -e "${BLUE}$1${NC}"
#     echo -e "${BLUE}================================================${NC}\n"
# }

# # Main script
# print_header "🚀 Stock Predictor Platform - Starting..."

# # Check if running as root
# if [ "$EUID" -eq 0 ]; then 
#     print_warning "Running as root is not recommended for security reasons"
# fi

# # Check prerequisites
# print_info "Checking prerequisites..."

# # Check Docker
# if ! command -v docker &> /dev/null; then
#     print_error "Docker is not installed. Please install Docker first."
#     exit 1
# fi
# print_success "Docker found"

# # Check Docker Compose
# if ! command -v docker-compose &> /dev/null; then
#     print_error "Docker Compose is not installed. Please install Docker Compose first."
#     exit 1
# fi
# print_success "Docker Compose found"

# # Check if .env exists
# if [ ! -f .env ]; then
#     print_error ".env file not found!"
#     print_info "Creating .env from .env.example..."
#     if [ -f .env.example ]; then
#         cp .env.example .env
#         print_warning "Please edit .env file with your configuration before proceeding."
#         print_info "Run: nano .env"
#         exit 1
#     else
#         print_error ".env.example not found. Cannot create .env file."
#         exit 1
#     fi
# fi
# print_success ".env file found"

# # Load environment variables safely
# print_info "Loading environment variables..."
# while IFS='=' read -r key value; do
#     # Skip comments and empty lines
#     [[ "$key" =~ ^#.*$ ]] && continue
#     [[ -z "$key" ]] && continue
#     # Remove inline comments
#     value=$(echo "$value" | sed 's/#.*//')
#     # Export safely
#     export "$key=$value"
# done < .env
# print_success "Environment variables loaded"

# # Create necessary directories
# print_info "Creating necessary directories..."
# mkdir -p logs models data backups nginx/ssl monitoring/grafana/dashboards
# print_success "Directories created"

# # Generate SSL certificates if not exist
# print_info "Checking SSL certificates..."
# if [ ! -f nginx/ssl/cert.pem ] || [ ! -f nginx/ssl/key.pem ]; then
#     print_warning "SSL certificates not found. Generating self-signed certificates..."
    
#     openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
#         -keyout nginx/ssl/key.pem \
#         -out nginx/ssl/cert.pem \
#         -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" \
#         2>/dev/null
    
#     if [ $? -eq 0 ]; then
#         print_success "SSL certificates generated"
#         print_warning "For production, use Let's Encrypt certificates!"
#     else
#         print_error "Failed to generate SSL certificates"
#         exit 1
#     fi
# else
#     print_success "SSL certificates found"
# fi

# # Stop existing containers
# print_info "Stopping existing containers (if any)..."
# docker-compose down 2>/dev/null || true

# # Pull latest images
# print_info "Pulling latest Docker images..."
# docker-compose pull

# # Build custom images
# print_info "Building Docker images..."
# docker-compose build

# # Start services
# print_header "🐳 Starting Docker Services"
# docker-compose up -d

# # Wait for services to be healthy
# print_info "Waiting for services to be ready..."
# sleep 10

# # Check service health
# print_info "Checking service health..."
# services=("postgres" "redis" "backend")
# all_healthy=true

# for service in "${services[@]}"; do
#     if docker-compose ps | grep -q "$service.*Up"; then
#         print_success "$service is running"
#     else
#         print_error "$service failed to start"
#         all_healthy=false
#     fi
# done

# if [ "$all_healthy" = false ]; then
#     print_error "Some services failed to start. Check logs:"
#     print_info "docker-compose logs -f"
#     exit 1
# fi

# # Run database migrations
# print_info "Running database migrations..."
# docker-compose exec -T backend alembic upgrade head 2>/dev/null || {
#     print_warning "Migrations failed or not configured yet"
# }

# # Optional: Seed database
# if [ "$1" = "--seed" ]; then
#     print_info "Seeding database with test data..."
#     docker-compose exec -T backend python database.py || {
#         print_warning "Database seeding failed"
#     }
# fi

# # Display service URLs
# print_header "✅ Stock Predictor Platform is Running!"

# echo ""
# echo "🌐 Service URLs:"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo -e "${GREEN}Backend API:${NC}       http://localhost:8000"
# echo -e "${GREEN}API Documentation:${NC} http://localhost:8000/docs"
# echo -e "${GREEN}API Redoc:${NC}         http://localhost:8000/redoc"
# echo -e "${GREEN}Grafana:${NC}           http://localhost:3001"
# echo -e "${GREEN}Prometheus:${NC}        http://localhost:9090"
# echo -e "${GREEN}PgAdmin:${NC}           http://localhost:5050"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo ""

# echo "📊 Default Credentials:"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo -e "${YELLOW}Grafana:${NC}   admin / admin123"
# echo -e "${YELLOW}PgAdmin:${NC}   admin@stockpred.com / (check .env)"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo ""

# echo "📝 Useful Commands:"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "  View logs:        docker-compose logs -f"
# echo "  View backend:     docker-compose logs -f backend"
# echo "  Stop services:    docker-compose down"
# echo "  Restart:          docker-compose restart"
# echo "  Shell access:     docker-compose exec backend bash"
# echo "  Run tests:        docker-compose exec backend pytest"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo ""

# print_success "Startup complete! 🎉"
# print_info "For troubleshooting, check: ./docs/TROUBLESHOOTING.md"

# # Optional: Open browser
# if command -v xdg-open &> /dev/null; then
#     read -p "Open API docs in browser? (y/n) " -n 1 -r
#     echo
#     if [[ $REPLY =~ ^[Yy]$ ]]; then
#         xdg-open http://localhost:8000/docs
#     fi
# elif command -v open &> /dev/null; then
#     read -p "Open API docs in browser? (y/n) " -n 1 -r
#     echo
#     if [[ $REPLY =~ ^[Yy]$ ]]; then
#         open http://localhost:8000/docs
#     fi
# fi

# exit 0



#!/bin/bash
# Startup script for production deployment

set -e

echo "🚀 Starting Stock Prediction Platform..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please copy .env.example to .env and configure it."
    exit 1
fi

# Load environment variables safely
set -a
source .env
set +a
echo "✓ Environment variables loaded"

# Create necessary directories
mkdir -p logs models data nginx/ssl
echo "✓ Directories created"

# Generate SSL certificates (self-signed for development)
if [ ! -f nginx/ssl/cert.pem ]; then
    echo "📜 Generating self-signed SSL certificate..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/key.pem \
        -out nginx/ssl/cert.pem \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    echo "✓ SSL certificates generated"
else
    echo "✓ SSL certificates already exist"
fi

# Stop existing containers if running
echo "🛑 Stopping existing containers (if any)..."
docker-compose down || true

# Pull latest Docker images
echo "⬇️ Pulling latest Docker images..."
docker-compose pull

# Start Docker containers
echo "🐳 Starting Docker containers..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to initialize..."
sleep 15

# Run database migrations
echo "🔧 Running database migrations..."
docker-compose exec -T backend alembic upgrade head

# Seed database (optional)
read -p "Do you want to seed the database with test data? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose exec -T backend python database.py
fi

echo "✅ Stock Prediction Platform is running!"
echo ""
echo "🌐 Services:"
echo "   - Backend API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Grafana: http://localhost:3001"
echo "   - Prometheus: http://localhost:9090"
echo "   - PgAdmin: http://localhost:5050"
echo ""
echo "📊 To view logs: docker-compose logs -f"
echo "🛑 To stop: docker-compose down"
