#!/bin/bash

# ===================================================
# Stock Predictor Platform - Complete Setup Script
# ===================================================

echo "🚀 Setting up Stock Prediction Platform..."
echo ""

# Create project root directory
PROJECT_NAME="stock-predictor"
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

echo "📁 Creating directory structure..."

# Create main directories
mkdir -p backend/alembic/versions
mkdir -p frontend/public
mkdir -p frontend/src/components
mkdir -p nginx/ssl
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources
mkdir -p scripts
mkdir -p models
mkdir -p logs
mkdir -p data
mkdir -p backups

echo "✅ Directories created"
echo ""

# Create placeholder files
touch nginx/ssl/.gitkeep
touch models/.gitkeep
touch logs/.gitkeep
touch data/.gitkeep
touch backups/.gitkeep

echo "📝 Please create the following files with the content I provided:"
echo ""
echo "ROOT DIRECTORY:"
echo "  - .env.example (copy from artifact 'root_env_example')"
echo "  - .gitignore (copy from artifact 'root_gitignore')"
echo "  - docker-compose.yml (copy from artifact 'docker_setup')"
echo "  - README.md (copy from artifact 'main_readme')"
echo ""
echo "BACKEND DIRECTORY (backend/):"
echo "  - main.py (copy from artifact 'production_backend')"
echo "  - database.py (copy from artifact 'database_config')"
echo "  - enhanced_predictor.py (copy from artifact 'enhanced_predictor_full')"
echo "  - tasks.py (copy from artifact 'docker_setup' - tasks section)"
echo "  - requirements.txt (from artifacts)"
echo "  - Dockerfile (from artifacts)"
echo ""
echo "FRONTEND DIRECTORY (frontend/):"
echo "  - package.json (copy from artifact 'frontend_package_json')"
echo "  - public/index.html (copy from artifact 'frontend_index_html')"
echo "  - src/index.js (copy from artifact 'frontend_index_js')"
echo "  - src/index.css (copy from artifact 'frontend_index_css')"
echo "  - src/App.jsx (copy from artifact 'react_frontend')"
echo ""
echo "NGINX DIRECTORY (nginx/):"
echo "  - nginx.conf (copy from artifact 'docker_setup' - nginx section)"
echo ""
echo "MONITORING DIRECTORY (monitoring/):"
echo "  - prometheus.yml (copy from artifact 'docker_setup' - prometheus section)"
echo ""
echo "SCRIPTS DIRECTORY (scripts/):"
echo "  - startup.sh (copy from artifact 'docker_setup' - startup script)"
echo "  - backup.sh (copy from artifact 'docker_setup' - backup script)"
echo "  - init.sql (copy from artifact 'database_config' - init script)"
echo ""

# Make scripts executable
chmod +x scripts/*.sh 2>/dev/null

echo "📋 Next steps:"
echo "1. Copy all file contents from the artifacts to their respective locations"
echo "2. Copy .env.example to .env and configure your settings"
echo "3. Run: docker-compose up -d"
echo ""
echo "🎉 Setup structure complete!"