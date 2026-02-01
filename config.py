"""
Configuration management for the Stock Prediction application.
Loads settings from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from typing import List

# Base directory
BASE_DIR = Path(__file__).parent

# Backend Configuration
BACKEND_PORT = int(os.environ.get('PORT', 8001))
BACKEND_HOST = os.environ.get('HOST', '0.0.0.0')
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:3000,http://localhost:3001').split(',')

# MongoDB Configuration
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')

# Frontend Configuration
FRONTEND_BACKEND_URL = os.environ.get('REACT_APP_BACKEND_URL', 'http://localhost:8001')

# Logging Configuration
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
LOG_FILE = BASE_DIR / 'stock_prediction.log'

# Directories
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
CACHE_DIR = BASE_DIR / 'cache'

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Model Configuration
DEFAULT_SEQ_LEN = int(os.environ.get('DEFAULT_SEQ_LEN', 60))
DEFAULT_HIDDEN_SIZE = int(os.environ.get('DEFAULT_HIDDEN_SIZE', 50))
DEFAULT_NUM_LAYERS = int(os.environ.get('DEFAULT_NUM_LAYERS', 2))
DEFAULT_EPOCHS = int(os.environ.get('DEFAULT_EPOCHS', 10))

# Training Configuration
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 5

# Data Download Configuration
MAX_RETRIES = 5
RETRY_DELAY = 5
CACHE_EXPIRY_DAYS = 1

# Validation
MIN_DATA_POINTS = 100
MIN_SEQ_LENGTH = 10
MAX_SEQ_LENGTH = 200
MAX_EPOCHS = 100
