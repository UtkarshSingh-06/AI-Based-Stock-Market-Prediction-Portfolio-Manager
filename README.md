# 📈 Advanced Stock Prediction Platform with GRU Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

A production-grade machine learning platform for stock price prediction using advanced GRU neural networks, featuring real-time predictions, uncertainty estimation, backtesting, and comprehensive monitoring.

## 🌟 Key Features

### Machine Learning & Predictions
- ✅ **Multiple Model Architectures**: GRU, LSTM, Transformer with attention mechanisms
- ✅ **Uncertainty Estimation**: Monte Carlo Dropout for confidence intervals
- ✅ **Advanced Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic
- ✅ **Sentiment Analysis**: News and social media sentiment integration
- ✅ **Ensemble Predictions**: Combine multiple models for robust forecasts
- ✅ **Real-time Predictions**: Live market data integration

### Backend & Infrastructure
- ✅ **FastAPI REST API**: High-performance async API with automatic documentation
- ✅ **PostgreSQL Database**: Reliable data persistence with connection pooling
- ✅ **Redis Caching**: Sub-millisecond response times for frequent queries
- ✅ **Celery Workers**: Distributed task processing for model training
- ✅ **Background Tasks**: Scheduled model retraining and data updates

### Security & Authentication
- ✅ **JWT Authentication**: Secure token-based auth with refresh tokens
- ✅ **API Key Management**: Programmatic access control
- ✅ **Role-Based Access**: Free, Premium, Enterprise tiers
- ✅ **Rate Limiting**: Prevent API abuse with configurable limits
- ✅ **Comprehensive Audit Logging**: Track all user actions
- ✅ **Password Hashing**: Bcrypt encryption for credentials
- ✅ **SQL Injection Protection**: Parameterized queries via SQLAlchemy

### Trading & Analytics
- ✅ **Advanced Backtesting**: Multiple strategies with transaction costs
- ✅ **Portfolio Management**: Track positions and performance
- ✅ **Risk Metrics**: Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio
- ✅ **Price Alerts**: Real-time notifications for price targets
- ✅ **Performance Tracking**: Detailed analytics and metrics

### SMS Stock Tracker Agent 💰
- ✅ **SMS Notifications**: Get notified via SMS when stocks move significantly
- ✅ **AI-Powered Explanations**: Understand why stocks moved with intelligent explanations
- ✅ **Customizable Thresholds**: Set minimum percentage change to trigger alerts
- ✅ **Smart Monitoring**: Automatic hourly monitoring of tracked stocks
- ✅ **Notification History**: Track all SMS notifications sent
- ✅ **Rate Limiting**: Daily limits to prevent spam

### DevOps & Monitoring
- ✅ **Docker Containerization**: Easy deployment with Docker Compose
- ✅ **Prometheus Metrics**: System and application monitoring
- ✅ **Grafana Dashboards**: Visual performance tracking
- ✅ **Nginx Reverse Proxy**: Load balancing and SSL termination
- ✅ **Automated Backups**: Database and model checkpointing
- ✅ **Health Checks**: Service availability monitoring

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 8GB+ RAM (16GB recommended)
- 50GB+ disk space

### Installation

```bash
# Clone repository
git clone https://github.com/UtkarshSingh-06/PBL_project.git
cd PBL_project

# Copy environment file
cp .env.example .env

# Edit .env with your configuration
nano .env

# Start all services
chmod +x scripts/startup.sh
./scripts/startup.sh

# Access the platform
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
# Grafana: http://localhost:3001
```

## 📱 SMS Stock Tracker Agent

Get real-time SMS notifications when your stocks move, with AI-powered explanations!

**Example SMS:**
```
Stock Tracker Agent 💰

⬇️ NVDA down 1.89% ($485.23)

Due to geopolitical risks in China, a modest data center revenue miss, lofty valuation, and the broader sentiment that the AI frenzy may be overheating.
```

### Quick Setup

1. **Get Twilio credentials** (for SMS)
2. **Add to `.env`**:
   ```bash
   TWILIO_ACCOUNT_SID=your_account_sid
   TWILIO_AUTH_TOKEN=your_auth_token
   TWILIO_PHONE_NUMBER=+1234567890
   OPENAI_API_KEY=sk-your-key  # Optional, for AI explanations
   ```

3. **Subscribe via API**:
   ```bash
   POST /api/v1/sms/subscribe
   {
     "phone_number": "+1234567890",
     "min_change_threshold": 1.0,
     "tracked_symbols": ["NVDA", "AAPL"]
   }
   ```

See [backend/SMS_SETUP.md](backend/SMS_SETUP.md) for detailed setup instructions.

## 📁 Project Structure

```
stock-predictor/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── database.py             # Database models & ORM
│   ├── enhanced_predictor.py   # ML models & training
│   ├── tasks.py                # Celery background tasks
│   ├── sms_service.py          # SMS service (Twilio)
│   ├── stock_explainer.py      # AI stock movement explanations
│   ├── requirements.txt        # Python dependencies
│   ├── SMS_SETUP.md            # SMS setup guide
│   └──