# SMS Stock Tracker Agent Setup Guide

## Overview

The SMS Stock Tracker Agent sends SMS notifications when your tracked stocks move significantly, with AI-powered explanations for the price movements.

Example message:
```
Stock Tracker Agent 💰

⬇️ NVDA down 1.89% ($485.23)

Due to geopolitical risks in China, a modest data center revenue miss, lofty valuation, and the broader sentiment that the AI frenzy may be overheating.
```

## Prerequisites

1. **Twilio Account** (for SMS)
   - Sign up at https://www.twilio.com
   - Get your Account SID, Auth Token, and Phone Number

2. **OpenAI API Key** (optional, for AI explanations)
   - Sign up at https://platform.openai.com
   - Get your API key
   - If not provided, the system will use rule-based explanations

3. **News API Key** (optional, for better explanations)
   - Sign up at https://newsapi.org
   - Get your API key

## Environment Variables

Add these to your `.env` file:

```bash
# Twilio SMS Configuration
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+1234567890  # Your Twilio phone number in E.164 format

# OpenAI (Optional - for AI explanations)
OPENAI_API_KEY=sk-your-openai-api-key-here

# News API (Optional - for better news context)
NEWS_API_KEY=your-news-api-key-here
```

## Database Setup

The SMS feature requires new database tables. Run migrations:

```bash
# If using Alembic
alembic upgrade head

# Or manually create tables
python -c "from database import init_db; init_db()"
```

## API Endpoints

### 1. Subscribe to SMS Notifications

```bash
POST /api/v1/sms/subscribe
Authorization: Bearer <token>

{
  "phone_number": "+1234567890",
  "min_change_threshold": 1.0,  # Minimum % change to trigger (default: 1.0%)
  "notification_frequency": "daily",  # daily, hourly, or realtime
  "tracked_symbols": ["NVDA", "AAPL", "TSLA"]  # Optional: specific stocks to track
}
```

### 2. Get Current Subscription

```bash
GET /api/v1/sms/subscription
Authorization: Bearer <token>
```

### 3. Update Subscription

```bash
PUT /api/v1/sms/subscription
Authorization: Bearer <token>

{
  "min_change_threshold": 2.0,
  "tracked_symbols": ["NVDA", "AAPL"],
  "is_active": true
}
```

### 4. Send Test SMS

```bash
POST /api/v1/sms/test
Authorization: Bearer <token>
```

### 5. Get Notification History

```bash
GET /api/v1/sms/notifications?limit=20
Authorization: Bearer <token>
```

### 6. Get Stock Movement Explanation

```bash
POST /api/v1/sms/explain/NVDA
Authorization: Bearer <token>
```

## How It Works

1. **Stock Monitoring**: Celery Beat runs hourly to check all tracked stocks
2. **Movement Detection**: Compares current price with previous price
3. **Threshold Check**: Only sends notification if change exceeds your threshold
4. **Explanation Generation**: Uses AI (if available) or rule-based logic to explain movement
5. **SMS Delivery**: Sends formatted message via Twilio
6. **Rate Limiting**: Respects daily notification limits (default: 10/day)

## Configuration Options

- **min_change_threshold**: Minimum percentage change to trigger (0.1% - 50%)
- **notification_frequency**: How often to check (daily, hourly, realtime)
- **tracked_symbols**: Specific stocks to monitor (empty = all watchlist stocks)
- **daily_notification_limit**: Max notifications per day (default: 10)

## Troubleshooting

### SMS Not Sending

1. Check Twilio credentials in `.env`
2. Verify phone number format (E.164: +1234567890)
3. Check Twilio account balance
4. Review logs: `docker logs stockpred-celery-worker`

### No Explanations

1. If using AI: Check OpenAI API key and credits
2. System falls back to rule-based explanations if AI unavailable
3. Check stock data availability (yfinance)

### Not Receiving Notifications

1. Verify subscription is active: `GET /api/v1/sms/subscription`
2. Check if movement exceeds threshold
3. Verify daily limit not reached
4. Check notification history: `GET /api/v1/sms/notifications`

## Cost Considerations

- **Twilio**: ~$0.0075 per SMS (US numbers)
- **OpenAI**: ~$0.002 per explanation (GPT-3.5-turbo)
- **News API**: Free tier: 100 requests/day

## Example Usage

```python
import requests

# Subscribe
response = requests.post(
    "http://localhost:8000/api/v1/sms/subscribe",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={
        "phone_number": "+1234567890",
        "min_change_threshold": 1.5,
        "tracked_symbols": ["NVDA", "AAPL"]
    }
)

# Get explanation for a stock
response = requests.post(
    "http://localhost:8000/api/v1/sms/explain/NVDA",
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
print(response.json()["explanation"])
```

## Monitoring

Check Celery task status:
```bash
docker logs stockpred-celery-worker
docker logs stockpred-celery-beat
```

View notification history in database or via API endpoint.
