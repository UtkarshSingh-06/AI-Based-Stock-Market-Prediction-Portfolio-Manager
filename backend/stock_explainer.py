# backend/stock_explainer.py
"""
AI-powered stock movement explanation service
Generates human-readable explanations for stock price movements
"""
import os
import logging
import yfinance as yf
import requests
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class StockExplainer:
    """Generate explanations for stock price movements"""
    
    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.use_ai = bool(self.openai_api_key)
        
    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get current stock data and recent history"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get recent price data
            hist = ticker.history(period="5d")
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change_pct = ((current_price - prev_price) / prev_price) * 100
            
            # Get volume data
            current_volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'previous_price': float(prev_price),
                'change_pct': float(change_pct),
                'change_amount': float(current_price - prev_price),
                'volume': float(current_volume),
                'avg_volume': float(avg_volume),
                'volume_ratio': float(volume_ratio),
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'pe_ratio': info.get('trailingPE'),
                'beta': info.get('beta'),
                '52w_high': info.get('fiftyTwoWeekHigh'),
                '52w_low': info.get('fiftyTwoWeekLow'),
            }
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return None
    
    def get_recent_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Get recent news articles for the stock"""
        news_items = []
        
        try:
            # Try using yfinance news
            ticker = yf.Ticker(symbol)
            news = ticker.news[:10]  # Get top 10 news items
            
            for item in news:
                news_items.append({
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', ''),
                    'link': item.get('link', ''),
                    'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)) if item.get('providerPublishTime') else None
                })
        except Exception as e:
            logger.warning(f"Error fetching news for {symbol}: {e}")
        
        # If NewsAPI is configured, use it as well
        if self.news_api_key:
            try:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': symbol,
                    'apiKey': self.news_api_key,
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'pageSize': 10
                }
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    for article in data.get('articles', []):
                        news_items.append({
                            'title': article.get('title', ''),
                            'publisher': article.get('source', {}).get('name', ''),
                            'link': article.get('url', ''),
                            'published': datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')) if article.get('publishedAt') else None
                        })
            except Exception as e:
                logger.warning(f"Error fetching NewsAPI data: {e}")
        
        return news_items
    
    def generate_explanation_ai(self, symbol: str, stock_data: Dict, news_items: List[Dict]) -> str:
        """Generate explanation using OpenAI API"""
        if not self.openai_api_key:
            return self.generate_explanation_rule_based(symbol, stock_data, news_items)
        
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            # Build context
            context = f"Stock: {symbol}\n"
            context += f"Current Price: ${stock_data['current_price']:.2f}\n"
            context += f"Change: {stock_data['change_pct']:+.2f}%\n"
            context += f"Volume: {stock_data['volume_ratio']:.2f}x average\n"
            
            if stock_data.get('pe_ratio'):
                context += f"P/E Ratio: {stock_data['pe_ratio']:.2f}\n"
            
            if news_items:
                context += "\nRecent News:\n"
                for item in news_items[:5]:
                    context += f"- {item['title']}\n"
            
            prompt = f"""You are a financial analyst. Explain why {symbol} moved {stock_data['change_pct']:+.2f}% in a concise, informative way (2-3 sentences max). 
Focus on key factors like earnings, news, market sentiment, technical indicators, or sector trends.
Be specific and avoid generic statements.

Context:
{context}

Explanation:"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a concise financial analyst explaining stock movements."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            explanation = response.choices[0].message.content.strip()
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating AI explanation: {e}")
            return self.generate_explanation_rule_based(symbol, stock_data, news_items)
    
    def generate_explanation_rule_based(self, symbol: str, stock_data: Dict, news_items: List[Dict]) -> str:
        """Generate explanation using rule-based logic"""
        change_pct = stock_data['change_pct']
        volume_ratio = stock_data['volume_ratio']
        
        explanations = []
        
        # Price movement analysis
        if abs(change_pct) < 0.5:
            explanations.append("minimal price movement")
        elif abs(change_pct) < 2:
            if change_pct > 0:
                explanations.append("modest gains")
            else:
                explanations.append("modest decline")
        elif abs(change_pct) < 5:
            if change_pct > 0:
                explanations.append("significant upward movement")
            else:
                explanations.append("significant downward pressure")
        else:
            if change_pct > 0:
                explanations.append("strong rally")
            else:
                explanations.append("sharp decline")
        
        # Volume analysis
        if volume_ratio > 2:
            explanations.append("unusually high trading volume")
        elif volume_ratio < 0.5:
            explanations.append("below-average trading volume")
        
        # Valuation analysis
        if stock_data.get('pe_ratio'):
            pe = stock_data['pe_ratio']
            if pe > 30:
                explanations.append("lofty valuation")
            elif pe < 10:
                explanations.append("attractive valuation")
        
        # News analysis
        if news_items:
            recent_news = [n for n in news_items if n.get('published') and 
                          (datetime.now() - n['published'].replace(tzinfo=None)).days <= 1]
            if recent_news:
                explanations.append("recent news developments")
        
        # Sector/industry context
        if stock_data.get('sector'):
            sector = stock_data['sector']
            if sector in ['Technology', 'Consumer Cyclical']:
                if change_pct < 0:
                    explanations.append("broader sector sentiment concerns")
                else:
                    explanations.append("sector momentum")
        
        # Market context (beta analysis)
        if stock_data.get('beta'):
            beta = stock_data['beta']
            if beta > 1.5 and abs(change_pct) > 2:
                explanations.append("high volatility characteristic")
        
        # Combine explanations
        if not explanations:
            explanations.append("normal market activity")
        
        # Format final explanation
        if len(explanations) == 1:
            explanation = f"due to {explanations[0]}"
        elif len(explanations) == 2:
            explanation = f"due to {explanations[0]} and {explanations[1]}"
        else:
            explanation = f"due to {', '.join(explanations[:-1])}, and {explanations[-1]}"
        
        return explanation.capitalize()
    
    def explain_movement(self, symbol: str) -> Optional[Dict]:
        """
        Generate comprehensive explanation for stock movement
        
        Returns:
            Dict with 'explanation', 'change_pct', 'current_price', etc.
        """
        try:
            # Get stock data
            stock_data = self.get_stock_data(symbol)
            if not stock_data:
                return None
            
            # Get news
            news_items = self.get_recent_news(symbol)
            
            # Generate explanation
            if self.use_ai:
                explanation = self.generate_explanation_ai(symbol, stock_data, news_items)
            else:
                explanation = self.generate_explanation_rule_based(symbol, stock_data, news_items)
            
            return {
                'symbol': symbol,
                'explanation': explanation,
                'change_pct': stock_data['change_pct'],
                'current_price': stock_data['current_price'],
                'previous_price': stock_data['previous_price'],
                'volume_ratio': stock_data['volume_ratio'],
                'news_count': len(news_items)
            }
            
        except Exception as e:
            logger.error(f"Error explaining movement for {symbol}: {e}")
            return None

# Global instance
stock_explainer = StockExplainer()
