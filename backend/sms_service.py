# backend/sms_service.py
"""
SMS service for sending stock movement notifications via Twilio
"""
import os
import logging
from typing import Optional
from twilio.rest import Client
from twilio.base.exceptions import TwilioException

logger = logging.getLogger(__name__)

class SMSService:
    """Service for sending SMS notifications via Twilio"""
    
    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_PHONE_NUMBER")
        
        if not all([self.account_sid, self.auth_token, self.from_number]):
            logger.warning("Twilio credentials not configured. SMS will be disabled.")
            self.client = None
        else:
            try:
                self.client = Client(self.account_sid, self.auth_token)
                logger.info("SMS service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SMS service: {e}")
                self.client = None
    
    def send_sms(self, to_phone: str, message: str) -> bool:
        """
        Send SMS message to phone number
        
        Args:
            to_phone: Phone number in E.164 format (e.g., +1234567890)
            message: Message content (max 1600 characters)
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.client:
            logger.warning("SMS service not configured. Message not sent.")
            return False
        
        if not to_phone:
            logger.error("Phone number is required")
            return False
        
        # Ensure phone number is in E.164 format
        if not to_phone.startswith('+'):
            logger.warning(f"Phone number {to_phone} may not be in E.164 format")
        
        try:
            # Truncate message if too long (SMS limit is 1600 chars)
            if len(message) > 1600:
                message = message[:1597] + "..."
            
            message_obj = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_phone
            )
            
            logger.info(f"SMS sent successfully to {to_phone}. SID: {message_obj.sid}")
            return True
            
        except TwilioException as e:
            logger.error(f"Twilio error sending SMS to {to_phone}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending SMS to {to_phone}: {e}")
            return False
    
    def send_stock_alert(self, to_phone: str, symbol: str, change_pct: float, 
                        explanation: str, current_price: float) -> bool:
        """
        Send formatted stock movement alert
        
        Args:
            to_phone: Phone number
            symbol: Stock symbol
            change_pct: Percentage change (positive for up, negative for down)
            explanation: Explanation of the movement
            current_price: Current stock price
            
        Returns:
            bool: True if sent successfully
        """
        direction = "⬆️" if change_pct > 0 else "⬇️"
        change_str = f"{abs(change_pct):.2f}%"
        
        message = f"Stock Tracker Agent 💰\n\n{direction} {symbol} {change_pct:+.2f}% ({current_price:.2f})\n\n{explanation}"
        
        return self.send_sms(to_phone, message)
    
    def verify_phone_number(self, phone_number: str) -> bool:
        """
        Verify phone number format (basic validation)
        
        Args:
            phone_number: Phone number to verify
            
        Returns:
            bool: True if format looks valid
        """
        # Basic E.164 format check
        if not phone_number:
            return False
        
        # Should start with + and contain only digits after that
        if phone_number.startswith('+'):
            digits = phone_number[1:]
            return digits.isdigit() and len(digits) >= 10
        else:
            # Allow numbers without + for flexibility
            return phone_number.replace('-', '').replace(' ', '').replace('(', '').replace(')', '').isdigit() and len(phone_number.replace('-', '').replace(' ', '').replace('(', '').replace(')', '')) >= 10

# Global instance
sms_service = SMSService()
