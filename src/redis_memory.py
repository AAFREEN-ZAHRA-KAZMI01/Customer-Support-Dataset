import redis
import json
from datetime import datetime
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

# Environment variables load karein
load_dotenv()

class RedisMemory:
    def __init__(self, session_id: str = "default"):
        """
        Redis Memory Store - Conversation History Ko Permanent Save Karega
        
        Arguments:
            session_id: Har user/vartalaap ka unique ID (default: "default")
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session ID must be a non-empty string")
        
        try:
            # Redis connection establish karein
            self.redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                password=os.getenv("REDIS_PASSWORD", None),
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis.ping()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {str(e)}")

        self.session_id = session_id
        self.key = f"ecom_chat:{session_id}"

    def add_message(self, user_message: str, assistant_response: str) -> bool:
        """
        Naya Message Redis Mein Save Karein
        
        Arguments:
            user_message: User ka input/question
            assistant_response: Bot ka response
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not user_message or not assistant_response:
            raise ValueError("Messages cannot be empty")
            
        try:
            # Message dictionary banayein
            message = {
                "timestamp": datetime.now().isoformat(),
                "user": user_message,
                "assistant": assistant_response
            }
            
            # Redis mein JSON format mein save karein
            self.redis.rpush(self.key, json.dumps(message))
            
            # 7 din (604800 seconds) ke baad automatic delete ho jaye
            self.redis.expire(self.key, 604800)
            return True
        except Exception as e:
            print(f"Error saving message: {str(e)}")
            return False

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Purani Conversations Retrieve Karein
        
        Arguments:
            limit: Kitne recent messages chahiye (None = sab)
            
        Returns:
            List of message dictionaries
        """
        try:
            # Redis se messages retrieve karein
            messages = self.redis.lrange(self.key, 0, -1)
            
            # JSON strings ko python dictionaries mein convert karein
            history = [json.loads(msg) for msg in messages]
            
            # Agar limit diya hai to utne hi recent messages return karein
            return history[-limit:] if limit is not None else history
        except Exception as e:
            print(f"Error retrieving history: {str(e)}")
            return []

    def clear_history(self) -> bool:
        """Is session ki saari conversation history delete karein"""
        try:
            return self.redis.delete(self.key) > 0
        except Exception as e:
            print(f"Error clearing history: {str(e)}")
            return False

    def get_last_message(self) -> Optional[Dict[str, str]]:
        """
        Sabse Latest Message Retrieve Karein
        
        Returns:
            Dictionary with last message details ya None
        """
        try:
            last_msg = self.redis.lindex(self.key, -1)
            return json.loads(last_msg) if last_msg else None
        except Exception as e:
            print(f"Error getting last message: {str(e)}")
            return None