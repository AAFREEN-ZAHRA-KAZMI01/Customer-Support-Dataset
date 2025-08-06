from typing import List, Dict, Optional
import json
import os
from datetime import datetime

class ConversationMemory:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.memory_file = f"logs/chat_logs_{session_id}.json"
        self.conversation_history: List[Dict[str, str]] = []
        
        # Load existing conversation if available
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                self.conversation_history = json.load(f)
    
    def add_message(self, user_message: str, assistant_response: str):
        """Add a new message pair to the conversation history."""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "assistant": assistant_response
        })
        
        # Save to file
        self._save_to_file()
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation history, optionally limited to last N messages."""
        if limit is not None:
            return self.conversation_history[-limit:]
        return self.conversation_history
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        self._save_to_file()
    
    def _save_to_file(self):
        """Save conversation history to file."""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)