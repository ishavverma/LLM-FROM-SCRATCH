import re
import logging

logger = logging.getLogger(__name__)

class Guardrails:
    """Implements safety mechanisms for training and inference."""
    
    def __init__(self, apply_sentiment=True):
        self.apply_sentiment = apply_sentiment
        # Basic regex for PII like SSNs, emails, phone numbers
        self.pii_patterns = [
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), # SSN
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), # Email
            re.compile(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b') # Phone
        ]
        
        self.unsafe_keywords = [
            "hack", "exploit", "malware", "virus", "bomb"
        ]
        
    def filter_training_chunk(self, text):
        """Redacts PII from text before it hits the tokenizer."""
        clean_text = text
        for pattern in self.pii_patterns:
            clean_text = pattern.sub("[REDACTED]", clean_text)
        return clean_text
        
    def _is_hostile_or_toxic(self, text):
        """Custom heuristic to detect hostile or toxic sentiment without external libraries."""
        if not self.apply_sentiment: return False
        
        # Simple word-list based polarity check
        negative_words = ["hate", "kill", "stupid", "idiot", "awful", "terrible", "worst", "garbage"]
        hostile_words = ["die", "murder", "stab", "shoot", "attack"]
        
        text_lower = text.lower()
        neg_count = sum(1 for word in negative_words if word in text_lower)
        hostile_count = sum(1 for word in hostile_words if word in text_lower)
        
        # Heuristic: More than 3 negative words OR any extremely hostile word
        if hostile_count > 0 or neg_count > 3:
            return True
        return False
        
    def scan_prompt(self, prompt):
        """Checks if a user prompt violates safety rules."""
        if self._is_hostile_or_toxic(prompt):
            logger.warning("Hostile or toxic sentiment detected in prompt (custom heuristic).")
            return False, "Prompt violates safety policy (detected hostile sentiment)"
            
        lower_prompt = prompt.lower()
        for kw in self.unsafe_keywords:
            if kw in lower_prompt:
                logger.warning(f"Unsafe keyword detected in prompt: {kw}")
                return False, f"Prompt violates safety policy (detected: {kw})"
        return True, ""
        
    def scan_response(self, response):
        """Validates the model's generated response before returning it."""
        if self._is_hostile_or_toxic(response):
            logger.warning("Hostile or toxic sentiment detected in response (custom heuristic).")
            return False, "The generated response was blocked by sentiment guardrails."
            
        lower_response = response.lower()
        for kw in self.unsafe_keywords:
            if kw in lower_response:
                logger.warning("Unsafe keyword detected in model response.")
                return False, "The generated response was blocked by safety guardrails."
        
        # Also redact PII in response just in case
        safe_response = self.filter_training_chunk(response)
        return True, safe_response
