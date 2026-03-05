import json
import re
import logging

logger = logging.getLogger(__name__)

class DocumentIntelligence:
    """Applies rules from config.json to filter out PDF artifacts."""
    
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            
        self.ignored_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.config.get("ignored_patterns", [])
        ]
        
        self.target_language = self.config.get("target_language", "en")
        
    def is_header_or_footer(self, y0, y1, page_height=792): # 792 is typical 11 inches in points
        """Heuristic check for headers and footers based on coordinates."""
        # Top 10% for header, bottom 10% for footer
        if self.config.get("remove_headers", False) and y1 < page_height * 0.1:
            return True
        if self.config.get("remove_footers", False) and y0 > page_height * 0.9:
            return True
        return False
        
    def is_watermark_or_ignored(self, text):
        """Checks against bad strings or regex patterns."""
        if self.config.get("remove_watermarks", False):
            # very simplistic watermark check
            if "watermark" in text.lower():
                return True
                
        for pattern in self.ignored_patterns:
            if pattern.search(text):
                return True
        return False
        
    def is_page_number(self, text, y0, y1, page_height=792):
        """Checks if block is likely a page number (digit-only in header/footer area)."""
        if not self.config.get("remove_page_numbers", False):
            return False
        
        # Check if it's just numbers and whitespace
        if text.strip().isdigit() and self.is_header_or_footer(y0, y1, page_height):
            return True
        return False
        
    def is_low_quality(self, text):
        """Heuristic check to filter out non-textual or junk blocks (OCR noise, tables, foreign code)."""
        if len(text) < 10: return False # Too short to judge
        
        # Check digit-to-char ratio (excessive numbers usually mean data tables)
        digits = sum(c.isdigit() for c in text)
        if digits / len(text) > 0.4: return True
        
        # Check symbol-to-char ratio (junk characters)
        symbols = sum(not c.isalnum() and not c.isspace() for c in text)
        if symbols / len(text) > 0.3: return True
        
        return False

    def filter_blocks(self, blocks, page_height=792):
        """
        Takes a list of blocks: (x0, y0, x1, y1, text, block_no, block_type)
        Returns a cleaned text sequence.
        """
        cleaned_text = []
        for b in blocks:
            # We only care about text (block_type 0)
            if len(b) >= 7 and b[6] != 0:
                continue
                
            x0, y0, x1, y1, text, block_no, block_type = b[:7]
            text = text.strip()
            
            if not text:
                continue
                
            if self.is_page_number(text, y0, y1, page_height):
                continue
            if self.is_header_or_footer(y0, y1, page_height):
                continue
            if self.is_watermark_or_ignored(text):
                continue
                
            if self.is_low_quality(text):
                logger.warning(f"Filtered low-quality chunk: {text[:30]}...")
                continue
                     
            cleaned_text.append(text)
            
        return "\n".join(cleaned_text)
