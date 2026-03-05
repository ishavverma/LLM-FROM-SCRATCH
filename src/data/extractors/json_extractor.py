import logging
import json

logger = logging.getLogger(__name__)

class JSONExtractor:
    """Extracts text from .json or .jsonl files. Useful for Instruct datasets."""
    
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_pages(self):
        """
        Reads a JSON or JSONL file. 
        If it's an array of objects or JSONL, it extracts text from specific fields 
        (e.g., 'text', 'instruction', 'response').
        
        Yields (page_num, blocks) where blocks simulate the extractor format.
        """
        logger.info(f"Opening JSON for extraction: {self.file_path}")
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                # Naive check for JSONL vs JSON based on first char or we can just try JSON
                content = f.read().strip()
                
            entries = []
            if content.startswith('['):
                # Standard JSON array
                entries = json.loads(content)
            else:
                # Assume JSONL
                for line in content.split('\n'):
                    if line.strip():
                        entries.append(json.loads(line))
            
            text_blocks = []
            for i, entry in enumerate(entries):
                # We attempt to extract common keys
                text = ""
                if "instruction" in entry and "output" in entry:
                     # Alpaca style
                     text = f"<|prompt|>\n{entry['instruction']}\n{entry.get('input', '')}\n<|response|>\n{entry['output']}\n<|endoftext|>"
                elif "messages" in entry:
                     # ChatML style
                     for msg in entry["messages"]:
                         role = msg.get("role", "user")
                         content = msg.get("content", "")
                         text += f"<|{role}|>\n{content}\n"
                     text += "<|endoftext|>"
                elif "text" in entry:
                     text = entry["text"]
                else:
                     # Just stringify the dict
                     text = json.dumps(entry)
                     
                text_blocks.append((0, 0, 0, 0, text, i, 0))
                
            # Yield all blocks as "page 0"
            yield 0, text_blocks

        except Exception as e:
            logger.error(f"Error reading JSON {self.file_path}: {e}")
            raise
