import logging

logger = logging.getLogger(__name__)

class TXTExtractor:
    """Extracts text from plain .txt files."""
    
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_pages(self):
        """
        Reads the TXT and yields text mimicking the PDF extractor's format.
        Since TXT files don't have pages, we yield the entire text as page 0,
        or we could split by double newlines to simulate pages/blocks.
        
        Returns a generator of tuples:
        (page_num, blocks)
        where blocks is a list of tuples: (x0, y0, x1, y1, text, block_no, block_type)
        """
        logger.info(f"Opening TXT for extraction: {self.file_path}")
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Simulated block format: x0, y0, x1, y1, text, block_no, block_type=0
            # For pure text, coordinates don't matter, we just pass the text
            blocks = [(0, 0, 0, 0, text, 0, 0)]
            yield 0, blocks
            
        except Exception as e:
            logger.error(f"Error reading TXT {self.file_path}: {e}")
            raise
