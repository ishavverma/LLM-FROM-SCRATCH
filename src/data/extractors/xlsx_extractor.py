import pandas as pd
import logging

logger = logging.getLogger(__name__)

class XLSXExtractor:
    """Extracts text from Excel spreadsheets (.xlsx)."""
    
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_pages(self):
        logger.info(f"Opening XLSX for extraction: {self.file_path}")
        try:
            df = pd.read_excel(self.file_path)
            # Use Tab Separation to maintain loose layout structure
            text = df.to_csv(index=False, sep='\t')
            # Simulated block format
            blocks = [(0, 0, 0, 0, text, 0, 0)]
            yield 0, blocks
        except Exception as e:
            logger.error(f"Error reading XLSX {self.file_path}: {e}")
            raise
