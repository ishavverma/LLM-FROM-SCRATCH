import os
import logging
from src.data.extractors.pdf_extractor import PDFExtractor
from src.data.extractors.txt_extractor import TXTExtractor
from src.data.extractors.json_extractor import JSONExtractor
from src.data.extractors.csv_extractor import CSVExtractor
from src.data.extractors.xlsx_extractor import XLSXExtractor

logger = logging.getLogger(__name__)

class IngestionRouter:
    """Automatically dispatches files to their correct extractor based on extension."""
    
    @staticmethod
    def get_extractor(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return PDFExtractor(file_path)
        elif ext == '.txt':
            return TXTExtractor(file_path)
        elif ext in ['.json', '.jsonl']:
            return JSONExtractor(file_path)
        elif ext == '.csv':
            return CSVExtractor(file_path)
        elif ext in ['.xls', '.xlsx']:
            return XLSXExtractor(file_path)
        else:
            logger.warning(f"Unsupported extension {ext} for {file_path}. Defaulting to TXTExtractor.")
            return TXTExtractor(file_path)
