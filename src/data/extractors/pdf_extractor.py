import fitz  # PyMuPDF
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFExtractor:
    """Extracts text and layout information from a PDF document."""
    
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_pages(self):
        """
        Reads the PDF and yields the text blocks for each page.
        Returns a generator of lists. Each list contains tuples:
        (x0, y0, x1, y1, text, block_no, block_type)
        where block_type == 0 is text.
        """
        logger.info(f"Opening PDF for extraction: {self.file_path}")
        try:
            doc = fitz.open(self.file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # get_text("blocks") returns list of tuples representing text blocks
                blocks = page.get_text("blocks")
                yield page_num, blocks
        except Exception as e:
            logger.error(f"Error reading PDF {self.file_path}: {e}")
            raise
