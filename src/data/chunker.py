import re

class SemanticChunker:
    """Chunks text preserving semantic boundaries (paragraphs) with optional overlapping."""
    
    def __init__(self, max_words=256, chunk_overlap=50):
        self.max_words = max_words
        self.chunk_overlap = chunk_overlap
        
    def _split_paragraphs(self, text):
        """Splits text into paragraphs by double newlines."""
        return re.split(r'\n\s*\n', text.strip())
        
    def chunk_text(self, text):
        """
        Chunks text into segments of approximately max_words,
        preferring paragraph boundaries and applying overlap.
        """
        paragraphs = self._split_paragraphs(text)
        chunks = []
        current_chunk = [] # Stores words for the current chunk
        current_words = 0 # Word count for current_chunk
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            words = para.split() # Words in the current paragraph
            para_words = len(words)
            
            # If adding this paragraph exceeds max_words, and we have a current chunk
            if current_words + para_words > self.max_words and current_chunk:
                # Flush the current chunk
                chunks.append(" ".join(current_chunk))
                
                # Apply sliding overlap
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    current_chunk = current_chunk[-self.chunk_overlap:]
                    current_words = len(current_chunk)
                else:
                    current_chunk = []
                    current_words = 0
            
            # If the paragraph itself is too large, it needs to be split
            if para_words > self.max_words:
                # If there's a partial chunk before this large paragraph, flush it
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_words = 0
                
                # Split the large paragraph into chunks with overlap
                # Stride is max_words - chunk_overlap to create overlap
                for i in range(0, para_words, self.max_words - self.chunk_overlap):
                    end_idx = i + self.max_words
                    chunk_slice = words[i:end_idx]
                    chunks.append(" ".join(chunk_slice))
                    
                # Setup next chunk correctly if it didn't end perfectly on a boundary
                # This ensures the overlap is correctly handled for the *next* paragraph
                if para_words % (self.max_words - self.chunk_overlap) != 0:
                    # Take the last 'max_words - chunk_overlap' words as the start of the next potential chunk
                    # This is a bit tricky, as the last chunk might already be smaller than max_words
                    # Let's re-evaluate: the last chunk added was `words[i:end_idx]`.
                    # The next `current_chunk` should be the overlap from the *last* chunk created from this paragraph.
                    # If the last chunk was `words[i:end_idx]`, and `end_idx` was `para_words`,
                    # then the overlap should be from `words[para_words - self.chunk_overlap : para_words]`.
                    # However, the loop already handles adding the full chunks.
                    # The `current_chunk` should be the overlap from the *last* chunk generated from this paragraph.
                    
                    # The last chunk generated from this paragraph is `words[last_i : last_end_idx]`.
                    # We need the last `self.chunk_overlap` words from that.
                    # A simpler approach: if the paragraph was split, the `current_chunk` for the *next* paragraph
                    # should start with the overlap from the *last* generated chunk of this paragraph.
                    
                    # Calculate the start index for the overlap from the last split chunk
                    last_chunk_start_idx = max(0, para_words - self.max_words)
                    current_chunk = words[max(0, para_words - self.chunk_overlap):para_words]
                    current_words = len(current_chunk)
                else:
                    current_chunk = []
                    current_words = 0
            else:
                # Paragraph fits, add it to the current chunk
                current_chunk.extend(words)
                current_words += para_words
                
        # Add any remaining words in current_chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
