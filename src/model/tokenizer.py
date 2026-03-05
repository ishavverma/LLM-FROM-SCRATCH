import json
import logging
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

class BPETokenizer:
    """A minimal Byte Pair Encoding (BPE) subword tokenizer."""
    
    def __init__(self, vocab_size=1000):
        self.target_vocab_size = vocab_size
        # For simplicity, byte vocab
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {} # (idx1, idx2) -> new_idx
        
        self.special_tokens = {
            "<|endoftext|>": self.target_vocab_size,
            "<|prompt|>": self.target_vocab_size + 1,
            "<|response|>": self.target_vocab_size + 2,
            "<|system|>": self.target_vocab_size + 3,
            "<|user|>": self.target_vocab_size + 4,
            "<|assistant|>": self.target_vocab_size + 5
        }
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.total_vocab_size = self.target_vocab_size + len(self.special_tokens)
        
    def _get_stats(self, ids):
        """Counts frequency of token pairs."""
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts
        
    def _merge(self, ids, pair, idx):
        """Replaces all occurrences of `pair` in `ids` with `idx`."""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text):
        """Trains the tokenizer and builds the merge tree."""
        # For training BPE, we typically ignore special tokens or treat them as raw boundaries.
        # We will quickly string replace them before training, or just naively train on them. 
        # Since this is a simple implementation, let's strip them from the training text string.
        for st in self.special_tokens.keys():
            text = text.replace(st, "")
            
        logger.info(f"Training tokenizer on {len(text)} bytes. Target vocab size: {self.target_vocab_size}")
        tokens = list(text.encode("utf-8"))
        
        num_merges = self.target_vocab_size - 256
        for i in range(num_merges):
            stats = self._get_stats(tokens)
            if not stats:
                break
            # Find the most frequent pair
            best = max(stats, key=stats.get)
            
            new_idx = 256 + i
            # Merge
            tokens = self._merge(tokens, best, new_idx)
            
            # Save to models
            self.merges[best] = new_idx
            self.vocab[new_idx] = self.vocab[best[0]] + self.vocab[best[1]]
            
        logger.info(f"Training complete. Vocab size: {len(self.vocab)}")
        
    def encode(self, text):
        """Converts text into a list of token ids, preserving special tokens."""
        pattern = "(" + "|".join(map(re.escape, self.special_tokens.keys())) + ")"
        parts = re.split(pattern, text)
        
        all_tokens = []
        for part in parts:
            if part in self.special_tokens:
                all_tokens.append(self.special_tokens[part])
            elif part:
                tokens = list(part.encode("utf-8"))
                while len(tokens) >= 2:
                    stats = self._get_stats(tokens)
                    # Find pair in merges that appears in tokens and has the lowest merge index
                    pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
                    
                    if pair not in self.merges:
                        break # nothing else can be merged
                    
                    idx = self.merges[pair]
                    tokens = self._merge(tokens, pair, idx)
                all_tokens.extend(tokens)
                
        return all_tokens
        
    def decode(self, ids):
        """Converts a list of token ids back into text."""
        decoded_bytes = []
        decoded_str = ""
        for idx in ids:
            if idx in self.inverse_special_tokens:
                if decoded_bytes:
                    decoded_str += b"".join(decoded_bytes).decode('utf-8', errors='replace')
                    decoded_bytes = []
                decoded_str += self.inverse_special_tokens[idx]
            else:
                decoded_bytes.append(self.vocab.get(idx, b""))
                
        if decoded_bytes:
            decoded_str += b"".join(decoded_bytes).decode('utf-8', errors='replace')
            
        return decoded_str

    def save(self, filepath):
        """Saves merges and vocab configuration."""
        # Convert keys from tuple to string for JSON serialization
        str_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        with open(filepath, 'w') as f:
            json.dump({'merges': str_merges, 'vocab_size': len(self.vocab)}, f, indent=4)
            
    def load(self, filepath):
        """Loads merges and reconstructs vocab."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.merges = {}
        str_merges = data['merges']
        for k_str, v in str_merges.items():
            k1, k2 = map(int, k_str.split(','))
            self.merges[(k1, k2)] = v
            
        # Reconstruct vocab
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        # Merges dictate the vocab reconstruction
        for (k1, k2), vid in self.merges.items():
            self.vocab[vid] = self.vocab[k1] + self.vocab[k2]
