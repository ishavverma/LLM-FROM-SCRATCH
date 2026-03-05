import json
import os
import faiss
import numpy as np
import torch
from src.model.tokenizer import BPETokenizer
from rank_bm25 import BM25Okapi

# A simple mock embedder since we don't have a pretrained embedding model ready
# in a real scenario we extract embeddings from our model

class VectorStore:
    """Manages document embeddings using FAISS and sparse keyword retrieval using BM25."""
    
    def __init__(self, embedding_dim=384, alpha=0.5):
        self.embedding_dim = embedding_dim
        # Using L2 distance
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks = []
        self.bm25 = None
        self.tokenized_corpus = []
        self.alpha = alpha # Weight for dense vs sparse fusion
        
    def add_texts(self, texts, embeddings):
        """Adds texts and their embeddings to the FAISS index and BM25 corpus."""
        embeddings_np = np.array(embeddings).astype('float32')
        self.index.add(embeddings_np)
        self.chunks.extend(texts)
        
        # Add to BM25
        for text in texts:
            tokens = text.lower().split()
            self.tokenized_corpus.append(tokens)
            
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
    def search(self, query, query_embedding, k=3):
        """
        Retrieves top k chunks using Hybrid Search (Reciprocal Rank Fusion of FAISS and BM25).
        """
        if not self.chunks:
            return []
            
        k = min(k, len(self.chunks))
        
        # 1. Dense Search (FAISS)
        query_np = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_np, k * 2) # Get more for fusion
        
        dense_ranks = {}
        for rank, idx in enumerate(indices[0]):
            if idx != -1:
                 dense_ranks[idx] = rank + 1
                 
        # 2. Sparse Search (BM25)
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_sparse_indices = np.argsort(bm25_scores)[::-1][:k * 2]
        
        sparse_ranks = {}
        for rank, idx in enumerate(top_sparse_indices):
             sparse_ranks[idx] = rank + 1
             
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k_rrf = 60 # standard RRF constant
        
        all_indices = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
        for idx in all_indices:
             dense_score = 1.0 / (k_rrf + dense_ranks.get(idx, 1000))
             sparse_score = 1.0 / (k_rrf + sparse_ranks.get(idx, 1000))
             # Weighted sum
             rrf_scores[idx] = (self.alpha * dense_score) + ((1 - self.alpha) * sparse_score)
             
        # 4. Sort and return top k
        sorted_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)[:k]
        
        return [self.chunks[i] for i in sorted_indices]

    def save(self, directory):
        """Saves the FAISS index and chunk metadata."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "vector_index.faiss"))
        with open(os.path.join(directory, "chunks_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load(self, directory):
        """Loads the FAISS index, chunk metadata, and rebuilds BM25."""
        index_path = os.path.join(directory, "vector_index.faiss")
        chunks_path = os.path.join(directory, "chunks_metadata.json")
        
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(index_path)
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
                
            # Rebuild BM25
            self.tokenized_corpus = [text.lower().split() for text in self.chunks]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            return True
        return False

class RAGPipeline:
    def __init__(self, model, tokenizer, vector_store, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.vector_store = vector_store
        self.device = device
        
    def _embed_text(self, text):
        """
        Mocks the embedding step by averaging token embeddings from our own model.
        """
        tokens = self.tokenizer.encode(text)
        if not tokens:
            return np.zeros((1, self.vector_store.embedding_dim), dtype=np.float32)
            
        idx = torch.tensor([tokens], dtype=torch.long).to(self.device)
        idx_cond = idx if idx.size(1) <= self.model.block_size else idx[:, -self.model.block_size:]
        
        with torch.no_grad():
            tok_emb = self.model.transformer.wte(idx_cond) # (1, T, D)
            # mean pooling
            emb = tok_emb.mean(dim=1).cpu().numpy()
        return np.array(emb, dtype=np.float32)

    def build_index(self, data_path="data/approved_chunks.json"):
        if not os.path.exists(data_path):
            print(f"Skipping FAISS index build. {data_path} not found.")
            return
            
        with open(data_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            
        embeddings = []
        texts = []
        for chunk in chunks:
            chunk_text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            emb = self._embed_text(chunk_text)
            embeddings.append(emb[0])
            texts.append(chunk_text)
            
        self.vector_store.add_texts(texts, np.array(embeddings))
        
        os.makedirs("checkpoints", exist_ok=True)
        self.vector_store.save("checkpoints")
        print("Vector index saved.")
        
    def generate_with_rag(self, query_text, max_new_tokens=100, temperature=0.7, top_p=0.9, repetition_penalty=1.1):
        # 1. Embed query
        query_emb = self._embed_text(query_text)
        
        # 2. Retrieve relevant chunks (Hybrid Search)
        context_text = ""
        if self.vector_store.index.ntotal > 0:
            retrieved_chunks = self.vector_store.search(query_text, query_emb, k=3)
            # Filter duplicates and clean up headers/footers
            unique_chunks = []
            for c in retrieved_chunks:
                # Remove Page labels and recurring headers
                cleaned = re.sub(r"Page \d+ \| AI Research Series", "", c).strip()
                if cleaned not in unique_chunks:
                    unique_chunks.append(cleaned)
            
            context_text = "\n".join([f"- {c}" for c in unique_chunks])
            
        # 3. Construct Lean prompt for grounding
        if context_text:
            prompt = (
                f"Use the context below to answer accurately. If unsure, say you don't know.\n"
                f"Context:\n{context_text}\n"
                f"Q: {query_text}\n"
                f"A:"
            )
        else:
            prompt = f"Q: {query_text}\nA:"
            
        # 4. Generate response with robust parameters
        tokens = self.tokenizer.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        stop_id = self.tokenizer.special_tokens.get("<|endoftext|>")
        
        generated_idx = self.model.generate(
            idx, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_token_id=stop_id
        )
        
        response_tokens = generated_idx[0].tolist()
        # only return the newest tokens
        response_tokens = response_tokens[len(tokens):]
        
        return self.tokenizer.decode(response_tokens)
