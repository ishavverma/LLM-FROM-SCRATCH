import math
import torch
import torch.nn as nn
from torch.nn import functional as F

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the frequency tensor for complex exponentials (RoPE)"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply Rotary Positional Embedding to query and key tensors"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast freqs_cis to match the shape of xq_ and xk_
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # (1, T, 1, C/2)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) with RoPE and KV Caching."""

    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        
        # Num query groups per KV head
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_cis, kv_cache=None, use_cache=False):
        B, T, C = x.size()

        # Calculate Query, Key, Value
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # KV Cache logic
        if kv_cache is not None:
            # kv_cache is a tuple (k_cache, v_cache)
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
            
        current_kv_cache = (k, v) if use_cache else None

        # Repeat K and V to match the number of Q heads (GQA logic)
        # (B, T_k, n_kv_heads, head_dim) -> (B, T_k, n_kv_heads, n_rep, head_dim) -> (B, T_k, n_heads, head_dim)
        k = k[:, :, :, None, :].expand(B, k.size(1), self.n_kv_heads, self.n_rep, self.head_dim).reshape(B, k.size(1), self.n_heads, self.head_dim)
        v = v[:, :, :, None, :].expand(B, v.size(1), self.n_kv_heads, self.n_rep, self.head_dim).reshape(B, v.size(1), self.n_heads, self.head_dim)

        # Transpose to (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Causal Self-Attention
        # is_causal varies depending on whether we are processing the prompt or generating token-by-token
        is_causal = q.size(2) > 1 # if decoding 1 token at a time, causal masking is irrelevant/handled by cache
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=self.attn_dropout.p if self.training else 0)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.wo(y))
        
        return y, current_kv_cache

class FeedForward(nn.Module):
    """A simple linear network followed by a non-linearity."""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # Use SwiGLU-ish or standard GELU
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model, dropout)

    def forward(self, x, freqs_cis, kv_cache=None, use_cache=False):
        attn_out, new_kv_cache = self.attn(self.ln_1(x), freqs_cis, kv_cache, use_cache)
        x = x + attn_out
        x = x + self.ffwd(self.ln_2(x))
        return x, new_kv_cache

class GPTLanguageModel(nn.Module):
    """A decoder-only transformer language model with RoPE, GQA, and KV Cache."""

    def __init__(self, vocab_size, d_model=256, n_heads=8, n_kv_heads=None, n_layer=6, block_size=1024, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.n_layer = n_layer
        
        if n_kv_heads is None:
            n_kv_heads = n_heads // 2 # default to half heads for GQA
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, d_model),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(d_model, n_heads, n_kv_heads, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(d_model),
        ))
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # with weight tying
        self.transformer.wte.weight = self.lm_head.weight
        
        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(d_model // n_heads, self.block_size * 2)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Special initialization for GPT
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale weights of residual projections for deep models
        for name, p in module.named_parameters():
            if name.endswith('wo.weight') or name.endswith('net.2.weight'):
                # Net.2 is the second linear in FeedForward
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))

    def forward(self, idx, targets=None, kv_caches=None, use_cache=False, start_pos=0):
        device = idx.device
        b, t = idx.size()
        
        # Get RoPE frequencies for the current position window
        freqs_cis = self.freqs_cis[start_pos : start_pos + t].to(device)
        
        # forward the GPT model itself (No WPE because of RoPE)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, d_model)
        
        x = self.transformer.drop(tok_emb)
        
        new_kv_caches = []
        for i, block in enumerate(self.transformer.h):
            past_kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, freqs_cis, past_kv, use_cache)
            new_kv_caches.append(new_kv)
            
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100) # Instruct masking
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
            
        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0, stop_token_id=None):
        """
        Token generation using KV Caching for O(1) step complexity.
        """
        device = idx.device
        
        # Pre-fill phase (process the entire prompt context)
        kv_caches = None
        start_pos = 0
        
        # To avoid prompt length exceeding max block size in dummy systems:
        if idx.size(1) > self.block_size:
            idx = idx[:, -self.block_size:]
            
        # Get initial logits and priming cache
        logits, _, kv_caches = self(idx, use_cache=True, start_pos=0)
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for i in range(idx.shape[1]):
                token = idx[0, i].item()
                logits[0, -1, token] = torch.where(logits[0, -1, token] < 0, logits[0, -1, token] * repetition_penalty, logits[0, -1, token] / repetition_penalty)

        # Pluck the logits at the final step
        logits = logits[:, -1, :] / temperature
        
        # Optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
            
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Track full output and new single index
        output = torch.cat((idx, idx_next), dim=1)
        current_idx = idx_next
        start_pos = idx.size(1) # Next generation step starts at index T
        
        if stop_token_id is not None and current_idx.item() == stop_token_id:
            return output
            
        for _ in range(1, max_new_tokens):
            # forward the model to get the logits for the NEXT single index using cache
            logits, _, kv_caches = self(current_idx, kv_caches=kv_caches, use_cache=True, start_pos=start_pos)
            
            if repetition_penalty != 1.0:
                for i in range(output.shape[1]):
                    token = output[0, i].item()
                    logits[0, -1, token] = torch.where(logits[0, -1, token] < 0, logits[0, -1, token] * repetition_penalty, logits[0, -1, token] / repetition_penalty)
                     
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            output = torch.cat((output, idx_next), dim=1)
            current_idx = idx_next
            start_pos += 1 # shift window forward by 1 token
            
            if stop_token_id is not None and current_idx.item() == stop_token_id:
                break
            
        return output

    @torch.no_grad()
    def stream_generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0, stop_token_id=None):
        """
        Token generation using KV Caching, yielding tokens one by one for streaming.
        """
        device = idx.device
        
        kv_caches = None
        start_pos = 0
        output_tokens = []
        for i in range(idx.shape[1]):
             output_tokens.append(idx[0, i].item())
        
        if idx.size(1) > self.block_size:
            idx = idx[:, -self.block_size:]
            
        logits, _, kv_caches = self(idx, use_cache=True, start_pos=0)
        
        if repetition_penalty != 1.0:
            for token in set(output_tokens):
                logits[0, -1, token] = torch.where(logits[0, -1, token] < 0, logits[0, -1, token] * repetition_penalty, logits[0, -1, token] / repetition_penalty)
                
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        current_idx = idx_next
        output_tokens.append(current_idx.item())
        start_pos = idx.size(1)
        
        if stop_token_id is not None and current_idx.item() == stop_token_id:
            return
            
        yield current_idx.item()
            
        for _ in range(1, max_new_tokens):
            logits, _, kv_caches = self(current_idx, kv_caches=kv_caches, use_cache=True, start_pos=start_pos)
            
            if repetition_penalty != 1.0:
                for token in set(output_tokens):
                    logits[0, -1, token] = torch.where(logits[0, -1, token] < 0, logits[0, -1, token] * repetition_penalty, logits[0, -1, token] / repetition_penalty)
                    
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            current_idx = idx_next
            output_tokens.append(current_idx.item())
            start_pos += 1
            
            if stop_token_id is not None and current_idx.item() == stop_token_id:
                break
                
            yield current_idx.item()
