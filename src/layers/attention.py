import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, pos_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, num_heads, head_dim = xq.shape
    
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float().to(xq.device) / head_dim))
    freqs = torch.einsum("bi,j->bij", pos_ids.float().to(xq.device), inv_freq)
    freqs = freqs.unsqueeze(2)
    
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    return (xq * cos) + (rotate_half(xq) * sin), (xk * cos) + (rotate_half(xk) * sin)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, x: torch.Tensor, attn_bias: torch.Tensor | None = None,
        token_weights: torch.Tensor | None = None, pos_ids: torch.Tensor | None = None, pad_mask: torch.Tensor | None = None
    ):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        if pos_ids is not None:
            q, k = apply_rotary_emb(q, k, pos_ids)
            
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attn_bias is not None:
            scores = scores + attn_bias
            
        if pad_mask is not None:
            pad_bias = torch.zeros((batch_size, 1, seq_len, seq_len), device=x.device)
            pad_bias.masked_fill_(pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            scores = scores + pad_bias
            
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Metric for token merging: Paper uses Keys specifically
        # We average across heads as per standard practices in reference repos
        metric = k.mean(dim=1) # [B, N, head_dim]
        return self.o_proj(out), metric

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(int(d_model * mlp_ratio), d_model)
        )
        
    def forward(
        self, x: torch.Tensor, attn_bias: torch.Tensor | None = None, token_weights: torch.Tensor | None = None,
        pos_ids: torch.Tensor | None = None, pad_mask: torch.Tensor | None = None, return_metric: bool = False
    ):
        attn_out, metric = self.attn(self.norm1(x), attn_bias, token_weights, pos_ids, pad_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return (x, metric) if return_metric else x

def make_local_attn_bias(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    mask = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)) >= window_size
    bias = torch.zeros(seq_len, seq_len, device=device)
    return bias.masked_fill_(mask, float("-inf")).unsqueeze(0).unsqueeze(0)

class LocalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, window_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.window_size = window_size
        self.block = TransformerBlock(d_model, num_heads, mlp_ratio, dropout)
        
    def forward(
        self, x: torch.Tensor, token_weights: torch.Tensor | None = None, pos_ids: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None, return_metric: bool = False
    ):
        return self.block(x, make_local_attn_bias(x.size(1), self.window_size, x.device), token_weights, pos_ids, pad_mask, return_metric)
