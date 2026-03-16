import torch
import torch.nn as nn
from src.config import MergeDNAConfig
from src.layers.attention import TransformerBlock, LocalTransformerBlock
from src.layers.token_merging import local_merge, global_merge, unmerge

def _merges_per_layer(current_len: int, target_len: int, layers_left: int) -> float:
    if current_len <= target_len or layers_left <= 0: return 0.0
    return 1.0 - (target_len / current_len) ** (1.0 / layers_left)

class MergeDNA(nn.Module):
    def __init__(self, config: MergeDNAConfig):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        
        self.local_encoder = nn.ModuleList([
            LocalTransformerBlock(config.d_model, config.num_heads, config.local_window_size, config.mlp_ratio, config.dropout)
            for _ in range(config.local_encoder_layers)
        ])
        
        self.latent_encoder = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.latent_encoder_layers)
        ])
        
        self.latent_decoder = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.latent_decoder_layers)
        ])
        
        self.local_decoder = nn.ModuleList([
            LocalTransformerBlock(config.d_model, config.num_heads, config.local_window_size, config.mlp_ratio, config.dropout)
            for _ in range(config.local_decoder_layers)
        ])
        
        self.head = nn.Linear(config.d_model, config.vocab_size)
        
    def forward(self, input_ids: torch.Tensor, disable_latent_merge: bool = False, local_target_ratio: float = None, latent_target_ratio: float = None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        local_ratio = local_target_ratio if local_target_ratio is not None else self.config.local_target_ratio
        latent_ratio = latent_target_ratio if latent_target_ratio is not None else self.config.latent_target_ratio
        
        x = self.token_emb(input_ids)
        
        # Initial states
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pad_mask = (input_ids == self.config.pad_token_id)
        token_weights = torch.ones(batch_size, seq_len, device=device)
        s_local = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 1. Local Encoder (N -> L)
        target_l = max(1, int(seq_len * local_ratio))
        
        for i, layer_block in enumerate(self.local_encoder):
            x, metric = layer_block(x, token_weights=token_weights, pos_ids=pos_ids, pad_mask=pad_mask, return_metric=True)
            
            layers_left = len(self.local_encoder) - i
            merge_rate = _merges_per_layer(x.size(1), target_l, layers_left)
            
            x, s_local, token_weights, pos_ids, pad_mask = local_merge(
                x, token_weights, metric, s_local, window_size=self.config.local_window_size, 
                pos_ids=pos_ids, pad_mask=pad_mask, merge_rate=merge_rate
            )
            
        z_L = x
        sizes_L = token_weights
        
        # 2. Latent Encoder (L -> K)
        len_l = x.size(1)
        target_k = max(1, int(len_l * latent_ratio))
        s_latent = torch.eye(len_l, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        for i, layer_block in enumerate(self.latent_encoder):
            x, metric = layer_block(x, token_weights=token_weights, pos_ids=pos_ids, pad_mask=pad_mask, return_metric=True)
            
            if not disable_latent_merge:
                layers_left = len(self.latent_encoder) - i
                target_len_step = max(target_k, int(x.size(1) * (1.0 - _merges_per_layer(x.size(1), target_k, layers_left))))
                
                x, s_latent, token_weights, pos_ids, pad_mask = global_merge(
                    x, token_weights, metric, target_len_step, pos_ids=pos_ids, pad_mask=pad_mask
                )
                
        z_K = x
        
        # 3. Latent Decoder (K -> L)
        x = unmerge(z_K, s_latent)
        token_weights = torch.ones(batch_size, len_l, device=device)
        pos_ids_l = torch.arange(len_l, device=device).unsqueeze(0).expand(batch_size, -1)
        pad_mask_l = None # Assume no padding in latent space reconstruction
        
        for layer_block in self.latent_decoder:
            x = layer_block(x, token_weights=token_weights, pos_ids=pos_ids_l, pad_mask=pad_mask_l)
            
        # 4. Local Decoder (L -> N)
        x = unmerge(x, s_local)
        token_weights = torch.ones(batch_size, seq_len, device=device)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pad_mask = (input_ids == self.config.pad_token_id)
        
        for layer_block in self.local_decoder:
            x = layer_block(x, token_weights=token_weights, pos_ids=pos_ids, pad_mask=pad_mask)
            
        # 5. Output
        logits = self.head(x)
        
        return {
            "logits": logits,
            "z_L": z_L,
            "sizes_L": sizes_L,
            "s_local": s_local,
            "s_latent": s_latent
        }
