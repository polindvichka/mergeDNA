from dataclasses import dataclass

@dataclass
class MergeDNAConfig:
    vocab_size: int = 6  # A, C, G, T, PAD, MASK
    max_seq_len: int = 4096
    
    # Architecture Dimensions
    d_model: int = 1024
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_rotary_embeddings: bool = True
    
    # Token Merging and Windowing
    local_window_size: int = 16
    local_target_ratio: float = 0.5  # Local encoder targets L ~= N/2
    latent_target_ratio: float = 0.5 # Latent encoder targets K ~= L/2
    local_ratio_min: float = 0.4
    local_ratio_max: float = 0.6

    # Layer Counts
    local_encoder_layers: int = 4
    latent_encoder_layers: int = 20
    latent_decoder_layers: int = 4
    local_decoder_layers: int = 2
    
    # Optimizer & Pretraining Defaults
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 10_000
    total_steps: int = 100_000

    # Pretraining Loss Weights
    lambda_latent_mtr: float = 0.25
    mask_ratio: float = 0.15
    mask_token_id: int = 5
    pad_token_id: int = 4
