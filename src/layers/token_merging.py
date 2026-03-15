import torch
import torch.nn.functional as F

def local_merge(hidden_states: torch.Tensor, token_weights: torch.Tensor, source_tracking_matrix: torch.Tensor, pos_ids: torch.Tensor | None = None, pad_mask: torch.Tensor | None = None, merge_rate: float = 0.5) -> tuple:
    """
    Merges adjacent contiguous blocks of DNA tokens rather than bipartite matching.
    Takes a sequence of length N and returns one roughly of length N * (1 - merge_rate)
    """
    batch_size, seq_len, d_model = hidden_states.shape
    device = hidden_states.device
    
    if merge_rate <= 0.0 or seq_len <= 1:
        return hidden_states, source_tracking_matrix, token_weights, pos_ids, pad_mask

    target_len = max(1, int(round(seq_len * (1.0 - merge_rate))))
    chunk_size = seq_len / target_len
    
    # We create a new unmerged sequence of target_len
    new_hidden_states = []
    new_token_weights = []
    new_pos_ids = []
    new_pad_mask = []
    
    # The new tracking matrix will map the original N bases to the new target_len chunks
    step_tracking_matrix = torch.zeros(batch_size, seq_len, target_len, device=device)

    for b in range(batch_size):
        b_hidden = []
        b_weights = []
        b_pos = []
        b_pad = []
        
        for i in range(target_len):
            start_idx = int(i * chunk_size)
            end_idx = int((i + 1) * chunk_size) if i < target_len - 1 else seq_len
            
            # Extract the contiguous chunk
            chunk_hidden = hidden_states[b, start_idx:end_idx]
            chunk_weights = token_weights[b, start_idx:end_idx]
            
            # Weighted average based on how many raw tokens are already in this chunk
            total_weight = chunk_weights.sum()
            merged_hidden = (chunk_hidden * chunk_weights).sum(dim=0) / total_weight.clamp(min=1e-5)
            
            b_hidden.append(merged_hidden)
            b_weights.append(total_weight)
            
            # For tracking matrix: all bases from start_idx to end_idx map 100% to this new chunk i
            step_tracking_matrix[b, start_idx:end_idx, i] = 1.0
            
            if pos_ids is not None:
                # Assign the pos ID of the first token in the chunk
                b_pos.append(pos_ids[b, start_idx])
            if pad_mask is not None:
                # If ALL tokens in the chunk are padding, the merged chunk is padding
                b_pad.append(pad_mask[b, start_idx:end_idx].all())
                
        new_hidden_states.append(torch.stack(b_hidden))
        new_token_weights.append(torch.stack(b_weights))
        if pos_ids is not None: new_pos_ids.append(torch.stack(b_pos))
        if pad_mask is not None: new_pad_mask.append(torch.stack(b_pad))

    hidden_states = torch.stack(new_hidden_states)
    token_weights = torch.stack(new_token_weights)
    pos_ids = torch.stack(new_pos_ids) if pos_ids is not None else None
    pad_mask = torch.stack(new_pad_mask) if pad_mask is not None else None
    
    # Multiply the global tracking matrix by this step's tracking matrix
    source_tracking_matrix = torch.bmm(source_tracking_matrix, step_tracking_matrix)
    
    return hidden_states, source_tracking_matrix, token_weights, pos_ids, pad_mask

def global_merge(hidden_states: torch.Tensor, token_weights: torch.Tensor, metric: torch.Tensor, target_len: int, pos_ids: torch.Tensor | None = None, pad_mask: torch.Tensor | None = None) -> tuple:
    """
    Skeleton logic for global latent token merging. 
    (For now, we simply re-use contiguous chunking for stability, but this can be upgraded to Bipartite if desired for the latent space).
    """
    batch_size, seq_len, _ = hidden_states.shape
    if seq_len <= target_len:
        step_tracking_matrix = torch.eye(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1, -1)
        return hidden_states, step_tracking_matrix, token_weights, pos_ids, pad_mask
        
    merge_rate = 1.0 - (target_len / seq_len)
    
    # For a stable Latent Space, we continue using biologically continuous chunking
    # We pass a dummy identity tracking matrix since the actual update happens outside in mergedna.py
    dummy_tracker = torch.eye(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1, -1)
    
    return local_merge(hidden_states, token_weights, dummy_tracker, pos_ids, pad_mask, merge_rate=merge_rate)

def unmerge(hidden_states: torch.Tensor, source_tracking_matrix: torch.Tensor) -> torch.Tensor:
    """
    Unmerges a compressed tensor (Batch, K, Dim) back to its original size (Batch, N, Dim)
    using the mathematical tracking matrix (Batch, N, K).
    Returns (Batch, N, Dim).
    """
    # tracking matrix is shape (B, N, K). hidden_states is (B, K, D)
    # We want output (B, N, D).
    # Math: (B, N, K) @ (B, K, D) -> (B, N, D)
    
    # We must normalize the tracking matrix so that if a token is born from multiple chunks, it averages them
    row_sums = source_tracking_matrix.sum(dim=-1, keepdim=True).clamp(min=1e-5)
    normalized_tracking = source_tracking_matrix / row_sums
    
    return torch.bmm(normalized_tracking, hidden_states)
