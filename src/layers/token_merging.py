import torch
import torch.nn.functional as F

def calculate_adjacent_similarities(hidden_states: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    """
    Calculates cosine similarity between each token and its immediate right neighbor (i, i+1).
    Returns a tensor of shape (batch, seq_len - 1)
    """
    # metric is shape (batch, seq_len, dim)
    left_tokens = metric[:, :-1, :]
    right_tokens = metric[:, 1:, :]
    
    # Cosine similarity: (A dot B) / (||A|| * ||B||)
    return F.cosine_similarity(left_tokens, right_tokens, dim=-1)

def local_merge(hidden_states: torch.Tensor, token_weights: torch.Tensor, metric: torch.Tensor, source_tracking_matrix: torch.Tensor, window_size: int = 16, pos_ids: torch.Tensor | None = None, pad_mask: torch.Tensor | None = None, merge_rate: float = 0.5) -> tuple:
    """
    Biologically conservative Token Merging respecting attention windows:
    Chunks the sequence into blocks of `window_size`, calculates adjacent similarity 
    inside each block, and merges the highest scoring pairs until `merge_rate` is met.
    """
    batch_size, seq_len, d_model = hidden_states.shape
    device = hidden_states.device
    
    if merge_rate <= 0.0 or seq_len <= 1:
        return hidden_states, source_tracking_matrix, token_weights, pos_ids, pad_mask

    # We must process batch by batch and window by window.
    merged_hidden_list = []
    merged_weights_list = []
    merged_pos_ids_list = []
    merged_pad_masks_list = []
    step_tracking_matrices = []

    for b in range(batch_size):
        b_hidden_new = []
        b_weights_new = []
        b_pos_new = []
        b_pad_new = []
        
        # We will build the step tracking matrix diagonally block by block
        # Because we don't know the exact new length yet, we build a list of tracking sub-matrices
        b_tracking_blocks = []
        
        for start_idx in range(0, seq_len, window_size):
            end_idx = min(start_idx + window_size, seq_len)
            w_len = end_idx - start_idx
            
            w_hidden = hidden_states[b, start_idx:end_idx]
            w_weights = token_weights[b, start_idx:end_idx]
            w_metric = metric[b:b+1, start_idx:end_idx]
            
            if pos_ids is not None: w_pos = pos_ids[b, start_idx:end_idx]
            if pad_mask is not None: w_pad = pad_mask[b, start_idx:end_idx]
            
            num_to_merge = min(int(w_len * merge_rate), w_len // 2)
            
            if num_to_merge <= 0 or w_len < 2:
                b_hidden_new.extend(w_hidden)
                b_weights_new.extend(w_weights)
                b_tracking_blocks.append(torch.eye(w_len, device=device))
                if pos_ids is not None: b_pos_new.extend(w_pos)
                if pad_mask is not None: b_pad_new.extend(w_pad)
                continue
                
            w_sim = calculate_adjacent_similarities(w_metric, w_metric)[0] # (w_len - 1)
            sorted_indices = torch.argsort(w_sim, descending=True).tolist()
            
            selected_pairs = set()
            blocked_indices = set()
            
            for idx in sorted_indices:
                if len(selected_pairs) >= num_to_merge: break
                if idx in blocked_indices or (idx - 1) in blocked_indices or (idx + 1) in blocked_indices:
                    continue
                selected_pairs.add(idx)
                blocked_indices.add(idx)
                blocked_indices.add(idx + 1)

            # Reconstruct this specific window
            new_w_len = w_len - len(selected_pairs)
            w_step_tracking = torch.zeros((w_len, new_w_len), device=device)
            
            i = 0
            new_idx = 0
            while i < w_len:
                if i in selected_pairs:
                    weight_i, weight_j = w_weights[i], w_weights[i + 1]
                    total_weight = weight_i + weight_j
                    merged_h = (w_hidden[i] * weight_i + w_hidden[i + 1] * weight_j) / total_weight.clamp(min=1e-6)
                    
                    b_hidden_new.append(merged_h)
                    b_weights_new.append(total_weight)
                    
                    w_step_tracking[i, new_idx] = 1.0
                    w_step_tracking[i + 1, new_idx] = 1.0
                    
                    if pos_ids is not None: b_pos_new.append(w_pos[i])
                    if pad_mask is not None: b_pad_new.append(w_pad[i] and w_pad[i + 1])
                    
                    i += 2
                else:
                    b_hidden_new.append(w_hidden[i])
                    b_weights_new.append(w_weights[i])
                    
                    w_step_tracking[i, new_idx] = 1.0
                    
                    if pos_ids is not None: b_pos_new.append(w_pos[i])
                    if pad_mask is not None: b_pad_new.append(w_pad[i])
                    
                    i += 1
                new_idx += 1
            
            b_tracking_blocks.append(w_step_tracking)

        # Combine blocks into the full batch tracking matrix
        b_step_tracking = torch.block_diag(*b_tracking_blocks)

        merged_hidden_list.append(torch.stack(b_hidden_new))
        merged_weights_list.append(torch.stack(b_weights_new))
        step_tracking_matrices.append(b_step_tracking)
        
        if pos_ids is not None: merged_pos_ids_list.append(torch.stack(b_pos_new))
        if pad_mask is not None: merged_pad_masks_list.append(torch.stack(b_pad_new))

    hidden_states = torch.stack(merged_hidden_list)
    token_weights = torch.stack(merged_weights_list)
    step_tracking_matrix = torch.stack(step_tracking_matrices)
    
    pos_ids = torch.stack(merged_pos_ids_list) if pos_ids is not None else None
    pad_mask = torch.stack(merged_pad_masks_list) if pad_mask is not None else None
    
    source_tracking_matrix = torch.bmm(source_tracking_matrix, step_tracking_matrix)
    
    return hidden_states, source_tracking_matrix, token_weights, pos_ids, pad_mask

def global_merge(hidden_states: torch.Tensor, token_weights: torch.Tensor, metric: torch.Tensor, target_len: int, pos_ids: torch.Tensor | None = None, pad_mask: torch.Tensor | None = None) -> tuple:
    """
    Standard ToMe Bipartite Matching for the abstract Global Latent Space.
    Matches the most similar tokens across the ENTIRE sequence, regardless of adjacency.
    """
    batch_size, seq_len, _ = hidden_states.shape
    device = hidden_states.device
    
    num_to_merge = seq_len - target_len
    if num_to_merge <= 0 or seq_len <= 1:
        step_tracking = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        return hidden_states, step_tracking, token_weights, pos_ids, pad_mask
        
    num_to_merge = min(num_to_merge, seq_len // 2)

    # 1. Split sequence roughly into A (sources) and B (destinations) like paper ToMe
    a_tokens, b_tokens = hidden_states[:, ::2, :], hidden_states[:, 1::2, :]
    a_metric, b_metric = metric[:, ::2, :], metric[:, 1::2, :]
    a_weights, b_weights = token_weights[:, ::2], token_weights[:, 1::2]
    
    if pos_ids is not None: a_pos, b_pos = pos_ids[:, ::2], pos_ids[:, 1::2]
    if pad_mask is not None: a_pad, b_pad = pad_mask[:, ::2], pad_mask[:, 1::2]
    
    # 2. Calculate pairwise similarity ALL(A) to ALL(B)
    a_norm = a_metric / a_metric.norm(dim=-1, keepdim=True).clamp(min=1e-5)
    b_norm = b_metric / b_metric.norm(dim=-1, keepdim=True).clamp(min=1e-5)
    similarity = torch.bmm(a_norm, b_norm.transpose(1, 2))  # (batch, len_a, len_b)
    
    # For each A, find best B
    best_scores, best_b_idx = similarity.max(dim=-1)
    
    # Sort A's by highest matching score to *any* B
    sorted_scores_indices = torch.argsort(best_scores, dim=-1, descending=True)
    
    # We will only merge the top `num_to_merge` A tokens.
    a_merge_indices = sorted_scores_indices[:, :num_to_merge]
    
    # We use scatter logic to build the new compressed representations.
    # Because scatter/gather on batched sequence lengths is wildly complex in raw standard torch,
    # we iteratively process the batch (the latent chunks are small so this is very fast).
    
    merged_hidden_list = []
    merged_weights_list = []
    step_tracking_list = []
    merged_pos_list = []
    merged_pad_list = []
    
    for b in range(batch_size):
        len_a, len_b = a_tokens.shape[1], b_tokens.shape[1]
        
        # Get indices of A tokens we ARE merging
        a_merge = a_merge_indices[b]
        
        # Get indices of A tokens we ARE NOT merging
        a_unmerge_mask = torch.ones(len_a, dtype=torch.bool, device=device)
        a_unmerge_mask[a_merge] = False
        a_unmerge = torch.nonzero(a_unmerge_mask).squeeze()
        if a_unmerge.dim() == 0: a_unmerge = a_unmerge.unsqueeze(0)
        
        # Get corresponding B destinations for the merged A's
        b_dests = best_b_idx[b, a_merge]
        
        # Output components
        out_b_hidden = b_tokens[b].clone()
        out_b_weights = b_weights[b].clone()
        out_b_pad = b_pad[b].clone() if pad_mask is not None else None
        
        # Tracking matrix rows: Len_A + Len_B. Cols: Len_A_Unmerged + Len_B
        new_seq_len = len(a_unmerge) + len_b
        step_tracking = torch.zeros((seq_len, new_seq_len), device=device)
        
        # Phase 1: Transfer unmerged A tokens perfectly mapping to the front of the new tensor
        new_a_hidden = a_tokens[b, a_unmerge]
        new_a_weights = a_weights[b, a_unmerge]
        if pos_ids is not None: new_a_pos = a_pos[b, a_unmerge]
        if pad_mask is not None: new_a_pad = a_pad[b, a_unmerge]
        
        for new_idx, old_a_idx in enumerate(a_unmerge.tolist()):
            original_idx = old_a_idx * 2  # A tokens are at even indices
            step_tracking[original_idx, new_idx] = 1.0
            
        # Phase 2: Add merged A tokens into their destination B tokens
        for i, a_idx in enumerate(a_merge.tolist()):
            dest_b_idx = b_dests[i].item()
            
            # Weighted average
            wa = a_weights[b, a_idx]
            wb = out_b_weights[dest_b_idx]
            new_w = wa + wb
            
            out_b_hidden[dest_b_idx] = (a_tokens[b, a_idx] * wa + out_b_hidden[dest_b_idx] * wb) / new_w.clamp(min=1e-5)
            out_b_weights[dest_b_idx] = new_w
            if pad_mask is not None: out_b_pad[dest_b_idx] = out_b_pad[dest_b_idx] and a_pad[b, a_idx]
            
            # Map tracking
            original_a_idx = a_idx * 2
            original_b_idx = (dest_b_idx * 2) + 1
            new_composite_idx = len(a_unmerge) + dest_b_idx
            
            step_tracking[original_a_idx, new_composite_idx] = 1.0
            
        # Phase 3: Map all B tokens (modified and unmodified)
        for old_b_idx in range(len_b):
            original_b_idx = (old_b_idx * 2) + 1
            new_composite_idx = len(a_unmerge) + old_b_idx
            step_tracking[original_b_idx, new_composite_idx] = 1.0
            
        # Concat unmerged A with all B
        merged_hidden_list.append(torch.cat([new_a_hidden, out_b_hidden]))
        merged_weights_list.append(torch.cat([new_a_weights, out_b_weights]))
        step_tracking_list.append(step_tracking)
        
        if pos_ids is not None: merged_pos_list.append(torch.cat([new_a_pos, b_pos[b]]))
        if pad_mask is not None: merged_pad_list.append(torch.cat([new_a_pad, out_b_pad]))
        
    hidden_states = torch.stack(merged_hidden_list)
    token_weights = torch.stack(merged_weights_list)
    step_tracking_matrix = torch.stack(step_tracking_list)
    pos_ids = torch.stack(merged_pos_list) if pos_ids is not None else None
    pad_mask = torch.stack(merged_pad_list) if pad_mask is not None else None
    
    return hidden_states, step_tracking_matrix, token_weights, pos_ids, pad_mask

def unmerge(hidden_states: torch.Tensor, source_tracking_matrix: torch.Tensor) -> torch.Tensor:
    """
    Unmerges a compressed tensor back to its original size using the tracking matrix.
    """
    row_sums = source_tracking_matrix.sum(dim=-1, keepdim=True).clamp(min=1e-5)
    normalized_tracking = source_tracking_matrix / row_sums
    return torch.bmm(normalized_tracking, hidden_states)
