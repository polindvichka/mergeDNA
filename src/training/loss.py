import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.mergedna import MergeDNA

class MergeDNALoss(nn.Module):
    """
    Implements the advanced MTR and AMTM loss formulations for MergeDNA.
    """
    def __init__(self, model: MergeDNA):
        super().__init__()
        self.model = model
        self.mask_token_id = model.config.mask_token_id

    def sample_amtm_masks(self, s_latent: torch.Tensor, s_local: torch.Tensor):
        """
        Samples the informative tokens using the S_latent grouping probability
        and projects the mask back to the N base nucleotides using S_local.
        """
        B, L, K = s_latent.shape
        _, N, _ = s_local.shape
        
        # 1. Weight groups heavily penalizing uniform blocks: P ~ 1 / g^2
        g = s_latent.sum(dim=1).clamp(min=1.0) # (B, K)
        group_weights = 1.0 / (g * g)
        
        # 2. Project K weights back to L tokens
        token_weights = torch.bmm(s_latent, group_weights.unsqueeze(-1)).squeeze(-1) # (B, L)
        probs = token_weights / token_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # 3. Sample exactly K informative local tokens without replacement
        k_sample = min(K, L)
        
        mask_L = torch.zeros(B, L, dtype=torch.bool, device=s_latent.device)
        for b in range(B):
            idx = torch.multinomial(probs[b], num_samples=k_sample, replacement=False)
            mask_L[b, idx] = True
            
        # 4. Project L mask back to N tokens
        mask_N = torch.bmm(s_local, mask_L.float().unsqueeze(-1)).squeeze(-1) > 0
        return mask_L, mask_N

    def forward(self, input_ids: torch.Tensor, local_target_ratio: float = None):
        B, N = input_ids.shape
        
        # 1. Standard Merged Token Reconstruction (MTR) Loss
        out_mtr = self.model(input_ids, local_target_ratio=local_target_ratio)
        loss_mtr = F.cross_entropy(out_mtr["logits"].permute(0, 2, 1), input_ids, reduction="mean")
        
        # 2. Latent MTR Loss (Fix Local Encoder, train Latent only)
        # As per paper Eq. 8, we freeze the local encoder outputs and run just the latent + decode pass
        z_L_detached = out_mtr["z_L"].detach()
        sizes_L_detached = out_mtr["sizes_L"].detach()
        s_local_detached = out_mtr["s_local"].detach()
        
        # We need a dedicated pass for the latent decoder to compute its own MTR loss
        target_k = max(1, int(z_L_detached.size(1) * self.model.config.latent_target_ratio))
        s_latent_init = torch.eye(z_L_detached.size(1), device=input_ids.device).unsqueeze(0).expand(B, -1, -1)
        
        # Manually reconstruct the latent-only branch to get logits for Latent MTR
        x_latent = z_L_detached
        tok_weights = sizes_L_detached
        
        # Latent encode
        for i, layer_block in enumerate(self.model.latent_encoder):
            x_latent, metric = layer_block(x_latent, token_weights=tok_weights, return_metric=True)
            layers_left = len(self.model.latent_encoder) - i
            target_step = max(target_k, int(x_latent.size(1) * (1.0 - (1.0 - (target_k / x_latent.size(1)) ** (1.0 / layers_left)))))
            x_latent, s_latent_init, tok_weights, _, _ = global_merge(x_latent, tok_weights, metric, target_step)
            
        z_K_latent = x_latent
        
        # Latent decode
        x_latent = unmerge(z_K_latent, s_latent_init)
        for layer_block in self.model.latent_decoder:
            x_latent = layer_block(x_latent)
        
        # Local decode (reconstruct N bases)
        x_latent = unmerge(x_latent, s_local_detached)
        for layer_block in self.model.local_decoder:
            x_latent = layer_block(x_latent)
            
        logits_latent = self.model.head(x_latent)
        loss_latent_mtr = F.cross_entropy(logits_latent.permute(0, 2, 1), input_ids, reduction="mean")
        
        # 3. Adaptive Masked Token Modeling (AMTM) Loss
        s_latent = out_mtr["s_latent"].detach()
        s_local = out_mtr["s_local"].detach()
        
        mask_L, mask_N = self.sample_amtm_masks(s_latent, s_local)
        
        masked_inputs = input_ids.clone()
        masked_inputs[mask_N] = self.mask_token_id
        
        # Forward AMTM (disable latent token merging conceptually for pure decoding)
        out_amtm = self.model(masked_inputs, disable_latent_merge=True, local_target_ratio=local_target_ratio)
        
        losses_amtm = []
        for b in range(B):
            mask_n_b = mask_N[b]
            mask_l_b = mask_L[b]
            
            # The denominator is K (number of informative tokens chosen), NOT the number of raw bases masked!
            k_selected = max(1, int(mask_l_b.sum().item()))
            
            if mask_n_b.sum() > 0:
                logits_masked = out_amtm["logits"][b, mask_n_b]
                targets_masked = input_ids[b, mask_n_b]
                
                ce_sum = F.cross_entropy(logits_masked, targets_masked, reduction="sum")
                losses_amtm.append(ce_sum / k_selected)
                
        loss_amtm = torch.stack(losses_amtm).mean() if losses_amtm else torch.tensor(0.0, device=input_ids.device)
        
        # Total Loss (Eq. 8)
        lambda_val = self.model.config.lambda_latent_mtr
        total_loss = loss_mtr + (lambda_val * loss_latent_mtr) + loss_amtm
        
        return total_loss, {
            "mtr_loss": loss_mtr.item(),
            "latent_mtr_loss": loss_latent_mtr.item(),
            "amtm_loss": loss_amtm.item()
        }
