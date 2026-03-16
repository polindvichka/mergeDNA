import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import math
from tqdm import tqdm

from src.config import MergeDNAConfig
from src.data.dataset import ExampleDNADataset
from src.models.mergedna import MergeDNA
from src.training.loss import MergeDNALoss

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """
    Implements the linear warmup + cosine annealing schedule from the paper.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(total_steps_override=None):
    # 1. Configuration
    config = MergeDNAConfig()
    total_steps = total_steps_override if total_steps_override is not None else config.total_steps
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Data
    dataset = ExampleDNADataset(num_samples=1000, seq_length=512)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 3. Model & Loss
    model = MergeDNA(config).to(device)
    loss_fn = MergeDNALoss(model)
    
    # 4. Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay
    )
    
    # 5. Scheduler
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, total_steps)
    
    # 6. Training Loop
    model.train()
    step = 0
    
    torch.autograd.set_detect_anomaly(True)
    pbar = tqdm(total=total_steps, desc="Training DNA")
    
    while step < total_steps:
        for batch in dataloader:
            if step >= total_steps:
                break
                
            input_ids = batch.to(device)
            
            # Stochastic Ratio Sampling (Insight from Paper Sec 3.3)
            random_ratio = random.uniform(config.local_ratio_min, config.local_ratio_max)
            
            optimizer.zero_grad()
            
            # Forward pass & Loss calculation (3 components: MTR + Latent MTR + AMTM)
            total_loss, metrics = loss_fn(input_ids, local_target_ratio=random_ratio)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            step += 1
            pbar.update(1)
            pbar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "MTR": f"{metrics['mtr_loss']:.4f}",
                "AMTM": f"{metrics['amtm_loss']:.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Periodic Evaluation (simplified)
            if step % 1000 == 0:
                print(f"\nStep {step}: Total Loss = {total_loss.item():.4f}")
                print(f"Metrics: {metrics}")

    pbar.close()
    print("Training complete!")

if __name__ == "__main__":
    import sys
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else None
    train(total_steps_override=steps)
