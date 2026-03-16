import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import math
import sys
import os
from tqdm import tqdm

# Add project root to path so 'src' can be imported easily
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.dont_write_bytecode = True

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

def train(steps=None, seq_len=None, batch_size=None, device_str=None):
    # 1. Configuration
    config = MergeDNAConfig()
    total_steps = steps if steps is not None else config.total_steps
    
    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Data
    s_len = seq_len if seq_len is not None else 512
    b_size = batch_size if batch_size is not None else 4
    
    dataset = ExampleDNADataset(num_samples=1000, seq_length=s_len)
    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True)
    
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
    
    last_batch = None
    while step < total_steps:
        for batch in dataloader:
            if step >= total_steps:
                break
                
            input_ids = batch.to(device)
            last_batch = input_ids
            
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
    print("FINAL SAMPLE PREDICTION")
    
    # Let's decode a sample from the last batch
    # DNA Map: 0:A, 1:C, 2:G, 3:T
    dna_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'P', 5: 'M'}
    
    # Get predictions for the first sample in the last batch
    if last_batch is not None:
        with torch.no_grad():
            out = model(last_batch)
            logits = out["logits"]
            preds = torch.argmax(logits[0], dim=-1)
            
            sample_input = "".join([dna_map.get(i.item(), '?') for i in last_batch[0][:40]])
            sample_output = "".join([dna_map.get(i.item(), '?') for i in preds[:40]])
            
            print(f"INPUT (First 40bp):  {sample_input}")
            print(f"OUTPUT (Predictions): {sample_output}")
    
    print("Training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Support both 'python train.py 5' and 'python train.py --steps 5'
    parser.add_argument("pos_steps", type=int, nargs="?", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    # Resolve steps: positional takes precedence, then flag, then default 100
    final_steps = 100
    if args.pos_steps is not None:
        final_steps = args.pos_steps
    elif args.steps is not None:
        final_steps = args.steps

    train(steps=final_steps, seq_len=args.seq_len, batch_size=args.batch_size, device_str=args.device)
