from dit import DiT
from torch import Tensor
import torch
from data import PromoterDataset, ATCG
import pickle 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import time
import tqdm 
from ddsm import *
import torch.nn.functional as F
from torch.optim import Adam
import os
import numpy as np
import wandb
from ddfm import DirichletConditionalFlow, expand_simplex, sample_cond_prob_path, simplex_proj
import copy

torch.autograd.set_detect_anomaly(True)

class ModelParameters:
    hits_path = '/n/groups/marks/users/erik/Promoter_Poet_private/data/hits.pkl'
    query_path = '/n/groups/marks/users/erik/Promoter_Poet_private/data/query.pkl'
    weights_file = '/n/groups/marks/users/erik/prose-ddsm/preprocessing/steps400.cat7.time4.0.samples100000.pth'
    device = 'cuda'
    batch_size = 4
    num_workers = 1
    n_time_steps = 400
    ncat = 7
    num_epochs = 10
    lr = 1e-4
    max_len = 16384
    padding_idx = -100
    output_dir = '/n/groups/marks/users/erik/prose-ddsm/models'
    wandb_project = 'diffusion-prose-flow'
    warmup_steps = 1000
    total_training_steps = None
    preprocess_steps = 10000
    
    mode = 'dirichlet'  # 'dirichlet', 'riemannian', 'ardm', 'lrar'
    alpha_max = 100.0
    prior_pseudocount = 0.1
    flow_temp = 1.0
    num_integration_steps = 50
    validate = False
    alpha_scale = 2.0
    fix_alpha = None

config = ModelParameters()
alphabet = ATCG()

run = wandb.init(
    project=config.wandb_project,
)

# Load data
with open(config.hits_path, "rb") as f:
    hits = pickle.load(f)

with open(config.query_path, "rb") as f:
    query = pickle.load(f)

dataset = PromoterDataset(sequences=hits,
                          queries=query,
                          alphabet=alphabet,
                          max_length=config.max_len)
dataloader = DataLoader(dataset, 
                        batch_size=config.batch_size, 
                        shuffle=True, 
                        num_workers=config.num_workers,
                        collate_fn=dataset.padded_collate_packed)

# Initialize conditional flow
condflow = DirichletConditionalFlow(K=config.ncat)
device = config.device

# Initialize model
flow_model = DiT(n_vocab=config.ncat * 2)
flow_model = flow_model.to(device)

if config.total_training_steps is None:
    config.total_training_steps = config.num_epochs * len(dataloader)
    print(f"Calculated total_training_steps: {config.total_training_steps}")

# Learning rate scheduler
def get_lr_scheduler_lambda(warmup_steps, total_training_steps, base_lr):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
        return max(0.0, 1.0 - progress)
    return lr_lambda

optimizer = Adam(flow_model.parameters(), lr=config.lr)
# lr_scheduler_func = get_lr_scheduler_lambda(config.warmup_steps, config.total_training_steps, config.lr)
# scheduler = LambdaLR(optimizer, lr_lambda=lr_scheduler_func)

# utilities
def sample_flow_trajectory(seq, mode, alphabet_size, alpha_max, prior_pseudocount):
    """Sample flow trajectory for training"""
    B, L = seq.shape
    
    if mode == 'dirichlet':
        xt, alphas = sample_cond_prob_path(config, seq, alphabet_size)
        xt, prior_weights = expand_simplex(xt, alphas, prior_pseudocount)
        return xt, alphas, prior_weights
    
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")

@torch.no_grad()
def dirichlet_flow_inference(model, seq, segment_sizes, alpha_max, num_integration_steps, condflow, prior_pseudocount, flow_temp):
    """Dirichlet flow inference for validation"""
    B, L = seq.shape
    alphabet_size = config.ncat
    
    # Start from Dirichlet prior
    x0 = torch.distributions.Dirichlet(torch.ones(B, L, alphabet_size, device=seq.device)).sample()
    eye = torch.eye(alphabet_size).to(x0.device)
    xt = x0
    
    t_span = torch.linspace(1, alpha_max, num_integration_steps, device=seq.device)
    
    for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
        prior_weight = prior_pseudocount / (s + prior_pseudocount - 1)
        seq_xt = torch.cat([xt * (1 - prior_weight), xt * prior_weight], -1)
        
        logits = model(seq_xt, segment_sizes, s[None].expand(B))
        out_probs = torch.nn.functional.softmax(logits / flow_temp, -1)
        
        c_factor = condflow.c_factor(xt.cpu().numpy(), s.item())
        c_factor = torch.from_numpy(c_factor).to(xt.device)
        
        if torch.isnan(c_factor).any():
            print(f'NAN cfactor: xt.min(): {xt.min()}, out_probs.min(): {out_probs.min()}')
            c_factor = torch.nan_to_num(c_factor)
        
        cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
        flow = (out_probs.unsqueeze(-2) * cond_flows).sum(-1)
        xt = xt + flow * (t - s)
        
        if not torch.allclose(xt.sum(2), torch.ones((B, L), device=xt.device), atol=1e-4) or not (xt >= 0).all():
            print(f'WARNING: xt.min(): {xt.min()}. Projecting to simplex.')
            xt = simplex_proj(xt)
    
    return logits, x0

print("--- Starting Flow Matching Training ---")
tqdm_epoch = tqdm.trange(config.num_epochs, desc="Epochs")

for epoch in tqdm_epoch:
    flow_model.train()
    avg_loss = 0.
    num_items = 0
    epoch_start_time = time.time()
    
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc="Training Batch", leave=False)):
        tokens = batch["tokens"].to(device)
        segment_sizes = batch["segment_sizes"].to(device)
        B, L = tokens.shape
        
        # Handle padding
        padding_mask = (tokens == config.padding_idx)
        tokens_safe = tokens.clone()
        tokens_safe[padding_mask] = 0
        
        # Convert to one-hot
        seq_one_hot = F.one_hot(tokens_safe, num_classes=config.ncat).float()
        seq_one_hot.masked_fill_(padding_mask.unsqueeze(-1), 0.0)
        
        # Sample flow trajectory
        if config.mode == 'dirichlet':
            xt, alphas, prior_weights = sample_flow_trajectory(
                tokens, config.mode, config.ncat, config.alpha_max, config.prior_pseudocount
            )
        else:
            raise NotImplementedError
        # print(xt.shape, alphas.shape, segment_sizes.shape)
        # torch.Size([4, 16384, 14]) torch.Size([4]) torch.Size([4, 18]) 
        # Forward pass through model
        logits = flow_model(xt, segment_sizes, alphas)
        
        # For flow-based modes, use cross-entropy loss against ground truth
        target_seq = tokens_safe
        losses = F.cross_entropy(logits.transpose(1, 2), target_seq, reduction='none')
    
        # Apply masking
        seq_lengths = segment_sizes.sum(dim=1).long()
        mask = torch.arange(losses.shape[1], device=device)[None, :] >= seq_lengths[:, None]
        losses = losses.masked_fill(mask, 0.0)
        
        # Calculate mean loss over non-padded elements
        num_valid_elements = (~mask).sum()
        loss = losses.sum() / num_valid_elements if num_valid_elements > 0 else torch.tensor(0.0, device=device)
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        wandb.log({
            "loss": loss.item(), 
            "alpha_mean": alphas.mean().item() if isinstance(alphas, torch.Tensor) else 0.0
        })
        
        avg_loss += loss.item() * B
        num_items += B
    
    epoch_end_time = time.time()
    avg_loss /= num_items
    tqdm_epoch.set_description(f'Epoch {epoch+1}/{config.num_epochs} | Avg Loss: {avg_loss:.5f} | Time: {epoch_end_time - epoch_start_time:.2f}s')
    
    # Validation step
    if (epoch + 1) % 5 == 0 and config.validate:
        flow_model.eval()
        val_losses = []
        val_recoveries = []
        
        print(f"Epoch {epoch+1}: Running validation...")
        
        for val_batch_idx, val_batch in enumerate(tqdm.tqdm(dataloader, desc="Validation", leave=False)):
            if val_batch_idx >= 10:  # Limit validation batches
                break
                
            tokens = val_batch["tokens"].to(device)
            segment_sizes = val_batch["segment_sizes"].to(device)
            B, L = tokens.shape
            
            padding_mask = (tokens == config.padding_idx)
            tokens_safe = tokens.clone()
            tokens_safe[padding_mask] = 0
            
            # Generate predictions using flow inference
            if config.mode == 'dirichlet':
                logits_pred, _ = dirichlet_flow_inference(
                    flow_model, tokens_safe, segment_sizes, config.alpha_max, 
                    config.num_integration_steps, condflow, config.prior_pseudocount, config.flow_temp
                )
            
            else:
                raise NotImplementedError
            
            seq_pred = torch.argmax(logits_pred, dim=-1)
            
            # Calculate validation metrics
            val_loss = F.cross_entropy(logits_pred.transpose(1, 2), tokens_safe, reduction='none')
            seq_lengths = segment_sizes.sum(dim=1).long()
            mask = torch.arange(val_loss.shape[1], device=device)[None, :] >= seq_lengths[:, None]
            val_loss = val_loss.masked_fill(mask, 0.0)
            val_losses.append(val_loss.sum() / (~mask).sum())
            
            # Calculate recovery rate
            recovery = seq_pred.eq(tokens_safe).float()
            recovery = recovery.masked_fill(mask, 0.0)
            val_recoveries.append(recovery.sum() / (~mask).sum())
        
        avg_val_loss = torch.stack(val_losses).mean().item()
        avg_val_recovery = torch.stack(val_recoveries).mean().item()
        
        wandb.log({
            "val_loss": avg_val_loss,
            "val_recovery": avg_val_recovery,
            "epoch": epoch + 1
        })
        
        print(f"Validation - Loss: {avg_val_loss:.5f}, Recovery: {avg_val_recovery:.3f}")
        flow_model.train()
    
    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        save_path = os.path.join(config.output_dir, f"dit_flow_epoch_{epoch+1}.pth")
        torch.save({
            'model_state_dict': flow_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch + 1,
            'config': config.__dict__
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

print("Training finished.")
final_save_path = os.path.join(config.output_dir, "dit_flow_final.pth")
torch.save({
    'model_state_dict': flow_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config.__dict__
}, final_save_path)
print(f"Saved final model to {final_save_path}")