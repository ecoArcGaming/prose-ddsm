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
    device = 'cuda'
    batch_size = 8
    num_workers = 1
    ncat = 7
    num_epochs = 10
    lr = 1e-4
    max_len = 8192
    padding_idx = -100
    output_dir = '/n/groups/marks/users/erik/prose-ddsm/models'
    wandb_project = 'diffusion-prose-flow'
    warmup_steps = 1000
    total_training_steps = None
    mode = 'dirichlet'  # 'dirichlet', 'riemannian', 'ardm', 'lrar'
    alpha_max = 100.0
    prior_pseudocount = 0.1
    flow_temp = 1.0
    num_integration_steps = 50
    validate = True
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

train_seq, train_query, val_seq, val_query = (
        PromoterDataset._train_validation_split(19, hits, query)
    )

train_dataset = PromoterDataset(train_seq, train_query, alphabet, config.max_len)
val_dataset = PromoterDataset(val_seq, val_query, alphabet, config.max_len)

train_dataloader = DataLoader(train_dataset, 
                        batch_size=config.batch_size, 
                        shuffle=True, 
                        num_workers=config.num_workers,
                        collate_fn=train_dataset.padded_collate_packed)
val_dataloader = DataLoader(train_dataset, 
                        batch_size=config.batch_size, 
                        shuffle=False, 
                        num_workers=config.num_workers,
                        collate_fn=val_dataset.padded_collate_packed)

# Initialize conditional flow
condflow = DirichletConditionalFlow(K=config.ncat)
device = config.device

# Initialize model
flow_model = DiT(n_vocab=config.ncat)
flow_model = flow_model.to(device)

if config.total_training_steps is None:
    config.total_training_steps = config.num_epochs * len(train_dataloader)
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
lr_scheduler_func = get_lr_scheduler_lambda(config.warmup_steps, config.total_training_steps, config.lr)
scheduler = LambdaLR(optimizer, lr_lambda=lr_scheduler_func)

# utilities
def sample_flow_trajectory(seq, mode, alphabet_size, alpha_max, prior_pseudocount):
    """Sample flow trajectory for training"""
    
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
    
    for batch_idx, batch in enumerate(tqdm.tqdm(train_dataloader, desc="Training Batch", leave=False)):
        tokens = batch["tokens"].to(device)
        segment_sizes = batch["segment_sizes"].to(device)
        B, L = tokens.shape
        
        # Sample flow trajectory
        if config.mode == 'dirichlet':
            xt, alphas, prior_weights = sample_flow_trajectory(
                tokens, config.mode, config.ncat, config.alpha_max, config.prior_pseudocount
            )
        else:
            raise NotImplementedError
       
        logits = flow_model(xt, segment_sizes, alphas)
        
        # cross-entropy loss against ground truth
        loss = F.cross_entropy(logits.transpose(1, 2), 
                                 tokens, reduction='mean',  
                                 ignore_index=config.padding_idx)

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        wandb.log({
            "loss": loss.item(), 
            "alpha_mean": alphas.mean().item() 
        })
        
        avg_loss += loss.item() * B
        num_items += B
    
    epoch_end_time = time.time()
    avg_loss /= num_items
    tqdm_epoch.set_description(f'Epoch {epoch+1}/{config.num_epochs} | Avg Loss: {avg_loss:.5f} | Time: {epoch_end_time - epoch_start_time:.2f}s')
    
    # Validation step
    if (epoch + 1) % 1 == 0 and config.validate:
        flow_model.eval()
        val_losses = []
        val_recoveries = []
        
        print(f"Epoch {epoch+1}: Running validation...")
        
        with torch.no_grad():  # Add no_grad context for validation
            for val_batch_idx, val_batch in enumerate(tqdm.tqdm(val_dataloader, desc="Validation", leave=False)):
                if val_batch_idx >= 10:  # Limit validation batches
                    break
                    
                tokens = val_batch["tokens"].to(device)
                segment_sizes = val_batch["segment_sizes"].to(device)
                B, L = tokens.shape
                
                # Create padding mask
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
                
                # Calculate validation loss - only on non-padded tokens
                val_loss = F.cross_entropy(
                    logits_pred.transpose(1, 2), 
                    tokens, 
                    reduction='mean',  # Changed to 'none' to get per-token losses
                    ignore_index=config.padding_idx
                )
                
                # Create sequence length mask
                seq_lengths = segment_sizes.sum(dim=1).long()
                length_mask = torch.arange(L, device=device)[None, :] < seq_lengths[:, None]
                
                # Apply mask and calculate mean loss
                val_losses.append(val_loss.item())
                
                # Calculate recovery rate - only on valid (non-padded) tokens
                recovery = seq_pred.eq(tokens).float()
                valid_recovery = recovery[length_mask & ~padding_mask]
                if valid_recovery.numel() > 0:
                    val_recoveries.append(valid_recovery.mean())
        
        # Calculate averages
        if val_losses:
            avg_val_loss = torch.stack(val_losses).mean().item()
        else:
            avg_val_loss = float('inf')
            
        if val_recoveries:
            avg_val_recovery = torch.stack(val_recoveries).mean().item()
        else:
            avg_val_recovery = 0.0
        
        wandb.log({
            "val_loss": avg_val_loss,
            "val_recovery": avg_val_recovery,
            "epoch": epoch + 1
        })
        
        print(f"Validation - Loss: {avg_val_loss:.5f}, Recovery: {avg_val_recovery:.3f}")
        flow_model.train()
    
    # Save checkpoint
    if (epoch + 1) % 1 == 0:
        save_path = os.path.join(config.output_dir, f"dit_flow_epoch_{epoch+1}.pth")
        torch.save({
            'model_state_dict': flow_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
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