import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import pickle
import os
import numpy as np
import tqdm

from dit import DiT
from data import PromoterDataset, ATCG
from ddfm import DirichletConditionalFlow, expand_simplex, sample_cond_prob_path, simplex_proj

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

class FlowMatchingModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.__dict__)
        
        # Initialize model
        self.flow_model = DiT(n_vocab=config.ncat)
        
        # Initialize conditional flow
        self.condflow = DirichletConditionalFlow(K=config.ncat)
        
        # Store alphabet
        self.alphabet = ATCG()
        
        # Will be set in setup()
        self.total_training_steps = None
        
    def setup(self, stage=None):
        """Called at the beginning of fit, validate, test, or predict"""
        if stage == 'fit' or stage is None:
            # Calculate total training steps
            if self.total_training_steps is None:
                train_loader = self.trainer.datamodule.train_dataloader()
                self.total_training_steps = self.config.num_epochs * len(train_loader)
                print(f"Calculated total_training_steps: {self.total_training_steps}")
    
    def sample_flow_trajectory(self, seq, mode, alphabet_size, alpha_max, prior_pseudocount):
        """Sample flow trajectory for training"""
        if mode == 'dirichlet':
            xt, alphas = sample_cond_prob_path(self.config, seq, alphabet_size)
            xt, prior_weights = expand_simplex(xt, alphas, prior_pseudocount)
            return xt, alphas, prior_weights
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")
    
    @torch.no_grad()
    def dirichlet_flow_inference(self, seq, segment_sizes, alpha_max, num_integration_steps, prior_pseudocount, flow_temp):
        """Dirichlet flow inference for validation"""
        model_dtype = next(self.flow_model.parameters()).dtype

        B, L = seq.shape
        alphabet_size = self.config.ncat
        
        # Start from Dirichlet prior
        x0 = torch.distributions.Dirichlet(torch.ones(B, L, alphabet_size, device=seq.device)).sample()
        eye = torch.eye(alphabet_size).to(x0.device)
        xt = x0
        
        t_span = torch.linspace(1, alpha_max, num_integration_steps, device=seq.device)
        
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            prior_weight = prior_pseudocount / (s + prior_pseudocount - 1)
            seq_xt = torch.cat([xt * (1 - prior_weight), xt * prior_weight], -1)
            
            logits = self.flow_model(seq_xt.to(dtype=model_dtype), segment_sizes.to(dtype=model_dtype), s[None].expand(B).to(dtype=model_dtype))
            out_probs = torch.nn.functional.softmax(logits / flow_temp, -1)
            
            c_factor = self.condflow.c_factor(xt.cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt.device)
            
            if torch.isnan(c_factor).any():
                print(f'NAN cfactor: xt.min(): {xt.min()}, out_probs.min(): {out_probs.min()}')
                c_factor = torch.nan_to_num(c_factor)
            
            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (out_probs.unsqueeze(-2) * cond_flows).sum(-1)
            xt = xt + flow * (t - s)
            
            if not torch.allclose(xt.sum(2), torch.ones((B, L), device=xt.device, dtype=xt.dtype), atol=1e-4) or not (xt >= 0).all():
                print(f'WARNING: xt.min(): {xt.min()}. Projecting to simplex.')
                xt = simplex_proj(xt)
        
        return logits, x0
    
    @torch.no_grad()
    def inpaint_last_sequence(self, partial_seq, segment_sizes, alpha_max, 
                                     num_integration_steps, prior_pseudocount, flow_temp):
        """
        Generate the last segment of a sequence-of-sequences (RePaint approach).
        
        Args:
            partial_seq: [B, L] - sequence with some positions to be filled
            segment_sizes: [B, num_segments] - segment information
        """
        model_dtype = next(self.flow_model.parameters()).dtype
        device = partial_seq.device
        
        B, L = partial_seq.shape
        alphabet_size = self.config.ncat
        
        # Convert known parts to one-hot
        eye = torch.eye(alphabet_size, device=device, dtype=model_dtype)
        known_probs = eye[partial_seq.long()]  # [B, L, alphabet_size]
        
        # Start from Dirichlet prior
        xt = torch.distributions.Dirichlet(
            torch.ones(B, L, alphabet_size, device=device, dtype=model_dtype)
        ).sample()
    
        # start_indices_last_segment[b] = sum of lengths of segments 0 to N-2 for batch item b.
        start_indices_last_segment = segment_sizes[:, :-1].sum(dim=1) # Shape: [B]
        indices_matrix = torch.arange(L, device=device).expand(B, -1) # Shape: [B, L]
        mask = indices_matrix >= start_indices_last_segment.unsqueeze(1)
        
        # Replace known positions with their true values
        mask_expanded = mask.unsqueeze(-1).float()  # [B, L, 1]
        xt = mask_expanded * xt + (1 - mask_expanded) * known_probs
        
        t_span = torch.linspace(1, alpha_max, num_integration_steps, device=device, dtype=model_dtype)
        
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            prior_weight = torch.tensor(prior_pseudocount / (s + prior_pseudocount - 1), 
                                      device=device, dtype=model_dtype)
            
            seq_xt = torch.cat([xt * (1 - prior_weight), xt * prior_weight], -1)
            time_tensor = s.unsqueeze(0).expand(B).to(dtype=model_dtype)
            
            logits = self.flow_model(seq_xt, segment_sizes, time_tensor)
            out_probs = torch.nn.functional.softmax(logits / flow_temp, -1)
            
            # Compute flow
            c_factor = self.condflow.c_factor(xt.cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(device=device, dtype=model_dtype)
            
            if torch.isnan(c_factor).any():
                c_factor = torch.nan_to_num(c_factor)
            
            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (out_probs.unsqueeze(-2) * cond_flows).sum(-1)
            xt_new = xt + flow * (t - s)
            
            # Only update masked positions, keep known positions fixed
            xt = mask_expanded * xt_new + (1 - mask_expanded) * known_probs
            
            if not torch.allclose(xt.sum(2), torch.ones((B, L), device=device, dtype=model_dtype), atol=1e-4) or not (xt >= 0).all():
                xt = simplex_proj(xt)
        
        return logits, xt
    
    def training_step(self, batch, batch_idx):
        tokens = batch["tokens"]
        segment_sizes = batch["segment_sizes"]
        
        # Sample flow trajectory
        if self.config.mode == 'dirichlet':
            xt, alphas, prior_weights = self.sample_flow_trajectory(
                tokens, self.config.mode, self.config.ncat, 
                self.config.alpha_max, self.config.prior_pseudocount
            )
        else:
            raise NotImplementedError
       
        logits = self.flow_model(xt, segment_sizes, alphas)
        
        # Cross-entropy loss against ground truth
        loss = F.cross_entropy(
            logits.transpose(1, 2), 
            tokens, 
            reduction='mean',  
            ignore_index=self.config.padding_idx
        )
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('alpha_mean', alphas.mean(), on_step=True, on_epoch=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Limit validation batches for efficiency
        # if batch_idx >= 10:
        #     return None
            
        tokens = batch["tokens"]
        segment_sizes = batch["segment_sizes"]
        B, L = tokens.shape
        
        # Create padding mask
        padding_mask = (tokens == self.config.padding_idx)
        tokens_safe = tokens.clone()
        tokens_safe[padding_mask] = 0
        
        # Generate predictions using flow inference
        if self.config.mode == 'dirichlet':
            logits_pred, _ = self.dirichlet_flow_inference(
                tokens_safe, segment_sizes, self.config.alpha_max, 
                self.config.num_integration_steps, self.config.prior_pseudocount, 
                self.config.flow_temp
            )
        else:
            raise NotImplementedError
        
        seq_pred = torch.argmax(logits_pred, dim=-1)
        
        # Calculate validation loss - only on non-padded tokens
        val_loss = F.cross_entropy(
            logits_pred.transpose(1, 2), 
            tokens, 
            reduction='mean',
            ignore_index=self.config.padding_idx
        )
        
        # Create sequence length mask
        seq_lengths = segment_sizes.sum(dim=1).long()
        length_mask = torch.arange(L, device=self.device)[None, :] < seq_lengths[:, None]
        
        # Calculate recovery rate - only on valid (non-padded) tokens
        recovery = seq_pred.eq(tokens).float()
        valid_recovery = recovery[length_mask & ~padding_mask]
        
        if valid_recovery.numel() > 0:
            recovery_rate = valid_recovery.mean()
        else:
            recovery_rate = torch.tensor(0.0, device=self.device)
        
        # Log validation metrics
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recovery', recovery_rate, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': val_loss, 'val_recovery': recovery_rate}
    
    def configure_optimizers(self):
        optimizer = Adam(self.flow_model.parameters(), lr=self.config.lr)
        
        # Learning rate scheduler
        def get_lr_scheduler_lambda(warmup_steps, total_training_steps):
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
                return max(0.0, 1.0 - progress)
            return lr_lambda
        
        lr_scheduler_func = get_lr_scheduler_lambda(self.config.warmup_steps, self.total_training_steps)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_scheduler_func)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def on_before_optimizer_step(self, optimizer):
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), max_norm=1.0)

class PromoterDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.alphabet = ATCG()
        
    def setup(self, stage=None):
        # Load data
        with open(self.config.hits_path, "rb") as f:
            hits = pickle.load(f)

        with open(self.config.query_path, "rb") as f:
            query = pickle.load(f)

        train_seq, train_query, val_seq, val_query = (
            PromoterDataset._train_validation_split(19, hits, query)
        )

        self.train_dataset = PromoterDataset(train_seq, train_query, self.alphabet, self.config.max_len)
        self.val_dataset = PromoterDataset(val_seq, val_query, self.alphabet, self.config.max_len)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=self.config.num_workers,
            collate_fn=self.train_dataset.padded_collate_packed
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers,
            collate_fn=self.val_dataset.padded_collate_packed
        )

def main():
    # Configuration
    config = ModelParameters()
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        log_model=True
    )
    
    # Initialize data module
    data_module = PromoterDataModule(config)
    
    # Initialize model
    model = FlowMatchingModule(config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.output_dir,
        filename='dit_flow_epoch_{epoch:02d}_val_loss_{val_loss:.3f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,  # Save checkpoint every epoch
        save_on_train_epoch_end=True  # Save at end of each epoch
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator='gpu' if config.device == 'cuda' else 'cpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        val_check_interval=100,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train the model
    print("--- Starting Flow Matching Training with PyTorch Lightning ---")
    trainer.fit(model, data_module)
    
    # Save final model
    final_save_path = os.path.join(config.output_dir, "dit_flow_final.ckpt")
    trainer.save_checkpoint(final_save_path)
    print(f"Saved final model to {final_save_path}")

if __name__ == "__main__":
    main()