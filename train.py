from dit import DiT
from torch import Tensor
import torch
from data import PromoterDataset, ATCG
import pickle 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import tqdm 
from ddsm import *
import torch.nn.functional as F
from torch.optim import Adam
import os
import numpy as np

# device = 'cuda'
# model = DiT(n_vocab=8).to(device)
# xs =  Tensor([[0, 1, 2, 3, 0, 1, 2, 2, 3, 0, 4, 5]]).long().to(device)
# segment_sizes = Tensor([[4, 5, 3]]).to(device)
# time = Tensor([1]).to(device)
# out = model(xs, segment_sizes, time)
# print(xs.shape)
# print(out.shape)
# breakpoint()

class ModelParameters:
    hits_path = 'data/hits.pkl'
    query_path = 'data/query.pkl'
    weights_file = '/home/erik/ddsm/steps400.cat2.time4.0.samples10000.pth'
    device = 'cuda'
    batch_size = 2
    num_workers = 1
    n_time_steps = 400
    random_order = False
    speed_balanced = True
    ncat = 7
    num_epochs = 200
    lr = 5e-4
    max_len = 8000
    padding_idx = -100
    output_dir = '.'

config = ModelParameters()
alphabet = ATCG()

# with open(config.hits_path, "rb") as f:
#     hits = pickle.load(f)

# with open(config.query_path, "rb") as f:
#     query = pickle.load(f)

h = {
        "OXKSZ": {"ACAGAGTAACTGC", "CACGCAAGCGACTA", "GGGCGGGTAGTACCC"},
        "GHXGF": {"AAAATGGTCTGTC", "AATAAGTGAG", "CGCGACGCCATAGTT"},
        "XSVNO": {"CCGATCCCGCCCGC", "GCGGGCCGGCATGCC", "TTTAGTTGTGTT"},
        "LPTSW": {"ATTGCTGGACA", "GCAGTGCCAGTTTC", "GTCATCGTGCACAT"},
        "SAPJM": {"AGAGGTACCC", "CGGGTGAAATT", "CTGGTCACGAGTT"},
        "KNGBV": {"AGCACGCACG", "GAGCGTATCGCAGC"},
        "YTWYS": {"AATGCAACTGGTT", "ATGAAATTATT", "GTACAGACCC"},
        "IMXVE": {
            "TAGTGCAACC",
            "TGAGACGGACGAT",
        },
        "QDKFG": {"CAGGTTTGGCCTGT", "CCCAGTTACCA", "GTTTGACTGC"},
        "GDLYH": {"AACGACAAGCATG", "TACCCGAGTGTAT", "TGGGATCTCA"},
        "XQIRL": {"CACGTGGCGCCGCTT", "CCGTTTTATTG", "GTCCTTACATGCCCC"},
        "BSKUP": {"GAGCCTCTTGCG", "GAGCGTATCGCAGC"},
    }
q = {
    "OXKSZ": "ACAGAGTAACTGC",
    "GHXGF": "AAAATGGTCTGTC",
    "XSVNO": "CCGATCCCGCCCGC",
    "LPTSW": "ATTGCTGGACA",
    "SAPJM": "CTGGTCACGAGTT",
    "KNGBV": "CGGGTAGTTGCGAAC",
    "YTWYS": "GTACAGACCC",
    "IMXVE": "TAGTGCAACC",
    "QDKFG": "GTTTGACTGC",
    "GDLYH": "TGGGATCTCA",
    "XQIRL": "GTCCTTACATGCCCC",
    "BSKUP": "TCGACGAATG",
}

dataset = PromoterDataset(sequences=h,
                          queries=q,
                          alphabet=alphabet,
                          max_length = config.max_len)
dataloader = DataLoader(dataset, 
                        batch_size=config.batch_size, 
                        shuffle=True, 
                        num_workers=config.num_workers,
                        collate_fn=dataset.padded_collate_packed)

v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = torch.load(config.weights_file)
v_one = v_one.cpu().expand(-1, -1,config.ncat - 1 )
v_zero = v_zero.cpu().expand(-1, -1,config.ncat - 1 )
v_one_loggrad = v_one_loggrad.cpu().expand(-1, -1,config.ncat - 1 )
v_zero_loggrad = v_zero_loggrad.cpu().expand(-1, -1,config.ncat - 1 )
timepoints = timepoints.cpu()
alpha = torch.ones(config.ncat - 1).float()
beta =  torch.arange(config.ncat - 1, 0, -1).float()
device = "cuda"
sb = UnitStickBreakingTransform()

print("--- Stage 1: Calculating Time-Dependent Weights ---")
time_dependent_cums = torch.zeros(config.n_time_steps).to(device)
time_dependent_counts = torch.zeros(config.n_time_steps).to(device)
num_items_stage1 = 0

# with torch.no_grad(): # No gradients needed for weight calculation
for batch in tqdm.tqdm(dataloader, desc="Stage 1 Weights"):
    tokens = batch["tokens"].to(device) # Shape: [B, L]
    # print(tokens.shape) # torch.Size([2, 7992])
    B, L = tokens.shape
    
    padding_mask = (tokens == config.padding_idx) # Shape [B, L], True where padded
    tokens_safe = tokens.clone()
    tokens_safe[padding_mask] = 0 

    # Convert tokens to one-hot encoding
    x_one_hot = F.one_hot(tokens_safe, num_classes=config.ncat).float() # Shape: [B, L, V]
    x_one_hot.masked_fill_(padding_mask.unsqueeze(-1), 0.0)
    x_one_hot = x_one_hot[..., :config.ncat]
    # Sample random time indices
    random_t_idx = torch.randint(0, config.n_time_steps, (B,), device=device) # Discrete indices
    # print(x_one_hot.shape, random_t_idx.shape, alpha.shape, v_zero.shape) 
    # torch.Size([2, 7998, 7]) torch.Size([2]) torch.Size([6]),  torch.Size([10000, 400, 1])
    
    perturbed_x, perturbed_x_grad = diffusion_factory(
        x_one_hot.cpu(), random_t_idx.cpu(), v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta
    )
    perturbed_x = perturbed_x.to(device)
    perturbed_x_grad = perturbed_x_grad.to(device)

    # Calculate contribution to weights (using stick-breaking space)
    if config.random_order:
            raise NotImplementedError("Random order logic needs careful implementation matching gx_to_gv")
    else:
        perturbed_v = sb._inverse(perturbed_x, prevent_nan=True) # Shape [B, L, V-1]
        grad_v = gx_to_gv(perturbed_x_grad, perturbed_x) # Shape [B, L, V-1]

    # Weighting factor (from original script)
    if config.speed_balanced:
        s_weights = 2 / (torch.ones(config.ncat - 1, device=device) + torch.arange(config.ncat - 1, 0, -1, device=device).float())
    else:
        s_weights = torch.ones(config.ncat - 1, device=device)

    # Calculate squared magnitude in v-space, weighted
    variance_term = (perturbed_v * (1 - perturbed_v) * s_weights * grad_v**2) # Shape [B, L, V-1]

    # Average over sequence length and V-1 dimensions, sum contributions per time index
    batch_variances = variance_term.sum(dim=[1, 2]) # Sum over L and V-1 -> Shape [B]

    # Use scatter_add_ to sum variances for each time index present in the batch
    time_dependent_cums.scatter_add_(0, random_t_idx, batch_variances)
    time_dependent_counts.scatter_add_(0, random_t_idx, torch.ones_like(random_t_idx, dtype=torch.float))
    num_items_stage1 += B

# Avoid division by zero for time steps that were not sampled
time_dependent_counts[time_dependent_counts == 0] = 1
time_dependent_weights = time_dependent_cums / time_dependent_counts
# Normalize weights
time_dependent_weights = time_dependent_weights / time_dependent_weights.mean()
# Use sqrt as in the original script's ScoreNet init and loss calculation
time_loss_weights_sqrt = torch.sqrt(time_dependent_weights).detach() # Shape [n_time_steps]

print("Finished calculating time weights.")
plt.figure()
plt.plot(time_loss_weights_sqrt.cpu().numpy())
plt.title("Sqrt Time-Dependent Loss Weights $\sqrt{\lambda(t)}$")
plt.xlabel("Time Step Index")
plt.ylabel("Weight")
plt.savefig(os.path.join(config.output_dir, "timedependent_weight_sqrt.png"))
plt.close()

# --- Stage 2: Training the Score Model ---
print("--- Stage 2: Training DiT Score Model ---")
# Instantiate your actual DiT model here
score_model = DiT(config.ncat)
# Apply DataParallel if using multiple GPUs
# score_model = nn.DataParallel(score_model)
score_model = score_model.to(device)

optimizer = Adam(score_model.parameters(), lr=config.lr)
timepoints = timepoints.to(device)
tqdm_epoch = tqdm.trange(config.num_epochs, desc="Epochs")
for epoch in tqdm_epoch:
    score_model.train()
    avg_loss = 0.
    num_items = 0
    epoch_start_time = time.time()

    for batch in tqdm.tqdm(dataloader, desc="Training Batch", leave=False):
        tokens = batch["tokens"].to(device) # Shape: [B, L]
        segment_sizes = batch["segment_sizes"].to(device) # Shape: [B, N]
        # labels = batch["labels"].to(device) # Labels might not be needed for DDSM training itself
        B, L = tokens.shape

        x_one_hot = F.one_hot(tokens_safe, num_classes=config.ncat).float() # Shape: [B, L, V]
        x_one_hot.masked_fill_(padding_mask.unsqueeze(-1), 0.0)
        x_one_hot = x_one_hot[..., :config.ncat]
        # Sample random time indices (potentially importance sampled)
        random_t_idx = torch.randint(0, config.n_time_steps, (B,), device=device) # Discrete indices

        # Apply forward diffusion
        perturbed_x, perturbed_x_grad = diffusion_factory(
            x_one_hot.cpu(), random_t_idx.cpu(), v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta
        )
        perturbed_x = perturbed_x.to(device) # Shape [B, L, V] (distribution)
        perturbed_x_grad = perturbed_x_grad.to(device) # Shape [B, L, V] (true score)

        # Get continuous time points for the model
        random_timepoints = timepoints[random_t_idx].to(device) # Shape [B]

        # torch.Size([2, 7998, 7]) torch.Size([2, 574]) torch.Size([2])
        # --- Model Forward Pass ---
        # Pass the noisy distribution, segment sizes, and continuous time
        predicted_score = score_model(perturbed_x, segment_sizes, random_timepoints) # Output: [B, L, V]

        # --- Calculate DDSM Loss ---
        if config.random_order:
                raise NotImplementedError("Random order logic needs careful implementation matching gx_to_gv")
        else:
            # Transform gradients to stick-breaking space
            perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach() # [B, L, V-1]
            pred_grad_v = gx_to_gv(predicted_score, perturbed_x, create_graph=True) # [B, L, V-1]
            true_grad_v = gx_to_gv(perturbed_x_grad, perturbed_x) # [B, L, V-1]

        # Get loss weights for the sampled times
        current_loss_weights = (1.0 / time_loss_weights_sqrt)[random_t_idx] # Shape [B]
        # Expand weights for broadcasting: [B, 1, 1]
        current_loss_weights = current_loss_weights.view(B, 1, 1)

        # Speed balancing factor
        if config.speed_balanced:
            s_weights = 2 / (torch.ones(config.ncat - 1, device=device) + torch.arange(config.ncat - 1, 0, -1, device=device).float())
        else:
            s_weights = torch.ones(config.ncat - 1, device=device)

        # Calculate weighted squared error in v-space
        loss_term = current_loss_weights * s_weights * perturbed_v * (1 - perturbed_v) * (pred_grad_v - true_grad_v)**2

        # --- Apply Masking to Loss ---
        # Create mask based on actual sequence lengths (False where valid, True where padded)
        seq_lengths = segment_sizes.sum(dim=1).long()
        mask = torch.arange(L, device=device)[None, :] >= seq_lengths[:, None] # Shape [B, L]
        # Expand mask for the V-1 dimension: [B, L, V-1]
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, config.ncat - 1)
        # Zero out loss terms for padded positions
        loss_term_masked = loss_term.masked_fill(mask_expanded, 0.0)

        # Calculate mean loss only over non-padded elements
        # Sum over V-1 and L, then divide by actual number of non-padded elements in batch
        total_loss_batch = loss_term_masked.sum()
        num_valid_elements = (~mask_expanded).sum()
        loss = total_loss_batch / num_valid_elements if num_valid_elements > 0 else torch.tensor(0.0, device=device)

        # --- Optimization Step ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item() * B # Use weighted average if batch sizes vary?
        num_items += B

    epoch_end_time = time.time()
    avg_loss /= num_items
    tqdm_epoch.set_description(f'Epoch {epoch+1}/{config.num_epochs} | Avg Loss: {avg_loss:.5f} | Time: {epoch_end_time - epoch_start_time:.2f}s')

    # --- Periodic Validation/Saving (Optional) ---
    if (epoch + 1) % 10 == 0:
        score_model.eval()
        print(f"Epoch {epoch+1}: Validation step placeholder.")
        # Add validation logic here (sampling, evaluation)
        save_path = os.path.join(config.output_dir, f"dit_ddsm_epoch_{epoch+1}.pth")
        state_dict_to_save = score_model.state_dict()
        torch.save(state_dict_to_save, save_path)
        print(f"Saved checkpoint to {save_path}")
        score_model.train()

print("Training finished.")
final_save_path = os.path.join(config.output_dir, "dit_ddsm_final.pth")
state_dict_to_save = score_model.state_dict()
torch.save(state_dict_to_save, final_save_path)
print(f"Saved final model to {final_save_path}")