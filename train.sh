#!/bin/bash 
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:1
#SBATCH -p gpu_quad
#SBATCH -t 2-12:00
#SBATCH --mem=64GB
#SBATCH --output=/n/groups/marks/users/erik/prose-ddsm/slurm/ddsm%j.out
#SBATCH --error=/n/groups/marks/users/erik/prose-ddsm/slurm/ddsm%j.err
#SBATCH --job-name="ddsm"

source activate base
conda activate ddsm
export LD_LIBRARY_PATH=/home/jix836/.conda/envs/promoet/lib:${LD_LIBRARY_PATH}
wandb login 6d15eff7893c02d9755493383bac182122241aae
cd /n/groups/marks/users/erik/prose-ddsm
python lightning_flow.py
