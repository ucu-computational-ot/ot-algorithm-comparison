#!/bin/bash
#SBATCH --job-name=benchmark_ot_methods
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

# Partition and time
#SBATCH --partition=gpu_nodes
#SBATCH --time=24:00:00

# CPU and memory
#SBATCH --ntasks=1                # single process
#SBATCH --cpus-per-task=4         # 4 CPU cores
#SBATCH --mem=32G                 # 32 GB RAM

# GPU request
# here we request for 2 type v100 gpu
#SBATCH --gpus-per-node=v100:2

# Email notifications (optional)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=youremail@domain.com

echo "Host: $(hostname)"
echo "Starting at: $(date)"

# Load modules or activate conda environment
# module load cuda/12.0
# module load python/3.10
# source ~/envs/mygpuenv/bin/activate

# Print GPU details
# nvidia-smi

# Run your GPU‐powered code (e.g., PyTorch / TensorFlow training)

echo "Finished at: $(date)"
