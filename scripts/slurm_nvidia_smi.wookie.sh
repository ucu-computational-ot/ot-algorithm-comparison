#!/bin/bash
#SBATCH --job-name=nvidia_smi
#SBATCH --output=logs/nvidia-smi-%j.out
#SBATCH --error=logs/nvidia-smi-%j.err

# Partition and time
#SBATCH --partition=gpu_nodes
#SBATCH --time=05:00

# CPU and memory
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100M

# GPU request
# here we request for 2 type v100 gpu
#SBATCH --gpus-per-node=v100:2

# Email notifications (optional)
# jSBATCH --mail-type=BEGIN,END,FAIL
# jSBATCH --mail-user=youremail@domain.com

echo "Host: $(hostname)"
echo "Starting at: $(date)"

# Load modules or activate conda environment
module load cuda/12.0
module load python/3.10
source ~/ot-algorithm-comparison/venv/bin/activate

# Print GPU details
nvidia-smi

echo "Finished at: $(date)"
