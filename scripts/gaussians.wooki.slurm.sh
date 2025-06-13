#!/bin/bash
#SBATCH --job-name=gaussian-ot-methods-benchmark
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1                          # Number of MPI ranks
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu                 # Partition name (adjust as needed)

#### 1. Load modules (adjust Python version & modules as on Titan) ####
module load python/3.8

#### 2. Prepare directories ####
SCRATCH_DIR=/share_scratch/izhytkevych
RESULT_DIR=/home/izhytkevych/results/raw
mkdir -p "${SCRATCH_DIR}"
mkdir -p "${RESULT_DIR}"

#### 3. Activate your virtualenv ####
# Assumes you created it in your home directory at ~/venv
# source /home/izhytkevych/ot-algorithm-comparison/venv/bin/activate
conda activate ot_comparison

#### 4. Change to working directory on scratch ####
cd "${SCRATCH_DIR}"

#### 5. Run benchmarks ####
python -m uot.problems.problem_serializer \
    --config configs/generators/gaussians.extensive.yaml \
    --export-dir "${SCRATCH_DIR}/synthetic"


python -m uot.experiments.synthetic.benchmark \
    --config configs/runners/gaussians.yaml \
    --folds 1 \
    --export "${RESULT_DIR}/gaussians.csv"

#### 6. Deactivate virtualenv (optional) ####
conda deactivate
