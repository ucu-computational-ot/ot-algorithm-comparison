#!/bin/bash
#SBATCH --job-name=gaussian-ot-methods-benchmark
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=1-10:00:00
#SBATCH --ntasks=1                          # Number of MPI ranks
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1


#### Prepare directories ####
SCRATCH_DIR=/home/izhytkevych/datasets
RESULT_DIR=/home/izhytkevych/results/raw
mkdir -p "${SCRATCH_DIR}"
mkdir -p "${RESULT_DIR}"

export JAX_ENABLE_X64="True"
export JAX_PLATFORM_NAME=gpu

#### Activate conda environment ####
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ot

if [ -d "${SCRATCH_DIR}/synthetic" ] && [ "$(ls -A ${SCRATCH_DIR}/synthetic)" ]; then
  echo "✔ Synthetic data already present; skipping generation."
else
  echo "⏳ Generating synthetic data..."
  python -m uot.problems.problem_serializer \
    --config configs/generators/gaussians.extensive.yaml \
    --export-dir "${SCRATCH_DIR}/synthetic"
fi

echo "⏳ Running benchmark..."
python -m uot.experiments.synthetic.benchmark \
    --config configs/runners/gaussians.extensive.yaml \
    --folds 1 \
    --export "${RESULT_DIR}/gaussians.extensive.csv"

#### Deactivate virtualenv ####
conda deactivate
