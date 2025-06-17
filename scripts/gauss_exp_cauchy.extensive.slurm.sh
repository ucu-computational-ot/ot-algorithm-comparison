#!/bin/bash
#SBATCH --job-name=ot-methods-benchmark
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1                          # Number of MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --gres=gpu:1

# Email notifications (optional)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=izhytkev@uottawa.ca


#### Prepare directories ####
SCRATCH_DIR=/home/izhytkevych/datasets
RESULT_DIR=/home/izhytkevych/results/raw
mkdir -p "${SCRATCH_DIR}"
mkdir -p "${RESULT_DIR}"

#### Activate conda environment ####
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ot

if [ -d "${SCRATCH_DIR}/synthetic_gauss_exp_cauchy" ] && [ "$(ls -A ${SCRATCH_DIR}/synthetic_gauss_exp_cauchy)" ]; then
  echo "✔ Synthetic data already present; skipping generation."
else
  echo "⏳ Generating synthetic data..."
  python -m uot.problems.problem_serializer \
    --config configs/generators/gaussians.extensive.yaml \
    --export-dir "${SCRATCH_DIR}/synthetic_gauss_exp_cauchy"
  python -m uot.problems.problem_serializer \
    --config configs/generators/exponential.extensive.yaml \
    --export-dir "${SCRATCH_DIR}/synthetic_gauss_exp_cauchy"
  python -m uot.problems.problem_serializer \
    --config configs/generators/cauchy.extensive.yaml \
    --export-dir "${SCRATCH_DIR}/synthetic_gauss_exp_cauchy"
fi

echo "⏳ Running benchmark..."
python -m uot.experiments.synthetic.benchmark \
    --config configs/runners/cauchy_exp_gauss.extensive.yaml \
    --folds 1 \
    --export "${RESULT_DIR}/gauss_exp_cauchy.extensive.csv"

#### Deactivate virtualenv ####
conda deactivate
