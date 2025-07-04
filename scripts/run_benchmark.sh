#!/usr/bin/env bash
#SBATCH --job-name=ot-benchmark
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <generator-config> <runner-config> [folds]" >&2
  exit 1
fi

GEN_CFG="$1"
RUN_CFG="$2"
FOLDS="${3:-1}"
DATA_DIR="${DATA_DIR:-datasets/synthetic}"
RESULT_DIR="${RESULT_DIR:-results}"

mkdir -p "$DATA_DIR" "$RESULT_DIR" logs

# activate conda environment if available
if [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate ot
fi

# JAX settings suitable for most runs
export JAX_ENABLE_X64="True"
export JAX_PLATFORM_NAME="gpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"

if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
  echo "Generating synthetic data into $DATA_DIR"
  python -m uot.problems.problem_serializer \
    --config "$GEN_CFG" \
    --export-dir "$DATA_DIR"
fi

echo "Running benchmark from $RUN_CFG"
python -m uot.experiments.synthetic.benchmark \
  --config "$RUN_CFG" \
  --dataset "$DATA_DIR" \
  --folds "$FOLDS" \
  --export "$RESULT_DIR/$(basename "$RUN_CFG" .yaml).csv"

if type conda >/dev/null 2>&1; then
  conda deactivate
fi
