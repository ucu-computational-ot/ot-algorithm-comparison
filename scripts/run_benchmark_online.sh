#!/usr/bin/env bash
#SBATCH --job-name=ot-benchmark
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <generator-config> <runner-config> [folds]" >&2
  exit 1
fi

GEN_CFG="$1"
RUN_CFG="$2"
FOLDS="${3:-1}"
RESULT_DIR="${RESULT_DIR:-results}"

mkdir -p "$RESULT_DIR" logs

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

echo "Running benchmark from $RUN_CFG"
python -m uot.experiments.synthetic.benchmark \
  --config "$RUN_CFG" \
  --generators-config "$GEN_CFG" \
  --folds "$FOLDS" \
  --export "$RESULT_DIR/$(basename "$RUN_CFG" .yaml)_$(basename "$GEN_CFG" .yaml)_$(date +'%d-%H-%M').csv"

if type conda >/dev/null 2>&1; then
  conda deactivate
fi
