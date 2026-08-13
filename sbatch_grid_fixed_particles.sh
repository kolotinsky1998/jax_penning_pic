#!/bin/bash
#SBATCH --job-name=penning_grid_fixed
#SBATCH --error=err_penning_grid_fixed_%A_%a
#SBATCH --output=output_penning_grid_fixed_%A_%a
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=rocky
#SBATCH --account=proj_1827
#SBATCH --array=0-7

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

GRID_SIZES=(35 70 140 280 35 70 140 280)
SOLVERS=(direct_inverse direct_inverse direct_inverse direct_inverse jacobi jacobi jacobi jacobi)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
GRID_SIZE="${GRID_SIZES[$TASK_ID]}"
SOLVER="${SOLVERS[$TASK_ID]}"
OUTPUT_DIR="outputs/grid_fixed_particles/${SOLVER}_${GRID_SIZE}x${GRID_SIZE}"

echo "grid=${GRID_SIZE}x${GRID_SIZE} solver=${SOLVER} it_num=2000000"

time srun -A proj_1827 --gpus=1 env PYTHONPATH=. python3 -u \
  scripts/run_simulation_circle_gyro_new.py \
  --poisson-solver "${SOLVER}" \
  --grid-size "${GRID_SIZE}" \
  --ptcls-per-cell 1 \
  --max-electrons 300000 \
  --max-ions 300000 \
  --it-num 2000000 \
  --output-dir "${OUTPUT_DIR}"
