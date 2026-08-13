#!/bin/bash
#SBATCH --job-name=penning_grid_scaled
#SBATCH --error=err_penning_grid_scaled_%A_%a
#SBATCH --output=output_penning_grid_scaled_%A_%a
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
PTCLS_PER_CELL=(0.25 1 4 16 0.25 1 4 16)
MAX_PER_SPECIES=(75000 300000 1200000 4600000 75000 300000 1200000 4600000)
SOLVERS=(direct_inverse direct_inverse direct_inverse direct_inverse jacobi jacobi jacobi jacobi)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
GRID_SIZE="${GRID_SIZES[$TASK_ID]}"
PPC="${PTCLS_PER_CELL[$TASK_ID]}"
MAX_PARTICLES="${MAX_PER_SPECIES[$TASK_ID]}"
SOLVER="${SOLVERS[$TASK_ID]}"
OUTPUT_DIR="outputs/grid_scaled_particles/${SOLVER}_${GRID_SIZE}x${GRID_SIZE}_ppc${PPC}"

echo "grid=${GRID_SIZE}x${GRID_SIZE} solver=${SOLVER} ppc=${PPC} max_per_species=${MAX_PARTICLES} it_num=2000000"

time srun -A proj_1827 --gpus=1 env PYTHONPATH=. python3 -u \
  scripts/run_simulation_circle_gyro_new.py \
  --poisson-solver "${SOLVER}" \
  --grid-size "${GRID_SIZE}" \
  --ptcls-per-cell "${PPC}" \
  --max-electrons "${MAX_PARTICLES}" \
  --max-ions "${MAX_PARTICLES}" \
  --it-num 2000000 \
  --output-dir "${OUTPUT_DIR}"
