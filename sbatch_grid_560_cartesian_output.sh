#!/bin/bash
#SBATCH --job-name=penning_grid_560
#SBATCH --error=err_penning_grid_560_%j
#SBATCH --output=output_penning_grid_560_%j
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=rocky
#SBATCH --account=proj_1827

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

GRID_SIZE=560
SOLVER=jacobi
IT_NUM=2000000

# Particle pools start empty in this branch. These values are pool capacities,
# not initial particle counts. 19.2M continues the old 4x capacity scaling
# from 280x280 / 4.8M particles per species to 560x560.
MAX_ELECTRONS=19200000
MAX_IONS=19200000

# Number of physical particles represented by one simulated macroparticle.
MACRO_PARTICLE_WEIGHT=10000000

LOG_INTERVAL=100
FIELD_DUMP_INTERVAL=10000
# 0 disables particle VTP output; 1 writes every live particle.
PARTICLE_SAMPLE_STRIDE=1000

# Including the job ID prevents accidental reuse of an existing result folder.
OUTPUT_DIR="outputs/grid_560_cartesian/${SOLVER}_${GRID_SIZE}x${GRID_SIZE}_w${MACRO_PARTICLE_WEIGHT}_job${SLURM_JOB_ID}"

echo "grid=${GRID_SIZE}x${GRID_SIZE} solver=${SOLVER}"
echo "max_electrons=${MAX_ELECTRONS} max_ions=${MAX_IONS} macro_particle_weight=${MACRO_PARTICLE_WEIGHT}"
echo "it_num=${IT_NUM} field_dump_interval=${FIELD_DUMP_INTERVAL} particle_sample_stride=${PARTICLE_SAMPLE_STRIDE}"
echo "output_dir=${OUTPUT_DIR}"

time srun --account=proj_1827 --gpus=1 env PYTHONPATH=. python3 -u \
  scripts/run_simulation_circle_gyro_new.py \
  --poisson-solver "${SOLVER}" \
  --grid-size "${GRID_SIZE}" \
  --macro-particle-weight "${MACRO_PARTICLE_WEIGHT}" \
  --max-electrons "${MAX_ELECTRONS}" \
  --max-ions "${MAX_IONS}" \
  --it-num "${IT_NUM}" \
  --log-interval "${LOG_INTERVAL}" \
  --field-dump-interval "${FIELD_DUMP_INTERVAL}" \
  --particle-sample-stride "${PARTICLE_SAMPLE_STRIDE}" \
  --output-dir "${OUTPUT_DIR}"
