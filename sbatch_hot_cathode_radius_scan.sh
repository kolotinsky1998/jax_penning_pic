#!/bin/bash
#SBATCH --job-name=penning_hot_radius_scan
#SBATCH --error=err_penning_hot_radius_scan_%A_%a
#SBATCH --output=output_penning_hot_radius_scan_%A_%a
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=rocky
#SBATCH --account=proj_1827
#SBATCH --array=0-5

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Low-index tasks are the lightest. No array concurrency limit is used, so
# Slurm may run all six tasks simultaneously when resources are available.
GRID_SIZES=(70 70 140 140 280 280)
HOT_CATHODE_RADII=(0.04 0.08 0.04 0.08 0.04 0.08)
MAX_PARTICLES=(625000 625000 2500000 2500000 10000000 10000000)
MACRO_WEIGHTS=(110482368 110482368 27620592 27620592 6905148 6905148)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
GRID_SIZE="${GRID_SIZES[$TASK_ID]}"
HOT_CATHODE_RADIUS="${HOT_CATHODE_RADII[$TASK_ID]}"
MAX_PARTICLES_PER_SPECIES="${MAX_PARTICLES[$TASK_ID]}"
MACRO_PARTICLE_WEIGHT="${MACRO_WEIGHTS[$TASK_ID]}"

SOLVER=cg
SOLVER_IT=1000
IT_NUM=2000000
LOG_INTERVAL=100
FIELD_DUMP_INTERVAL=10000
PARTICLE_SAMPLE_STRIDE=1000

OUTPUT_DIR="outputs/hot_cathode_radius_scan/${SOLVER}_${GRID_SIZE}x${GRID_SIZE}_r${HOT_CATHODE_RADIUS}_w${MACRO_PARTICLE_WEIGHT}_job${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "grid=${GRID_SIZE}x${GRID_SIZE} hot_cathode_radius=${HOT_CATHODE_RADIUS} solver=${SOLVER}"
echo "max_electrons=${MAX_PARTICLES_PER_SPECIES} max_ions=${MAX_PARTICLES_PER_SPECIES} macro_particle_weight=${MACRO_PARTICLE_WEIGHT}"
echo "it_num=${IT_NUM} field_dump_interval=${FIELD_DUMP_INTERVAL} particle_sample_stride=${PARTICLE_SAMPLE_STRIDE}"
echo "output_dir=${OUTPUT_DIR}"

time srun --account=proj_1827 --gpus=1 env PYTHONPATH=. python3 -u \
  scripts/run_simulation_circle_gyro_new.py \
  --poisson-solver "${SOLVER}" \
  --poisson-iterations "${SOLVER_IT}" \
  --grid-size "${GRID_SIZE}" \
  --hot-cathode-radius "${HOT_CATHODE_RADIUS}" \
  --macro-particle-weight "${MACRO_PARTICLE_WEIGHT}" \
  --max-electrons "${MAX_PARTICLES_PER_SPECIES}" \
  --max-ions "${MAX_PARTICLES_PER_SPECIES}" \
  --it-num "${IT_NUM}" \
  --log-interval "${LOG_INTERVAL}" \
  --field-dump-interval "${FIELD_DUMP_INTERVAL}" \
  --particle-sample-stride "${PARTICLE_SAMPLE_STRIDE}" \
  --output-dir "${OUTPUT_DIR}"
