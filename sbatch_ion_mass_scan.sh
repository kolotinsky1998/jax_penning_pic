#!/bin/bash
#SBATCH --job-name=penning_ion_mass
#SBATCH --error=err_penning_ion_mass_%A_%a
#SBATCH --output=output_penning_ion_mass_%A_%a
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

ION_MASS_RATIOS=(100 200 300 500 1000 2000 4000 8000)
IT_NUMS=(400000 800000 1200000 2000000 4000000 8000000 16000000 32000000)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
ION_MASS_RATIO="${ION_MASS_RATIOS[$TASK_ID]}"
IT_NUM="${IT_NUMS[$TASK_ID]}"
OUTPUT_DIR="outputs/ion_mass_scan/mi_${ION_MASS_RATIO}_me"

echo "ion_mass_ratio=${ION_MASS_RATIO} grid=70x70 it_num=${IT_NUM}"

time srun -A proj_1827 --gpus=1 env PYTHONPATH=. python3 -u \
  scripts/run_simulation_circle_gyro_new.py \
  --poisson-solver direct_inverse \
  --grid-size 70 \
  --ion-mass-ratio "${ION_MASS_RATIO}" \
  --ptcls-per-cell 1 \
  --max-electrons 300000 \
  --max-ions 300000 \
  --it-num "${IT_NUM}" \
  --output-dir "${OUTPUT_DIR}"
