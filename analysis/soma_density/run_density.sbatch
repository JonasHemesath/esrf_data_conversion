#!/bin/bash
#SBATCH --job-name=soma_density
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --array=0-31
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate raw_to

INPUT="/cajal/scratch/projects/xray/bm05/ng/instances/260306_Soma_distance_transform_multires_multipath_linearLR_soma_masked"
OUTBASE="cajal/scratch/projects/xray/bm05/ng/instances/260306_Soma_distance_transform_multires_multipath_linearLR_soma_masked/260306_Soma_distance_transform_multires_multipath_linearLR_soma_masked_density"

python compute_soma_density.py \
  --input_path "$INPUT" \
  --output_base "$OUTBASE" \
  --mip_hi 0 \
  --mip_lo 5 \
  --block 200 \
  --processes 64 \
  --z_slab 1 \
  --job_id "${SLURM_ARRAY_TASK_ID}" \
  --num_jobs "${SLURM_ARRAY_TASK_COUNT}"