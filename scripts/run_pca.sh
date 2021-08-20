#!/bin/bash

# Local variable
MODE=("full" "patch")

# Move into build folder
for mode in "${MODE[@]}"; do

  if [ "$mode" == "full" ]; then
    sbatch job_rgb_pca.slurm -m 'FULL'
    sbatch job_hsv_pca.slurm -m 'FULL'
    sbatch job_gray_pca.slurm -m 'FULL'

  elif [ "$mode" == "patch" ]; then
    sbatch job_rgb_pca.slurm -m 'PATCH'
    sbatch job_hsv_pca.slurm -m 'PATCH'
    sbatch job_gray_pca.slurm -m 'PATCH'

  else
    echo "Mode not found"

  fi
done