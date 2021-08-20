#!/bin/bash

# Local variable
MODE=("full" "patch")

# Move into build folder
for mode in "${MODE[@]}"; do

  if [ "$mode" == "full" ]; then
    sbatch job_rgb_wav.slurm -m 'FULL'
    sbatch job_hsv_wav.slurm -m 'FULL'
    sbatch job_gray_wav.slurm -m 'FULL'

  elif [ "$mode" == "patch" ]; then
    sbatch job_rgb_wav.slurm -m 'PATCH'
    sbatch job_hsv_wav.slurm -m 'PATCH'
    sbatch job_gray_wav.slurm -m 'PATCH'

  else
    echo "Mode not found"

  fi
done