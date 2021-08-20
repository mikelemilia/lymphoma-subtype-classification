#!/bin/bash

# Local variable
MODE=("full" "patch")

# Move into build folder
for mode in "${MODE[@]}"; do

  if [ "$mode" == "full" ]; then
    sbatch job_rgb_can.slurm -m 'FULL'
    sbatch job_hsv_can.slurm -m 'FULL'
    sbatch job_gray_can.slurm -m 'FULL'

  elif [ "$mode" == "patch" ]; then
    sbatch job_rgb_can.slurm -m 'PATCH'
    sbatch job_hsv_can.slurm -m 'PATCH'
    sbatch job_gray_can.slurm -m 'PATCH'

  else
    echo "Mode not found"

  fi
done