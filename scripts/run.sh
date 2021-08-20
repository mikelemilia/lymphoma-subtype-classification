#!/bin/bash

# Local variable
MODE=("full" "patch")

# Move into build folder
for mode in "${MODE[@]}"; do

  if [ "$mode" == "full" ]; then
    sbatch job_rgb.slurm -m "FULL"
    sbatch job_hsv.slurm -m "FULL"
    sbatch job_gray.slurm -m "FULL"

  elif [ "$mode" == "patch" ]; then
    sbatch job_rgb.slurm -m "PATCH"
    sbatch job_hsv.slurm -m "PATCH"
    sbatch job_gray.slurm -m "PATCH"

  else
    echo "Mode not found"

  fi
done