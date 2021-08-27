#!/bin/bash

PATH="/nfsd/hda/miliamikel/lymph/scripts"
COLOR=("rgb" "hsv" "gray")

for color in "${COLOR[@]}"; do

  echo "Submitted full image"
  sbatch "$PATH/$color".slurm

  echo "Submitted full image + CANNY"
  echo ""
  sbatch "$PATH/$color"_canny.slurm

  echo "Submitted full image + PCA"
  echo ""
  sbatch "$PATH/$color"_pca.slurm

  echo "Submitted full image + THRESH"
  echo ""
  sbatch "$PATH/$color"_thresh.slurm

  echo "Submitted full image + WAVELET"
  echo ""
  sbatch "$PATH/$color"_wavelet.slurm

done
