#!/bin/bash

echo "Submitted NN without feature extracted"
sbatch original.slurm
sbatch patched.slurm

#echo ""
#echo "Submitted NN with CANNY"
#sbatch original_can.slurm
#sbatch patched_can.slurm
#
#echo ""
#echo "Submitted NN with PCA"
#sbatch original_pca.slurm
#sbatch patched_pca.slurm
#
#echo ""
#echo "Submitted NN with WAV"
#sbatch original_wav.slurm
#sbatch patched_wav.slurm