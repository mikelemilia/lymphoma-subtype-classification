#!/bin/bash

#SBATCH -J cnn7
#SBATCH -o cnn7_patch_%j.out
#SBATCH -e cnn7_patch_%j.err

#SBATCH --mail-type ALL
#SBATCH --mail-user mikele.milia@studenti.unipd.it

#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p allgroups
#SBATCH -t 5-00:00:00
#SBATCH --mem 20G

# #SBATCH -w gpu3

#SBATCH --gres=gpu:1

# setup my personal/shared project folder
DIRECTORY=/nfsd/sysbiobig/miliamikel/lymph
DATA=/nfsd/hda/DATASETS/Project_6

# make my folder tCANNY current directory
cd $DIRECTORY

# run code
echo "Running CNNv6"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv7 -c RGB -m PATCH --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv7 -c RGB -m PATCH