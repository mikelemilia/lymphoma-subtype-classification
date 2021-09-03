#!/bin/bash

#SBATCH --job-name hda_patch_hsv_wavelet
#SBATCH --output patch_hsv_wavelet_%j.out
#SBATCH --error patch_hsv_wavelet_%j.err

#SBATCH --mail-type ALL
#SBATCH --mail-user mikele.milia@studenti.unipd.it

#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p allgroups
#SBATCH -t 5-00:00:00
#SBATCH --mem 75G

#SBATCH --gres=gpu:1

# setup my personal/shared project folder
DIRECTORY=/nfsd/hda/miliamikel/lymph
DATA=/nfsd/hda/DATASETS/Project_6

# make my folder tCANNY current directory
cd $DIRECTORY

# setup python3 environment for machine learning computations 
source /nfsd/opt/anaconda3/anaconda3.sh
conda activate /nfsd/signet/hda_env_gpu

# run code
echo "Running CNN"
echo ""
srun python $DIRECTORY/src/main.py -f $DATA -a CNN -c HSV -m PATCH -e WAV --train
srun python $DIRECTORY/src/main.py -f $DATA -a CNN -c HSV -m PATCH -e WAV

echo "Running DNN"
echo ""
srun python $DIRECTORY/src/main.py -f $DATA -a DNN -c HSV -m PATCH -e WAV --train
srun python $DIRECTORY/src/main.py -f $DATA -a DNN -c HSV -m PATCH -e WAV

echo "Running D-CNN"
echo ""
srun python $DIRECTORY/src/main.py -f $DATA -a D-CNN -c HSV -m PATCH -e WAV --train
srun python $DIRECTORY/src/main.py -f $DATA -a D-CNN -c HSV -m PATCH -e WAV

# deactivate environment at tCANNY end of tCANNY job
conda deactivate