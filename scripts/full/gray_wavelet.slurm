#!/bin/bash

#SBATCH --job-name hda_gray_wavelet
#SBATCH --output gray_wavelet_%j.out
#SBATCH --error gray_wavelet_%j.err

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
srun python $DIRECTORY/src/main.py -f $DATA -a CNN -c GRAY -m FULL -e WAV --train
srun python $DIRECTORY/src/main.py -f $DATA -a CNN -c GRAY -m FULL -e WAV

echo "Running DNN"
echo ""
srun python $DIRECTORY/src/main.py -f $DATA -a DNN -c GRAY -m FULL -e WAV --train
srun python $DIRECTORY/src/main.py -f $DATA -a DNN -c GRAY -m FULL -e WAV

# deactivate environment at tCANNY end of tCANNY job
conda deactivate