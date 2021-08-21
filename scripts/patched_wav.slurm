#!/bin/bash

#SBATCH --job-name patch_wav
#SBATCH --output output_patch_wav_%j.txt
#SBATCH --error errors_patch_wav_%j.txt

#SBATCH --mail-type ALL
#SBATCH --mail-user mikele.milia@studenti.unipd.it

#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p allgroups
#SBATCH -t 7-00:00:00
#SBATCH --mem 200G

# setup my personal/shared project folder
DIRECTORY=/nfsd/hda/miliamikel/lymph
DATA=/nfsd/hda/DATASETS/Project_6

# make my folder the current directory
cd $DIRECTORY

# setup python3 environment for machine learning computations 
source /nfsd/opt/anaconda3/anaconda3.sh
conda activate /nfsd/signet/hda_env_gpu

# run code
echo "Running CNN"
echo ""
srun python $DIRECTORY/src/main.py -f $DATA -a CNN -c RGB -m PATCH -e WAV
srun python $DIRECTORY/src/main.py -f $DATA -a CNN -c HSV -m PATCH -e WAV
srun python $DIRECTORY/src/main.py -f $DATA -a CNN -c GRAY -m PATCH -e WAV

echo "Running DNN"
echo ""
srun python $DIRECTORY/src/main.py -f $DATA -a DNN -c RGB -m PATCH -e WAV
srun python $DIRECTORY/src/main.py -f $DATA -a DNN -c HSV -m PATCH -e WAV
srun python $DIRECTORY/src/main.py -f $DATA -a DNN -c GRAY -m PATCH -e WAV

echo "Running D-CNN"
echo ""
srun python $DIRECTORY/src/main.py -f $DATA -a D-CNN -c RGB -m PATCH -e WAV
srun python $DIRECTORY/src/main.py -f $DATA -a D-CNN -c HSV -m PATCH -e WAV
srun python $DIRECTORY/src/main.py -f $DATA -a D-CNN -c GRAY -m PATCH -e WAV

# deactivate environment at the end of the job
conda deactivate