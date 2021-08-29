#!/bin/bash

#SBATCH --job-name hda_rgb_thresh
#SBATCH --output rgb_thresh_%j.out
#SBATCH --error rgb_thresh_%j.err

#SBATCH --mail-type ALL
#SBATCH --mail-user mikele.milia@studenti.unipd.it

#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p allgroups
#SBATCH -t 3-12:00:00
#SBATCH --mem 75G

#SBATCH --gres=gpu:1

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
srun python $DIRECTORY/src/main.py -f $DATA -a CNN -c RGB -m FULL -e BLOB

echo "Running DNN"
echo ""
srun python $DIRECTORY/src/main.py -f $DATA -a DNN -c RGB -m FULL -e BLOB

echo "Running D-CNN"
echo ""
srun python $DIRECTORY/src/main.py -f $DATA -a D-CNN -c RGB -m FULL -e BLOB

# deactivate environment at the end of the job
conda deactivate