#!/bin/bash

#SBATCH --job-name CNNv2
#SBATCH --output CNNv2_patch_rgb_%j.out
#SBATCH --error CNNv2_patch_rgb_%j.err

#SBATCH --mail-type ALL
#SBATCH --mail-user mikele.milia@studenti.unipd.it

#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p allgroups
#SBATCH -t 5-00:00:00
#SBATCH --mem 20G

#SBATCH --gres=gpu:1

# setup my personal/shared project folder
DIRECTORY=/nfsd/sysbiobig/miliamikel/lymph
DATA=/nfsd/hda/DATASETS/Project_6

# make my folder tCANNY current directory
cd $DIRECTORY

# setup python3 environment for machine learning computations 
source /nfsd/opt/anaconda3/anaconda3.sh
conda activate /nfsd/signet/hda_env_gpu
#conda activate /nfsd/signet/hda_env

# run code
echo "Running CNNv2"
echo ""
srun python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c RGB -m PATCH --train
srun python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c RGB -m PATCH

#echo "Running CNN"
#echo ""
#srun python $DIRECTORY/src/main.py -f $DATA -a CNN -c RGB -m PATCH --train
#srun python $DIRECTORY/src/main.py -f $DATA -a CNN -c RGB -m PATCH
#
#echo "Running DNN"
#echo ""
#srun python $DIRECTORY/src/main.py -f $DATA -a DNN -c RGB -m PATCH --train
#srun python $DIRECTORY/src/main.py -f $DATA -a DNN -c RGB -m PATCH

# deactivate environment at tCANNY end of tCANNY job
conda deactivate