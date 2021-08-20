#!/bin/bash

#SBATCH --job-name hda_gray
#SBATCH --output output_%j.txt
#SBATCH --error errors_%j.txt

#SBATCH --mail-type ALL
#SBATCH --mail-user mikele.milia@studenti.unipd.it

#SBATCH --gres=gpu:1

#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p allgroups
#SBATCH -t 2-00:00:00
#SBATCH --mem 50G

# Retrieve custom project path
while getopts m:h option; do
  case "${option}" in
  m) MODE=${OPTARG} ;;
  h) job_help
     exit ;;
  *) echo "Command not supported"
     exit ;;
  esac
done

# setup my personal/shared project folder
DIRECTORY=/nfsd/hda/miliamikel/lymph
DATA=/nfsd/hda/DATASETS/Project_6

# make my folder the current directory
cd $DIRECTORY

# setup python3 environment for machine learning computations 
source /nfsd/opt/anaconda3/anaconda3.sh
conda activate /nfsd/signet/hda_env_gpu

# run code
srun python $DIRECTORY/src/main.py -f $DATA -a CNN -c GRAY
srun python $DIRECTORY/src/main.py -f $DATA -a DNN -c GRAY
srun python $DIRECTORY/src/main.py -f $DATA -a DCNN -c GRAY

# deactivate environment at the end of the job
conda deactivate