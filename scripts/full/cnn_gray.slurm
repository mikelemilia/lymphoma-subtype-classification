#!/bin/bash

#SBATCH -J gray_full
#SBATCH -o gray_full_%j.out
#SBATCH -e gray_full_%j.err

#SBATCH --mail-type ALL
#SBATCH --mail-user mikele.milia@studenti.unipd.it

#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p allgroups
#SBATCH -t 3-00:00:00
#SBATCH --mem 75G

#SBATCH --gres=gpu:1

# setup my personal/shared project folder
DIRECTORY=/nfsd/sysbiobig/miliamikel/lymph
DATA=/nfsd/hda/DATASETS/Project_6

# make my folder tCANNY current directory
cd $DIRECTORY

# run code
echo "--------------------------------------------------------------------"
echo "Running CNN GRAY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c GRAY -m FULL --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c GRAY -m FULL

echo "--------------------------------------------------------------------"
echo "Running CNN GRAY WAV"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c GRAY -m FULL -e WAV --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c GRAY -m FULL -e WAV

echo "--------------------------------------------------------------------"
echo "Running CNN GRAY CANNY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c GRAY -m FULL -e CANNY --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c GRAY -m FULL -e CANNY

########################################################################################################################
echo "--------------------------------------------------------------------"
echo "Running CNNv1 GRAY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c GRAY -m FULL --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c GRAY -m FULL

echo "--------------------------------------------------------------------"
echo "Running CNNv1 GRAY WAV"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c GRAY -m FULL -e WAV --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c GRAY -m FULL -e WAV

echo "--------------------------------------------------------------------"
echo "Running CNNv1 GRAY CANNY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c GRAY -m FULL -e CANNY --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c GRAY -m FULL -e CANNY

########################################################################################################################
echo "--------------------------------------------------------------------"
echo "Running CNNv2 GRAY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c GRAY -m FULL --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c GRAY -m FULL

echo "--------------------------------------------------------------------"
echo "Running CNNv2 GRAY WAV"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c GRAY -m FULL -e WAV --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c GRAY -m FULL -e WAV

echo "--------------------------------------------------------------------"
echo "Running CNNv2 GRAY CANNY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c GRAY -m FULL -e CANNY --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c GRAY -m FULL -e CANNY

########################################################################################################################
echo "--------------------------------------------------------------------"
echo "Running CNNv3 GRAY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c GRAY -m FULL --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c GRAY -m FULL

echo "--------------------------------------------------------------------"
echo "Running CNNv3 GRAY WAV"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c GRAY -m FULL -e WAV --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c GRAY -m FULL -e WAV

echo "--------------------------------------------------------------------"
echo "Running CNNv3 GRAY CANNY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c GRAY -m FULL -e CANNY --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c GRAY -m FULL -e CANNY

########################################################################################################################
echo "--------------------------------------------------------------------"
echo "Running CNNv4 GRAY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c GRAY -m FULL --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c GRAY -m FULL

echo "--------------------------------------------------------------------"
echo "Running CNNv4 GRAY WAV"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c GRAY -m FULL -e WAV --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c GRAY -m FULL -e WAV

echo "--------------------------------------------------------------------"
echo "Running CNNv4 GRAY CANNY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c GRAY -m FULL -e CANNY --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c GRAY -m FULL -e CANNY