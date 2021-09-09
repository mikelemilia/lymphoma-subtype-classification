#!/bin/bash

#SBATCH -J rgb_full
#SBATCH -o rgb_full_%j.out
#SBATCH -e rgb_full_%j.err

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
echo "Running CNN RGB"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c RGB -m FULL --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c RGB -m FULL

echo "--------------------------------------------------------------------"
echo "Running CNN RGB WAV"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c RGB -m FULL -e WAV --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c RGB -m FULL -e WAV

echo "--------------------------------------------------------------------"
echo "Running CNN RGB CANNY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c RGB -m FULL -e CANNY --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c RGB -m FULL -e CANNY

########################################################################################################################
echo "--------------------------------------------------------------------"
echo "Running CNNv1 RGB"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c RGB -m FULL --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c RGB -m FULL

echo "--------------------------------------------------------------------"
echo "Running CNNv1 RGB WAV"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c RGB -m FULL -e WAV --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c RGB -m FULL -e WAV

echo "--------------------------------------------------------------------"
echo "Running CNNv1 RGB CANNY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c RGB -m FULL -e CANNY --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c RGB -m FULL -e CANNY

########################################################################################################################
echo "--------------------------------------------------------------------"
echo "Running CNNv2 RGB"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c RGB -m FULL --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c RGB -m FULL

echo "--------------------------------------------------------------------"
echo "Running CNNv2 RGB WAV"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c RGB -m FULL -e WAV --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c RGB -m FULL -e WAV

echo "--------------------------------------------------------------------"
echo "Running CNNv2 RGB CANNY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c RGB -m FULL -e CANNY --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c RGB -m FULL -e CANNY

########################################################################################################################
echo "--------------------------------------------------------------------"
echo "Running CNNv3 RGB"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c RGB -m FULL --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c RGB -m FULL

echo "--------------------------------------------------------------------"
echo "Running CNNv3 RGB WAV"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c RGB -m FULL -e WAV --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c RGB -m FULL -e WAV

echo "--------------------------------------------------------------------"
echo "Running CNNv3 RGB CANNY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c RGB -m FULL -e CANNY --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c RGB -m FULL -e CANNY

########################################################################################################################
echo "--------------------------------------------------------------------"
echo "Running CNNv4 RGB"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c RGB -m FULL --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c RGB -m FULL

echo "--------------------------------------------------------------------"
echo "Running CNNv4 RGB WAV"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c RGB -m FULL -e WAV --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c RGB -m FULL -e WAV

echo "--------------------------------------------------------------------"
echo "Running CNNv4 RGB CANNY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c RGB -m FULL -e CANNY --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c RGB -m FULL -e CANNY