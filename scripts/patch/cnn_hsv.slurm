#!/bin/bash

#SBATCH -J hsv_patch
#SBATCH -o hsv_patch_%j.out
#SBATCH -e hsv_patch_%j.err

#SBATCH --mail-type ALL
#SBATCH --mail-user mikele.milia@studenti.unipd.it

#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p allgroups
#SBATCH -t 3-00:00:00
#SBATCH --mem 20G

#SBATCH --gres=gpu:1

# setup my personal/shared project folder
DIRECTORY=/nfsd/sysbiobig/miliamikel/lymph
DATA=/nfsd/hda/DATASETS/Project_6

# make my folder tCANNY current directory
cd $DIRECTORY

# run code
#echo "--------------------------------------------------------------------"
#echo "Running CNN HSV"
#echo ""
#
## Train
#srun singularity exec --nv \
#                 --bind $DATA --bind $DIRECTORY \
#                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
#                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c HSV -m PATCH --train
#
## Prediction
#singularity exec --nv \
#                 --bind $DATA --bind $DIRECTORY \
#                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
#                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c HSV -m PATCH

echo "--------------------------------------------------------------------"
echo "Running CNN HSV WAV"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c HSV -m PATCH -e WAV --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c HSV -m PATCH -e WAV
#
#echo "--------------------------------------------------------------------"
#echo "Running CNN HSV CANNY"
#echo ""
#
## Train
#srun singularity exec --nv \
#                 --bind $DATA --bind $DIRECTORY \
#                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
#                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c HSV -m PATCH -e CANNY --train
#
## Prediction
#singularity exec --nv \
#                 --bind $DATA --bind $DIRECTORY \
#                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
#                 python $DIRECTORY/src/main.py -f $DATA -a CNN -c HSV -m PATCH -e CANNY

########################################################################################################################
#echo "--------------------------------------------------------------------"
#echo "Running CNNv1 HSV"
#echo ""
#
## Train
#srun singularity exec --nv \
#                 --bind $DATA --bind $DIRECTORY \
#                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
#                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c HSV -m PATCH --train
#
## Prediction
#singularity exec --nv \
#                 --bind $DATA --bind $DIRECTORY \
#                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
#                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c HSV -m PATCH
#
#echo "--------------------------------------------------------------------"
#echo "Running CNNv1 HSV WAV"
#echo ""
#
## Train
#srun singularity exec --nv \
#                 --bind $DATA --bind $DIRECTORY \
#                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
#                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c HSV -m PATCH -e WAV --train
#
## Prediction
#singularity exec --nv \
#                 --bind $DATA --bind $DIRECTORY \
#                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
#                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c HSV -m PATCH -e WAV

echo "--------------------------------------------------------------------"
echo "Running CNNv1 HSV CANNY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c HSV -m PATCH -e CANNY --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv1 -c HSV -m PATCH -e CANNY

########################################################################################################################
#echo "--------------------------------------------------------------------"
#echo "Running CNNv2 HSV"
#echo ""
#
## Train
#srun singularity exec --nv \
#                 --bind $DATA --bind $DIRECTORY \
#                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
#                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c HSV -m PATCH --train
#
## Prediction
#singularity exec --nv \
#                 --bind $DATA --bind $DIRECTORY \
#                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
#                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c HSV -m PATCH

echo "--------------------------------------------------------------------"
echo "Running CNNv2 HSV WAV"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c HSV -m PATCH -e WAV --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c HSV -m PATCH -e WAV

echo "--------------------------------------------------------------------"
echo "Running CNNv2 HSV CANNY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c HSV -m PATCH -e CANNY --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv2 -c HSV -m PATCH -e CANNY

########################################################################################################################
#echo "--------------------------------------------------------------------"
#echo "Running CNNv3 HSV"
#echo ""
#
## Train
#srun singularity exec --nv \
#                 --bind $DATA --bind $DIRECTORY \
#                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
#                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c HSV -m PATCH --train
#
## Prediction
#singularity exec --nv \
#                 --bind $DATA --bind $DIRECTORY \
#                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
#                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c HSV -m PATCH

echo "--------------------------------------------------------------------"
echo "Running CNNv3 HSV WAV"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c HSV -m PATCH -e WAV --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c HSV -m PATCH -e WAV

#echo "--------------------------------------------------------------------"
#echo "Running CNNv3 HSV CANNY"
#echo ""
#
## Train
#srun singularity exec --nv \
#                 --bind $DATA --bind $DIRECTORY \
#                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
#                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c HSV -m PATCH -e CANNY --train
#
## Prediction
#singularity exec --nv \
#                 --bind $DATA --bind $DIRECTORY \
#                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
#                 python $DIRECTORY/src/main.py -f $DATA -a CNNv3 -c HSV -m PATCH -e CANNY

########################################################################################################################
#echo "--------------------------------------------------------------------"
#echo "Running CNNv4 HSV"
#echo ""
#
## Train
#srun singularity exec --nv \
#                 --bind $DATA --bind $DIRECTORY \
#                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
#                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c HSV -m PATCH --train
#
## Prediction
#singularity exec --nv \
#                 --bind $DATA --bind $DIRECTORY \
#                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
#                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c HSV -m PATCH

echo "--------------------------------------------------------------------"
echo "Running CNNv4 HSV WAV"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c HSV -m PATCH -e WAV --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c HSV -m PATCH -e WAV

echo "--------------------------------------------------------------------"
echo "Running CNNv4 HSV CANNY"
echo ""

# Train
srun singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c HSV -m PATCH -e CANNY --train

# Prediction
singularity exec --nv \
                 --bind $DATA --bind $DIRECTORY \
                 /nfsd/opt/sif-images/tensorflow_latest-gpu.sif \
                 python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c HSV -m PATCH -e CANNY