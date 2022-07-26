#!/bin/bash
#$ -q gpu@@cvrl_gpu
#$ -l gpu_card=1
#$ -N retface
#$ -t 1-10

####################### Set Up ###############################

cd /afs/crc.nd.edu/user/p/ptinsley/insightface/detection/retinaface/

module load cuda/10.2
nvcc --version

export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/bin/activate retinaface
conda env list

#module load gcc/
#export CC=/opt/crc/g/gcc/8.3.0/bin/gcc
#export CXX=/opt/crc/g/gcc/8.3.0/bin/g++

######################################################################

python test.py --i=$SGE_TASK_ID
# python test.py
