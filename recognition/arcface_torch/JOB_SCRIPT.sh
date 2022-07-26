#!/bin/bash
#$ -q gpu
#$ -l gpu_card=1
#$ -N arcface


############################################

cd /afs/crc.nd.edu/user/p/ptinsley/insightface/recognition/arcface_torch/

module load cuda/11.6
module load cudnn/8.0.4 

# module load cudnn
# nvcc --version

export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/bin/activate arcface

############################################

# python FOR_NHAT.py --inputDir=./sample_inputDir --outFile=./sample_outFile.pkl
# python doMTCNN.py
# python FOR_ME.py --inputDir=/scratch365/ptinsley/results-truncation-sg3-chips/ --i=$SGE_TASK_ID --outFile=./results-truncation-sg3-subset$SGE_TASK_ID.csv

python FOR_DEEKSHA.py --inputDir=/scratch365/ptinsley/results-truncation-sg3-chips/ --outFile=./results.csv
