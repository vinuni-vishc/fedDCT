#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=36:00:00
#$ -o /home/aaa10078nj/Federated_Learning/Quan_fedDCT/logs/$JOB_NAME_$JOB_ID.log
#$ -j y

source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load openmpi/4.1.3
module load cuda/11.5/11.5.2
module load cudnn/8.3/8.3.3
module load nccl/2.11/2.11.4-1
module load python/3.10/3.10.4
source ~/venv/pytorch1.11+horovod/bin/activate

LOG_DIR="/home/aaa10078nj/Federated_Learning/Quan_fedDCT/logs/$JOB_NAME_$JOB_ID"
rm -r ${LOG_DIR}
mkdir ${LOG_DIR}

# #Dataset
DATA_DIR="$SGE_LOCALDIR/$JOB_ID/"
cp -r ../AAAI2023/easyFL/benchmark/cifar100/data ${DATA_DIR}

LOG_DIR='./fedDCT/models/splitnet/'
IS_FED=1
FIX_CLUSTER=0
SPLIT_FACTOR=4
NUM_CLUSTERS=20
NUM_SELECTED=20
ARCH="resnet110sl"
DATASET="cifar100"
NUM_CLASSES=100
IS_SINGLE_BRANCH=0
IS_AMP=0
NUM_ROUNDS=650
FED_EPOCHS=1
SPID="feddct_resnet110_split4_cifar100_80clients_80choose_650rounds"

cd fedDCT

# 80 client
python train_feddct.py --is_fed=${IS_FED} --fixed_cluster=${FIX_CLUSTER} --split_factor=${SPLIT_FACTOR} --num_clusters=${NUM_CLUSTERS} --num_selected=${NUM_SELECTED} --arch=${ARCH} --dataset=${DATASET} --num_classes=${NUM_CLASSES} --is_single_branch=${IS_SINGLE_BRANCH} --is_amp=${IS_AMP} --num_rounds=${NUM_ROUNDS} --fed_epochs=${FED_EPOCHS} --spid=${SPID} --data=${DATA_DIR} --model_dir=${LOG_DIR}

