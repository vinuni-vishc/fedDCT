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

cd fedDCT
python train_feddct.py --is_fed=1 --fixed_cluster=0 --split_factor=8 --num_clusters=12 --num_selected=12 --arch=wide_resnetsl16_8 --dataset=cifar100 --num_classes=100 --is_single_branch=0 --is_amp=0 --num_rounds=650 --fed_epochs=1 --spid="feddct_wrn168_split8_cifar100_96clients_96choose_650rounds" --data=${DATA_DIR}