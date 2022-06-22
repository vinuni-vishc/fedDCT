import os 
# Home directory
 
HOME = "/home/ubuntu/quan.nm/feddct/fedDCT"
# data directory 
data_dir=HOME
# Name of experiment
# SPID = "fed_resnet110_split4_cifar100_128_01_sched_rand_cluster_40clients_single_branch"
SPID = "splitfed_wideresnet16_8_split1_cifar10_20choose_300epochs" 
GPU_ID = "1"
# Model directory 
model_dir=str(HOME)+"/models/splitnet/"+str(SPID)
model_dir_fed=str(HOME)+"/models/splitnet/fed/"+str(SPID)
