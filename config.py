import os 
# Home directory
 
HOME = "/home/ubuntu/quan.nm209043/splitnet"
# data directory 
data_dir=HOME
# Name of experiment
# SPID = "fed_resnet110_split4_cifar100_128_01_sched_rand_cluster_40clients_single_branch"
SPID = "feddct_wideresnet50_2_layer_split8_ham10000_16clients_16choose_200rounds" 

# Model directory 
model_dir=str(HOME)+"/models/splitnet/"+str(SPID)
model_dir_fed=str(HOME)+"/models/splitnet/fed/"+str(SPID)
