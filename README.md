# Federated Divide and Cotraining

## Run Experiments for A Nguyen

```
(1) CIFAR100 Ablation Num client Training scripts 
```
```
# 80 client
python train_feddct.py --is_fed=1 --fixed_cluster=0 --split_factor=4 --num_clusters=20 --num_selected=20 --arch=resnet110sl --dataset=cifar100 --num_classes=100 --is_single_branch=0 --is_amp=0 --num_rounds=650 --fed_epochs=1 --spid="feddct_resnet110_split4_cifar100_80clients_80choose_650rounds"
# 100 client
python train_feddct.py --is_fed=1 --fixed_cluster=0 --split_factor=4 --num_clusters=25 --num_selected=25 --arch=resnet110sl --dataset=cifar100 --num_classes=100 --is_single_branch=0 --is_amp=0 --num_rounds=650 --fed_epochs=1 --spid="feddct_resnet110_split4_cifar100_100clients_100choose_650rounds"
```
(2) Tensorboard
```
# You can visualize the result using tensorboard 
tensorboard --logdir models/splitnet/
```
