# Federated Divide and Cotraining
**NOTE: We are going to release a new version on Arxiv soon.**  

V2: Release date 13.4.2022: Add new training script for FedGKT. 

This repository contains the code and experiments for the paper: 

> [FedDCT: A Novel Federated Learning Approach
for Training Large Convolutional Neural Networks
on Resource-constrained Devices](https://arxiv.org/abs/2211.10948)
>
> submitted to [ IEEE Transactions on Network and Service Management]([https://signalprocessingsociety.org/publications-resources/ieee-journal-selected-topics-signal-processing](https://www.comsoc.org/publications/journals/ieee-tnsm)) .
> Arxiv link: https://arxiv.org/abs/2211.10948

We propose FedDCT, a novel approach that reduces the memory requirements of training in the FL setting. FedDCT allows lightweight edge devices to overcome resource constraints and participate in the FL task to train large CNN models. To the best of our knowledge, this is the first work that enables the training of large deep learning networks in edge computing devices in FL settings. The proposed FedDCT architecture can divide large CNN networks into sub-networks and co-training them with local clients in parallel. This co-training process allows clients from the same cluster to learn representations from each other. The server takes a weighted average of all the ensemble models trained by all the clusters to improve the learning performance of the global model.

We conduct extensive experiments on natural and real-world medical image datasets. FedDCT significantly outperforms a set of the latest state-of-the-art FL methods and sets new state-of-the-art results on these datasets.


![](images/feddctvsfedavgfinal.png)
*Illustration of a typical FL architecture (left) and the proposed FedDCT (right). FedDCT enables a cluster of clients to cooperatively train a big deep learning model by breaking it into an ensemble of several small sub-models and training these sub-models on several devices concurrently, while maintaining client data privacy. We refer to this algorithm as Federated Divide and Co-training or FedDCT in short.*

![](images/feddct_quan.png)
*Federated Co-training of an ensemble model across S clients in a cluster of FedDCT*
## General Guidelines

We provide this codebase which contains:
* Training algorithms for 4 distributed collaborative machile learning algorithms: [FedAvg](https://arxiv.org/pdf/1602.05629.pdf), [FedProx](https://arxiv.org/abs/1812.06127), [SplitFed](https://arxiv.org/abs/2004.12088),[FedGKT](https://arxiv.org/abs/2007.14513) and our algorithm [FedDCT]() and normal centralized training.
* Training on 4 datasets : CIFAR10, CIFAR100, [HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000) and [VAIPE](https://smarthealth.vinuni.edu.vn/resources/) dataset. To make this work reproducible and encourage new advances, we make all images and annotations of the VAIPE dataset publicly available as a part of a bigger dataset that we will release on our project [website](https://smarthealth.vinuni.edu.vn/resources/)
* This codebase is easy to extend to other FL algorithms, models and datasets.
* For hyperparameters, see `params/train_params.py`

## Preparations
### Dataset generation

To prepare CIFAR10 and CIFAR100 for training, you can just run the training script to allow for self downloading. For other datasets, please download to respective folder (i.e., download [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) dataset to `dataset/ham10000` and unzip. ). Use `dump_pickle_ham10000.py` to generate train-test set. Alternatively, download from [here](https://drive.google.com/drive/folders/1k1h9YTaNcfH8UM8GBZ8hzN17LDvLTYcI?usp=sharing) for pre-generated train-test split. For pill dataset, download [here](https://drive.google.com/drive/folders/1k1h9YTaNcfH8UM8GBZ8hzN17LDvLTYcI?usp=sharing).

The statistics of real federated datasets are summarized as follows.
<center>

| Dataset       | Devices         | Training Samples|Num classes <br> |
| ------------- |-------------| -----| ---|
| CIFAR10      | 20 | 50000 | 10| 
| CIFAR100     | 20      |   50000 |100|
| HAM10000 | 20    |    8016 | 7|
| VAIPE| 20      |    8161 | 98|

</center>

### Downloading dependencies

```
Pytorch 1.10.2
OpenCV 4.5.5
```

## Run Experiments
![](images/accuracy.png)
*Top-1 accuracy (%) of FedDCT compared to state-of-the-art FL methods on the test sets of CIFAR-10, CIFAR-100,
HAM10000, and VAIPE datasets.*<br /> 

(1) Please add --spid for experiment name add the end of each training script. For example
```
python train_fedgkt.py --is_fed=1 --fixed_cluster=0 --split_factor=1 --num_clusters=20 --num_selected=20 --arch=wide_resnet16_8 --dataset=cifar10 --num_classes=10 --is_single_branch=0 --is_amp=0 --num_rounds=300 --fed_epochs=1 --spid="fedgkt_wrn168_split1_cifar10_20clients_20choose_300rounds"
```
(2) CIFAR10 Training scripts 
```
# Centralized Training 
python train_centralized.py  --is_fed=0 --split_factor=1 --dataset=cifar10 --num_classes=10 --is_single_branch=0 --is_amp=0 --arch=wide_resnet16_8 --epochs=300
# FedAvg
python train_fedavg.py --is_fed=1 --fixed_cluster=0 --split_factor=1 --num_clusters=20 --num_selected=20 --arch=wide_resnet16_8 --dataset=cifar10 --num_classes=10 --is_single_branch=0 --is_amp=0 --num_rounds=300 --fed_epochs=1 
# FedDCT
python train_feddct.py --is_fed=1 --fixed_cluster=0 --split_factor=4 --num_clusters=5 --num_selected=5 --arch=wide_resnetsl16_8 --dataset=cifar10 --num_classes=10 --is_single_branch=0 --is_amp=0 --num_rounds=300 --fed_epochs=1
#FedProx
python train_fedprox.py --is_fed=1 --fixed_cluster=0 --split_factor=1 --num_clusters=20 --num_selected=20 --arch=wide_resnet16_8 --dataset=cifar10 --num_classes=10 --is_single_branch=0 --is_amp=0 --num_rounds=300 --fed_epochs=1
#SplitFed
python train_splitfed.py --is_fed=1 --fixed_cluster=0 --split_factor=1 --num_clusters=20 --num_selected=20 --arch=wide_resnetsl16_8 --dataset=cifar10 --num_classes=10 --is_single_branch=0 --is_amp=0 --num_rounds=300 --fed_epochs=1
#FedGKT
python train_fedgkt.py --is_fed=1 --fixed_cluster=0 --split_factor=1 --num_clusters=20 --num_selected=20 --arch=wide_resnet16_8 --dataset=cifar10 --num_classes=10 --is_single_branch=0 --is_amp=0 --num_rounds=300 --fed_epochs=1 --spid="fedgkt_wrn168_split1_cifar10_20clients_20choose_300rounds"
```
(3) CIFAR100 Training scripts 
```
# Centralized Training 
python train_centralized.py  --is_fed=0 --split_factor=1 --dataset=cifar100 --num_classes=100 --is_single_branch=0 --is_amp=0 --arch=resnet110 --epochs=650
# FedAvg
python train_fedavg.py --is_fed=1 --fixed_cluster=0 --split_factor=1 --num_clusters=20 --num_selected=20 --arch=resnet110 --dataset=cifar100 --num_classes=100 --is_single_branch=0 --is_amp=0 --num_rounds=650 --fed_epochs=1
# FedDCT
python train_feddct.py --is_fed=1 --fixed_cluster=0 --split_factor=4 --num_clusters=5 --num_selected=5 --arch=resnet110sl --dataset=cifar100 --num_classes=100 --is_single_branch=0 --is_amp=0 --num_rounds=650 --fed_epochs=1
#FedProx
python train_fedprox.py --is_fed=1 --fixed_cluster=0 --split_factor=1 --num_clusters=20 --num_selected=20 --arch=resnet110 --dataset=cifar100 --num_classes=100 --is_single_branch=0 --is_amp=0 --num_rounds=650 --fed_epochs=1
#SplitFed
python train_splitfed.py --is_fed=1 --fixed_cluster=0 --split_factor=1 --num_clusters=20 --num_selected=20 --arch=resnet110sl --dataset=cifar100 --num_classes=100 --is_single_branch=0 --is_amp=0 --num_rounds=650 --fed_epochs=1
#FedGKT
python train_fedgkt.py --is_fed=1 --fixed_cluster=0 --split_factor=1 --num_clusters=20 --num_selected=20 --arch=resnet110sl --dataset=cifar100 --num_classes=100 --is_single_branch=0 --is_amp=0 --num_rounds=650 --fed_epochs=1 --spid="fedgkt_resnet110_split1_cifar100_20clients_20choose_650rounds" 
```
(3) HAM10000 Training scripts 
```
# Centralized Training 
python train_centralized.py  --is_fed=0 --split_factor=1 --dataset=ham10000 --num_classes=7 --is_single_branch=0 --is_amp=0 --arch=wide_resnet50_2 --num_rounds=200 --crop_size=64 --lr=1e-2 --slow_start_lr=1e-3
# FedAvg
python train_fedavg.py --is_fed=1 --fixed_cluster=0 --split_factor=1 --num_clusters=20 --num_selected=20 --dataset=ham10000 --num_classes=7 --is_single_branch=0 --is_amp=0 --fed_epochs=1 --arch=wide_resnet50_2 --num_rounds=200 --crop_size=64 --lr=1e-2 --slow_start_lr=1e-3
# FedDCT
python train_feddct.py --is_fed=1 --fixed_cluster=0 --split_factor=4 --num_clusters=5 --num_selected=5 --arch=wide_resnetsl50_2 --dataset=ham10000 --num_classes=7 --is_single_branch=0 --is_amp=0 --num_rounds=200 --fed_epochs=1 --lr=1e-2 --slow_start_lr=1e-3 --crop_size=64
#FedProx
python train_fedprox.py --is_fed=1 --fixed_cluster=0 --split_factor=1 --num_clusters=20 --num_selected=20 --dataset=ham10000 --num_classes=7 --is_single_branch=0 --is_amp=0 --fed_epochs=1 --arch=wide_resnet50_2 --num_rounds=200 --crop_size=64 --lr=1e-2 --slow_start_lr=1e-3
#SplitFed
python train_splitfed.py --is_fed=1 --fixed_cluster=0 --split_factor=1 --num_clusters=20 --num_selected=20 --arch=wide_resnetsl50_2 --dataset=ham10000 --num_classes=7 --is_single_branch=0 --is_amp=0 --num_rounds=200 --fed_epochs=1 --lr=1e-2 --slow_start_lr=1e-3 --crop_size=64
#FedGKT
python train_fedgkt.py --is_fed=1 --fixed_cluster=0 --split_factor=1 --num_clusters=20 --num_selected=20 --arch=wide_resnetsl50_2 --dataset=ham10000 --num_classes=7 --is_single_branch=0 --is_amp=0 --num_rounds=200 --fed_epochs=1 --lr=1e-2 --slow_start_lr=1e-3 --crop_size=64 --spid="fedgkt_wrn502_split1_ham10000_20clients_20choose_650rounds" 
```
(4) VAIPE dataset
```
# FedAvg
python train_fedavg.py --is_fed=1 --fixed_cluster=0 --split_factor=1 --num_clusters=20 --num_selected=20 --arch=wide_resnet50_2 --dataset=pill_base --num_classes=98 --is_single_branch=0 --is_amp=0 --num_rounds=350 --crop_size=224 --batch_size=32 --fed_epochs=1 
# FedProx
python train_fedprox.py --is_fed=1 --fixed_cluster=0 --split_factor=1 --num_clusters=20 --num_selected=20 --arch=wide_resnet50_2 --dataset=pill_base --num_classes=98 --is_single_branch=0 --is_amp=0 --num_rounds=350 --crop_size=224 --batch_size=32 --fed_epochs=1 
# FedDCT
python train_feddct.py --is_fed=1 --fixed_cluster=0 --split_factor=4 --num_clusters=5 --num_selected=5 --arch=wide_resnetsl50_2 --dataset=pill_base --num_classes=98 --is_single_branch=0 --is_amp=0 --num_rounds=350 --fed_epochs=1 --batch_size=32 --crop_size=224
# SplitFed
python train_splitfed.py --is_fed=1 --fixed_cluster=0 --split_factor=1 --num_clusters=20 --num_selected=20 --arch=wide_resnetsl50_2 --dataset=pill_base --num_classes=98 --is_single_branch=0 --is_amp=0 --num_rounds=350 --fed_epochs=1 --batch_size=32 --crop_size=224
# FedGKT
python train_fedgkt.py --is_fed=1 --fixed_cluster=0 --split_factor=1 --num_clusters=20 --num_selected=20 --arch=wide_resnetsl50_2 --dataset=pill_base --num_classes=98 --is_single_branch=0 --is_amp=0 --num_rounds=350 --fed_epochs=1 --batch_size=32 --crop_size=224 --spid="fedgkt_wrn502_split1_pill_base_20clients_20choose_350rounds"
```
(5) Non-iid CIFAR-10 and CIFAR-100
```
Please add --cifar10_non_iid="quantity_skew" for CIFAR-10 non iid and --cifar100_non_iid="quantity_skew" for non-iid CIFAR-100
```
(6) Tensorboard
```
# You can visualize the result using tensorboard 
tensorboard --logdir models/splitnet/
```