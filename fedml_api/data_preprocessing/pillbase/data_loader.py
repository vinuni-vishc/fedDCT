import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from dataset.pill_dataset_base import PillDataBase
from config import HOME
import pickle
from PIL import Image
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)



def load_partition_data_pillbase(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    train_data_num = 8161
    test_data_num = 1619
    mean = [0.4550, 0.5239, 0.5653]
    std = [0.2460, 0.2446, 0.2252]	
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.1, 1.0),
																				interpolation=Image.BILINEAR),
										transforms.RandomHorizontalFlip()
										])
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(mean,
														std))
    train_transform.transforms.append(transforms.RandomErasing(p=0.5, scale=(0.05, 0.12),
															ratio=(0.5, 1.5), value=0))
    train_dataset = PillDataBase(True,transform=train_transform,split_factor=1)
    train_sampler = None
    train_data_global = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_data_global = None
    data_local_num_dict =None 
    train_data_local_dict = None
    val_transform = transforms.Compose([transforms.Resize(int(224 * 1.15),
																	interpolation=Image.BILINEAR),
												transforms.CenterCrop(224),
												transforms.ToTensor(),
												transforms.Normalize(mean,
																		std),
	])
    val_dataset = PillDataBase(False,transform=val_transform,split_factor=1)
    images_per_client = int(len(val_dataset)/ client_number) 
    print("Images per client is "+ str(images_per_client))
    data_split = [images_per_client for _ in range(client_number-1)]
    data_split.append(len(val_dataset)-images_per_client*(client_number-1))
    #print(data_split)
    testdata_split = torch.utils.data.random_split(val_dataset,data_split,generator=torch.Generator().manual_seed(68))
    test_data_local_dict = [torch.utils.data.DataLoader(x,
														batch_size=32,
														shuffle=(train_sampler is None),
														drop_last=True,
														sampler=train_sampler,
													    ) for x in testdata_split]
    class_num = 98
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
