import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from dataset.ham10000 import SkinData
from config import HOME
import pickle
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)



def load_partition_data_ham10000(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    train_data_num = 8012
    test_data_num = 2003
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_file = open(HOME+'/dataset/ham10000/ham10000_train.pickle','rb')
    train = pickle.load(train_file)
    train_file.close
    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), 
                        transforms.RandomVerticalFlip(),
						transforms.RandomHorizontalFlip(), #new
						transforms.RandomAdjustSharpness(random.uniform(0, 4.0)),
						transforms.RandomAutocontrast(),
                        transforms.Pad(3),
                        transforms.RandomRotation(10),
                        transforms.CenterCrop(64),
                        transforms.ToTensor(), 
                        transforms.Normalize(mean = mean, std = std)
    ])
			
    train_dataset = SkinData(train, transform = train_transforms,split_factor=1)
    train_sampler = None
    train_data_global = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_data_global = None
    data_local_num_dict =None 
    train_data_local_dict = None
    test_file = open(HOME+"/dataset/ham10000/ham10000_test.pickle","rb")
    test = pickle.load(test_file)
    test_file.close
    val_transforms = transforms.Compose([
                        transforms.Pad(3),
                        transforms.CenterCrop(64),
                        transforms.ToTensor(), 
                        transforms.Normalize(mean = mean, std = std)
    ]) 
    val_dataset = SkinData(test, transform = val_transforms,split_factor=1)   
    images_per_client = int(len(val_dataset)/ client_number) 
    print("Images per client is "+ str(images_per_client))
    data_split = [images_per_client for _ in range(client_number-1)]
    data_split.append(len(val_dataset)-images_per_client*(client_number-1))
    print(data_split)
    testdata_split = torch.utils.data.random_split(val_dataset,data_split,generator=torch.Generator().manual_seed(68))
    test_data_local_dict = [torch.utils.data.DataLoader(x,
														batch_size=32,#needs to be < size of dataset on each client on test set)
														shuffle=(train_sampler is None),
														drop_last=True,
														sampler=train_sampler,
													    ) for x in testdata_split]
    class_num = 7
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
