#### get cifar dataset in x and y form
import torchvision
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from .cifar10_non_iid import *
np.random.seed(68)
random.seed(68)

def get_cifar100():
  '''Return CIFAR100 train/test data and labels as numpy arrays'''
  data_train = torchvision.datasets.CIFAR100('/home/ubuntu/quan.nm209043/splitnet/dataset/cifar/train', train=True, download=True)
  data_test = torchvision.datasets.CIFAR100('/home/ubuntu/quan.nm209043/splitnet/dataset/cifar/val', train=False, download=True) 
  
  x_train, y_train = data_train.data.transpose((0,3,1,2)), np.array(data_train.targets)
  x_test, y_test = data_test.data.transpose((0,3,1,2)), np.array(data_test.targets)
  
  return x_train, y_train, x_test, y_test

def split_image_data_realwd_cf100(data, labels, n_clients=100, verbose=True):
  '''
  Splits (data, labels) among 'n_clients s.t. every client can holds any number of classes which is trying to simulate real world dataset
  Input:
    data : [n_data x shape]
    labels : [n_data (x 1)] from 0 to n_labels(10)
    n_clients : number of clients
    verbose : True/False => True for printing some info, False otherwise
  Output:
    clients_split : splitted client data into desired format
  '''
  n_labels = np.max(labels) + 1
#   print("number of labels are  "+ n_labels)
  def break_into(n,m):
    ''' 
    return m random integers with sum equal to n 
    '''
    to_ret = [1 for i in range(m)]
    for i in range(n-m):
        ind = random.randint(0,m-1)
        to_ret[ind] += 1
    return to_ret

  #### constants ####
  n_classes = len(set(labels))
  classes = list(range(n_classes))
  np.random.shuffle(classes)
  label_indcs  = [list(np.where(labels==class_)[0]) for class_ in classes]
  
  #### classes for each client ####
  tmp = [np.random.randint(1,100) for i in range(n_clients)]
  total_partition = sum(tmp)

  #### create partition among classes to fulfill criteria for clients ####
  class_partition = break_into(total_partition, len(classes))

  #### applying greedy approach first come and first serve ####
  class_partition = sorted(class_partition,reverse=True)
  class_partition_split = {}

  #### based on class partition, partitioning the label indexes ###
  for ind, class_ in enumerate(classes):
      class_partition_split[class_] = [list(i) for i in np.array_split(label_indcs[ind],class_partition[ind])]
      
#   print([len(class_partition_split[key]) for key in  class_partition_split.keys()])

  clients_split = []
  count = 0
  for i in range(n_clients):
    n = tmp[i]
    j = 0
    indcs = []

    while n>0:
        class_ = classes[j]
        if len(class_partition_split[class_])>0:
            indcs.extend(class_partition_split[class_][-1])
            count+=len(class_partition_split[class_][-1])
            class_partition_split[class_].pop()
            n-=1
        j+=1

    ##### sorting classes based on the number of examples it has #####
    classes = sorted(classes,key=lambda x:len(class_partition_split[x]),reverse=True)
    if n>0:
        raise ValueError(" Unable to fulfill the criteria ")
    clients_split.append([data[indcs], labels[indcs]])
#   print(class_partition_split)
#   print("total example ",count)


  def print_split(clients_split): 
    print("Data split:")
    for i, client in enumerate(clients_split):
      split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
      print(" - Client {}: {}".format(i,split))
    print()
      
  if verbose:
    print_split(clients_split)
  
  clients_split = np.array(clients_split)
  
  return clients_split


def get_default_data_transforms_cf100(train=True, verbose=True):
  transforms_train = {
  'cifar100' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#(0.24703223, 0.24348513, 0.26158784)
  }
  transforms_eval = {    
  'cifar100' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  }
  if verbose:
    print("\nData preprocessing: ")
    for transformation in transforms_train['cifar100'].transforms:
      print(' -', transformation)
    print()

  return (transforms_train['cifar100'], transforms_eval['cifar100'])

def get_data_loaders_train_cf100(nclients,batch_size,classes_pc=10 ,verbose=True,transforms_train=None, transforms_eval=None,non_iid=None,split_factor=1):
  
  x_train, y_train, _, _ = get_cifar100()

  if verbose:
    print_image_data_stats_train(x_train, y_train)
  #print_image_data_stats(x_train, y_train, x_test, y_test)
  #transforms_train, transforms_eval = get_default_data_transforms(verbose=False)
  #print(transforms_train)
  split = None
  print('Non diid is '+ str(non_iid))
  if non_iid == 'quantity_skew':
    split = split_image_data_realwd_cf100(x_train, y_train, 
          n_clients=nclients, verbose=verbose)
#   elif non_iid == 'label_skew':
#     split = split_image_data(x_train, y_train, n_clients=nclients, 
#           classes_per_client=classes_pc, verbose=verbose)
  split_tmp = shuffle_list(split)
  
  client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train,split_factor=split_factor), 
                                                                batch_size=batch_size, shuffle=True) for x, y in split_tmp]

  #test_loader  = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100, shuffle=False) 

  return client_loaders

def get_data_loaders_test_cf100(nclients,batch_size,classes_pc=10 ,verbose=True,transforms_train=None, transforms_eval=None,non_iid=None,split_factor=1):
  
  _, _, x_test, y_test = get_cifar100()

  if verbose:
    print_image_data_stats_test(x_test, y_test)
  #print_image_data_stats(x_train, y_train, x_test, y_test)
  #transforms_train, transforms_eval = get_default_data_transforms(verbose=False)
  #print(transforms_train)

  test_loader  = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval,split_factor=1), batch_size=100, shuffle=False,) 

  return test_loader