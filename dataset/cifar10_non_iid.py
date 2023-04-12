#### get cifar dataset in x and y form
import torchvision
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from config import HOME
np.random.seed(68)
random.seed(68)

def get_cifar10(data_dir):
  '''Return CIFAR10 train/test data and labels as numpy arrays'''
  data_train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
  data_test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True) 
  
  x_train, y_train = data_train.data.transpose((0,3,1,2)), np.array(data_train.targets)
  x_test, y_test = data_test.data.transpose((0,3,1,2)), np.array(data_test.targets)
  
  return x_train, y_train, x_test, y_test

def print_image_data_stats_train(data_train, labels_train):
  print("\nData: ")
  print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
      np.min(labels_train), np.max(labels_train)))

def print_image_data_stats_test(data_test, labels_test):
  print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_test.shape, labels_test.shape, np.min(data_test), np.max(data_test),
      np.min(labels_test), np.max(labels_test)))
  
  
def clients_rand(train_len, nclients):
  '''
  train_len: size of the train data
  nclients: number of clients
  
  Returns: to_ret
  
  This function creates a random distribution 
  for the clients, i.e. number of images each client 
  possess.
  '''
  #np.random.seed(68)
  client_tmp=[]
  sum_=0
  #### creating random values for each client ####
  for i in range(nclients-1):
    tmp=random.randint(10,100)
    sum_+=tmp
    client_tmp.append(tmp)

  client_tmp= np.array(client_tmp)
  #### using those random values as weights ####
  clients_dist= ((client_tmp/sum_)*train_len).astype(int)
  num  = train_len - clients_dist.sum()
  to_ret = list(clients_dist)
  to_ret.append(num)
  return to_ret

def split_image_data_realwd(data, labels, n_clients=100, verbose=True):
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
  #np.random.seed(68)
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
  n_data = data.shape[0]
  n_labels = np.max(labels) + 1
  #### classes for each client ####
  tmp = [np.random.randint(1,10) for i in range(n_clients)]
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
      
   
  
  clients_split = np.array(clients_split)
  print_split(clients_split)
  return clients_split

def split_image_data(data, labels, n_clients=100, classes_per_client=10, shuffle=True, verbose=True):
  '''
  Splits (data, labels) among 'n_clients s.t. every client can holds 'classes_per_client' number of classes
  Input:
    data : [n_data x shape]
    labels : [n_data (x 1)] from 0 to n_labels
    n_clients : number of clients
    classes_per_client : number of classes per client
    shuffle : True/False => True for shuffling the dataset, False otherwise
    verbose : True/False => True for printing some info, False otherwise
  Output:
    clients_split : client data into desired format
  '''
  #### constants #### 
  #np.random.seed(68)
  n_data = data.shape[0]
  n_labels = np.max(labels) + 1


  ### client distribution ####
  data_per_client = clients_rand(len(data), n_clients)
  data_per_client_per_class = [np.maximum(1,nd // classes_per_client) for nd in data_per_client]
  
  # sort for labels
  data_idcs = [[] for i in range(n_labels)]
  for j, label in enumerate(labels):
    data_idcs[label] += [j]
  if shuffle:
    for idcs in data_idcs:
      np.random.shuffle(idcs)
    
  # split data among clients
  clients_split = []
  c = 0
  for i in range(n_clients):
    client_idcs = []
        
    budget = data_per_client[i]
    c = np.random.randint(n_labels)
    while budget > 0:
      take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)
      
      client_idcs += data_idcs[c][:take]
      data_idcs[c] = data_idcs[c][take:]
      
      budget -= take
      c = (c + 1) % n_labels
      
    clients_split += [(data[client_idcs], labels[client_idcs])]

  def print_split(clients_split): 
    print("Data split:")
    for i, client in enumerate(clients_split):
      split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
      print(" - Client {}: {}".format(i,split))
    print()
      
  
  clients_split = np.array(clients_split)
  print_split(clients_split)
  return clients_split


def shuffle_list(data):
  '''
  This function returns the shuffled data
  '''
  #np.random.seed(68)
  #random.seed(68)
  for i in range(len(data)):
    tmp_len= len(data[i][0])
    index = [i for i in range(tmp_len)]
    random.shuffle(index)
    data[i][0],data[i][1] = shuffle_list_data(data[i][0],data[i][1])
  return data

def shuffle_list_data(x, y):
  '''
  This function is a helper function, shuffles an
  array while maintaining the mapping between x and y
  '''
  inds = list(range(len(x)))
  #random.seed(68)
  random.shuffle(inds)
  return x[inds],y[inds]

class CustomImageDataset(Dataset):
  '''
  A custom Dataset class for images
  inputs : numpy array [n_data x shape]
  labels : numpy array [n_data (x 1)]
  '''
  def __init__(self, inputs, labels, transforms=None,split_factor = 1):
      assert inputs.shape[0] == labels.shape[0]
      self.inputs = torch.Tensor(inputs)
      self.labels = labels
      self.transforms = transforms 
      self.split_factor = split_factor
  def __getitem__(self, index):
      img, label = self.inputs[index], self.labels[index]
      imgs = []
      if self.transforms is not None:
        for i in range(self.split_factor):
          imgs.append(self.transforms(img))
      # print (torch.cat(imgs, dim=0).size())
      # print(label)
      return (torch.cat(imgs, dim=0), label)

  def __len__(self):
      return self.inputs.shape[0]
          

def get_default_data_transforms(train=True, verbose=True):
  transforms_train = {
  'cifar10' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#(0.24703223, 0.24348513, 0.26158784)
  }
  transforms_eval = {    
  'cifar10' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  }
  if verbose:
    print("\nData preprocessing: ")
    for transformation in transforms_train['cifar10'].transforms:
      print(' -', transformation)
    print()

  return (transforms_train['cifar10'], transforms_eval['cifar10'])

def get_data_loaders_train(data_dir, nclients,batch_size,classes_pc=10 ,verbose=True,transforms_train=None, transforms_eval=None,non_iid=None,split_factor=1):
  
  x_train, y_train, _, _ = get_cifar10(data_dir)

  if verbose:
    print_image_data_stats_train(x_train, y_train)
  #print_image_data_stats(x_train, y_train, x_test, y_test)
  #transforms_train, transforms_eval = get_default_data_transforms(verbose=False)
  #print(transforms_train)
  split = None
  print('Non diid is '+ str(non_iid))
  if non_iid == 'quantity_skew':
    split = split_image_data_realwd(x_train, y_train, 
          n_clients=nclients, verbose=verbose)
  elif non_iid == 'label_skew':
    split = split_image_data(x_train, y_train, n_clients=nclients, 
          classes_per_client=classes_pc, verbose=verbose)
  split_tmp = shuffle_list(split)
  
  client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train,split_factor=split_factor), 
                                                                batch_size=batch_size, shuffle=True) for x, y in split_tmp]

  #test_loader  = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100, shuffle=False) 

  return client_loaders

def get_data_loaders_test(data_dir,nclients,batch_size,classes_pc=10 ,verbose=True,transforms_train=None, transforms_eval=None,non_iid=None,split_factor=1):
  
  _, _, x_test, y_test = get_cifar10(data_dir)

  if verbose:
    print_image_data_stats_test(x_test, y_test)
  #print_image_data_stats(x_train, y_train, x_test, y_test)
  #transforms_train, transforms_eval = get_default_data_transforms(verbose=False)
  #print(transforms_train)

  test_loader  = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval,split_factor=1), batch_size=100, shuffle=False,) 

  return test_loader