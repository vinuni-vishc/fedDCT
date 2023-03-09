from PIL import Image
from cv2 import split
from torch.utils.data import DataLoader, Dataset
import torch
import os
import glob

folder_path = '/media/quannm/150be5a2-6412-4a07-a0ea-7a6184302592/code/fed-dct/splitnet/dataset/pill_base'


class PillDataBase(Dataset):
    def __init__(self, train = True,transform = None, split_factor=1):
        self.train = train
        self.transform = transform
        self.split_factor = split_factor
        # self.dataset = self.get_data()
        self.dataset = self.get_data()

    def __len__(self):
        return len(self.dataset)
    def get_data(self):
        dataset = []
        if self.train:
            #folder_path_ = folder_path+'/train'
            txt_path = folder_path+ '/train.txt'
        
        else: 
            #folder_path_ = folder_path+'/test'
            txt_path = folder_path+ '/test.txt'
        
        with open(txt_path,'r') as fr : 
            lines = fr.readlines()
            for line in lines:
                #print(line)
                fn, ln = line.split(' ')
                fn = fn.replace(f'/home/tung/Tung/research/Open-Pill/FACIL/data/Pill_Base_X', '/media/quannm/150be5a2-6412-4a07-a0ea-7a6184302592/code/fed-dct/splitnet/dataset/pill_base')
                #print ([fn,int(ln)])
                dataset.append([fn,int(ln)])
                #print(os.path.isfile(fn))
                #exit(0)
        return dataset
    # def  get_data(self):
        
        
    #     dataset = []
    #     for idc, (clsn, kn) in enumerate(class_map.items()):
    #         folder_class = os.path.join(folder_path, clsn)
    #         files_jpg = glob.glob(os.path.join(folder_class, '**', '*.jpg'), recursive=True)
    #         # dataset.append([ [fn, kn] for fn in files_jpg ])
    #         for fn in files_jpg:
    #             dataset.append([fn, kn])
    #     # dataset = np.array(dataset)
    #     # dataset = dataset.reshape(-1, 2)
    #     return dataset
        

    def __getitem__(self, index):
        Xs = []
        X = Image.open(self.dataset[index][0])
        y = torch.tensor(int(self.dataset[index][1]))
        
        if self.transform:
            for i in range(self.split_factor):
                Xs.append(self.transform(X))
           
        #return Xs,y
        return torch.cat(Xs, dim=0), y

if __name__ == "__main__":

    dataset = PillDataBase()
    # print(len(dataset))
    # print(dataset[0])