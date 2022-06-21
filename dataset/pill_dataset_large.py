from PIL import Image
from cv2 import split
from torch.utils.data import DataLoader, Dataset
import torch
import os
import glob

folder_paths = ['/media/quannm/150be5a2-6412-4a07-a0ea-7a6184302592/code/fed-dct/splitnet/dataset/pill_large/pill_img_by_class_train',
        '/media/quannm/150be5a2-6412-4a07-a0ea-7a6184302592/code/fed-dct/splitnet/dataset/pill_large/pill_img_by_class_test'] 

class PillDataLarge(Dataset):
    def __init__(self, train = True,transform = None, split_factor=1):
        self.train = train
        self.transform = transform
        self.split_factor = split_factor
        self.dataset = self.get_data()
        # self.folder_paths = ['/media/quannm/150be5a2-6412-4a07-a0ea-7a6184302592/code/fed-dct/splitnet/dataset/pill_large/pill_img_by_class_train',
        # '/media/quannm/150be5a2-6412-4a07-a0ea-7a6184302592/code/fed-dct/splitnet/dataset/pill_large/pill_img_by_class_test'] 

    def __len__(self):
        return len(self.dataset)
        
    def  get_data(self):
        if self.train:
            folder_path = folder_paths[0]
        else: 
            folder_path = folder_paths[1]
        class_names = sorted(os.listdir(folder_path))
        #print(len(class_names))
        class_map = {k:id_k for id_k,k in enumerate(class_names)}
        #print(class_names)
        #print(class_map)
        dataset = []
        for idc, (clsn, kn) in enumerate(class_map.items()):
            folder_class = os.path.join(folder_path, clsn)
            files_jpg = glob.glob(os.path.join(folder_class, '**', '*.jpg'), recursive=True)
            # dataset.append([ [fn, kn] for fn in files_jpg ])
            for fn in files_jpg:
                dataset.append([fn, kn])
        # dataset = np.array(dataset)
        # dataset = dataset.reshape(-1, 2)
        return dataset
        

    def __getitem__(self, index):
        Xs = []
        X = Image.open(self.dataset[index][0])
        y = torch.tensor(int(self.dataset[index][1]))
        
        if self.transform:
            for i in range(self.split_factor):
                Xs.append(self.transform(X))
           
        # return X,y
        return torch.cat(Xs, dim=0), y

if __name__ == "__main__":
    dataset = PillDataLarge()
    print(len(dataset))
    print(dataset[0])