from PIL import Image
from cv2 import split
from torch.utils.data import DataLoader, Dataset
import torch
class SkinData(Dataset):
    def __init__(self, df, transform = None, split_factor=1):
        self.df = df
        self.transform = transform
        self.split_factor = split_factor
        self.test_same_view = 0
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, index):
        Xs=[]
        X = Image.open(self.df['path'][index]).resize((64, 64))
        y = torch.tensor(int(self.df['target'][index]))
        if (self.test_same_view):
            #print("Testing same view of data")
            if self.transform:
                aug = self.transform(X)
                for i in range(self.split_factor):
                    Xs.append(aug)
        else:
            #print("Not same view of data")
            if self.transform:
                for i in range(self.split_factor):
                    Xs.append(self.transform(X))
           
        
        return torch.cat(Xs, dim=0), y