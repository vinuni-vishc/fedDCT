import pandas as pd
import os
from glob import glob
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
import pickle
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms
df = pd.read_csv('dataset/ham10000/data/HAM10000_metadata.csv')
print(df.head())

lesion_type = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# merging both folders of HAM1000 dataset -- part1 and part2 -- into a single directory
imageid_path = {os.path.splitext(os.path.basename(x))[0]: x
                for x in glob(os.path.join("dataset/ham10000/data", '*', '*.jpg'))}


df['path'] = df['image_id'].map(imageid_path.get)
df['cell_type'] = df['dx'].map(lesion_type.get)
df['target'] = pd.Categorical(df['cell_type']).codes
print(df['cell_type'].value_counts())
print(df['target'].value_counts())

#==============================================================
# Custom dataset prepration in Pytorch format
class SkinData(Dataset):
    def __init__(self, df, transform = None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, index):
        
        X = Image.open(self.df['path'][index]).resize((64, 64))
        y = torch.tensor(int(self.df['target'][index]))
        
        if self.transform:
            X = self.transform(X)
        
        return X, y


#=============================================================================
# Train-test split    
train, test = train_test_split(df, test_size = 0.2)

train = train.reset_index()
test = test.reset_index()

ham10000_train = open("ham10000_train.pickle", "wb")
pickle.dump(train, ham10000_train)
ham10000_train.close()
ham10000_test = open("ham10000_test.pickle", "wb")
pickle.dump(test, ham10000_test)
ham10000_test.close()