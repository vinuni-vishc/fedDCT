import pickle as pkl
import pandas as pd
from config import  HOME

train_file = open('file.pickle', 'rb')
object =pkl.load(train_file)
df = pd.DataFrame(object)
df.to_csv(r'file.csv')