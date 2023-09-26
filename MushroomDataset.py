import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch 
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import DataLoader

class MushroomDataSet(torch.utils.data.Dataset):
    DATA_FOLDER = 'data/'
    def __init__(self, split: str):
        super(MushroomDataSet, self).__init__()
        self.split = split
        # store the raw tensors
        self.preprocess_data()
        self.configure_dataloader()

    def preprocess_data(self):
        data =  pd.read_csv(self.DATA_FOLDER + 'agaricus.data', sep=",", index_col=None)
        with open(self.DATA_FOLDER + 'agaricus.names') as f:
            cols = ["poisonous"]
            lines = ""
            for line in f:
                lines += line
            lines=lines.split("7. Attribute Information: (classes: edible=e, poisonous=p)")[1]
            lines=lines.split("8. Missing Attribute Values:")[0]
            l = re.split('. | : ', lines)
            regex = re.compile("^[A-Za-z-?]{1,}$")
            cols += [ s for s in l if regex.match(s)]
        data.columns = cols
        data.drop("stalk-root", axis=1, inplace=True)
        label = data.poisonous
        var = data.drop("poisonous", axis=1)
        # encode
        enc = OrdinalEncoder()
        X = enc.fit_transform(var)
        y = enc.fit_transform(np.array(label).reshape(-1, 1)).flatten()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        
        if self.split == 'train':
            self._x = torch.tensor(X_train, dtype=torch.float32)
            self._y = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        elif self.split == 'val':
            self._x = torch.tensor(X_test, dtype=torch.float32)
            self._y = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
           

    def __len__(self):
    # a DataSet must know it size
        return self._x.shape[0]

    def __getitem__(self, index):
        x = self._x[index, :]
        y = self._y[index, :]
        return x, y

    def configure_dataloader(self):
        self.dataloader = DataLoader(self, batch_size=64, shuffle=True)
