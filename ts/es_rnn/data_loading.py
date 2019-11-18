import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ts.utils.helper_funcs import chop_series


class SeriesDataset(Dataset):

    def __init__(self, dataTrain, dataVal, dataTest, info, variable, chop_value, device, ts_labels, sample=False):
        dataTrain, mask = chop_series(dataTrain, chop_value)
        if sample:
            info = info[(info["M4id"].isin(ts_labels.keys())) & (info["SP"] == variable)]
        self.dataInfoCatOHE = pd.get_dummies(info[info["SP"] == variable]["category"])
        self.dataInfoCatHeaders = np.array([i for i in self.dataInfoCatOHE.columns.values])
        self.dataInfoCat = torch.from_numpy(self.dataInfoCatOHE[mask].values).float()
        self.dataTrain = [torch.tensor(dataTrain[i], dtype=torch.float32) for i in
                          range(len(dataTrain))]  # ALREADY MASKED IN CHOP FUNCTION
        self.dataVal = [torch.tensor(dataVal[i], dtype=torch.float32) for i in range(len(dataVal)) if mask[i]]
        self.dataTest = [torch.tensor(dataTest[i], dtype=torch.float32) for i in range(len(dataTest)) if mask[i]]
        self.ts_labels = dict([reversed(i) for i in ts_labels.items()])
        self.device = device

    def __len__(self):
        return len(self.dataTrain)

    def __getitem__(self, idx):
        return self.dataTrain[idx].to(self.device), \
               self.dataVal[idx].to(self.device), \
               self.dataTest[idx].to(self.device), \
               self.dataInfoCat[idx].to(self.device), \
               self.ts_labels[idx], \
               idx
