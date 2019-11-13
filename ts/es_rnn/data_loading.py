import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ts.utils.helper_funcs import shuffled_arrays

def read_file(file_location, sampling=False, sample_size=5):
    series = []
    ids = dict()
    with open(file_location, "r") as file:
        data = file.read().split("\n")

    for i in range(1, len(data) - 1):
        row = data[i].replace('"', "").split(",")
        series.append(np.array([float(j) for j in row[1:] if j != ""]))
        ids[row[0]] = i - 1
        if sampling and i == sample_size:
            series = np.array(series)
            return series, ids

    series = np.array(series)
    return series, ids


def create_val_set(train, output_size):
    val = []
    new_train = []
    for i in range(len(train)):
        val.append(train[i][-output_size:])
        new_train.append(train[i][:-output_size])
    return np.array(new_train), np.array(val)


def chop_series(train, chop_val):
    # CREATE MASK FOR VALUES TO BE CHOPPED
    train_len_mask = [True if len(i) >= chop_val else False for i in train]
    # FILTER AND CHOP TRAIN
    train = [train[i][-chop_val:] for i in range(len(train)) if train_len_mask[i]]
    return train, train_len_mask


def create_datasets(train_file_location, test_file_location, output_size, sample=False, sampling_size=5):
    train, train_idx = read_file(train_file_location, sample, sampling_size)
    #train, train_idx,p = shuffled_arrays(train, train_idx)

    test, test_idx = read_file(test_file_location, sample, sampling_size)
    #test = test[p], test_idx = test_idx[p]

    train, val = create_val_set(train, output_size)
    if sample:
        print("Sampling train data for {}".format(train_idx.keys()))
        print("Sampling test data for {}".format(test_idx.keys()))
    return train, train_idx, val, test, test_idx


class SeriesDataset(Dataset):

    def __init__(self, dataTrain, dataVal, dataTest, info, variable, chop_value, device, ts_labels, sample=False):
        dataTrain, mask = chop_series(dataTrain, chop_value)
        if sample:
            info = info[(info["M4id"].isin(ts_labels.keys())) & (info["SP"] == variable)]
        self.dataInfoCatOHE = pd.get_dummies(info[info["SP"] == variable]["category"])
        self.dataInfoCatHeaders = np.array([i for i in self.dataInfoCatOHE.columns.values])
        self.dataInfoCat = torch.from_numpy(self.dataInfoCatOHE[mask].values).float()
        self.dataTrain = [torch.tensor(dataTrain[i], dtype=torch.float32) for i in range(len(dataTrain))]  # ALREADY MASKED IN CHOP FUNCTION
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


def collate_lines(seq_list):
    train_, val_, test_, info_cat_, idx_ = zip(*seq_list)
    train_lens = [len(seq) for seq in train_]
    seq_order = sorted(range(len(train_lens)), key=train_lens.__getitem__, reverse=True)
    train = [train_[i] for i in seq_order]
    val = [val_[i] for i in seq_order]
    test = [test_[i] for i in seq_order]
    info_cat = [info_cat_[i] for i in seq_order]
    idx = [idx_[i] for i in seq_order]
    return train, val, test, info_cat, idx
