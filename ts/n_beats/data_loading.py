import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_file(file_location, sampling=False, sample_size=5):
    series = []
    ids = dict()
    with open(file_location, "r") as file:
        data = file.read().split("\n")

    for i in range(1, len(data) - 1):
        row = data[i].replace('"', '').split(',')
        ts_values = np.array([float(j) for j in row[1:] if j != ""])
        series.append(ts_values)
        ids[row[0]] = i - 1
        if sampling and i == sample_size:
            series = np.asarray(series)
            return series, ids

    series = np.asarray(series)
    return series, ids


def create_val_set(train, output_size):
    val = []
    new_train = []
    for i in range(len(train)):
        val.append(train[i][-output_size:])
        new_train.append(train[i][:-output_size])
    return np.array(new_train), np.array(val)


def create_datasets(train_file_location, test_file_location, output_size, sample=False, sampling_size=5):
    train, train_idx = read_file(train_file_location, sample, sampling_size)
    test, test_idx = read_file(test_file_location, sample, sampling_size)
    train, val = create_val_set(train, output_size)
    if sample:
        print("Sampling train data for {}".format(train_idx.keys()))
        print("Sampling test data for {}".format(test_idx.keys()))

    return train, train_idx, val, test, test_idx


def chop_series(train, chop_val):
    # CREATE MASK FOR VALUES TO BE CHOPPED
    train_len_mask = [True if len(i) >= chop_val else False for i in train]
    # FILTER AND CHOP TRAIN
    train = [train[i][-chop_val:] for i in range(len(train)) if train_len_mask[i]]
    return train, train_len_mask


def determine_chop_value(data, backcast_length, forecast_length):
    ts_lengths = []
    for i in range(len(data)):
        ts = data[i]
        length = len(ts) - (forecast_length + backcast_length)
        if length > 0:
            ts_lengths.append(len(ts))
        # print(len(ts), length)
    if ts_lengths:
        return np.amin(np.array(ts_lengths)).astype(dtype=int)
    return -1
    # return np.quantile(ts_lengths, 0.8).astype(dtype=int)


class SeriesDataset(Dataset):

    def __init__(self, info, variable, sample, data_train, train_idx, data_val, data_test, backcast_length,
                 foreccast_length, device):
        chop_val = determine_chop_value(data_train, backcast_length, foreccast_length)
        data_train, mask = chop_series(data_train, chop_val)
        if sample:
            info = info[(info["M4id"].isin(train_idx.keys())) & (info["SP"] == variable)]
        self.dataInfoCatOHE = pd.get_dummies(info[info["SP"] == variable]["category"])
        self.dataInfoCatHeaders = np.array([i for i in self.dataInfoCatOHE.columns.values])
        self.dataInfoCat = torch.from_numpy(self.dataInfoCatOHE[mask].values).float()
        self.dataTrain = [torch.tensor(data_train[i], dtype=torch.float32) for i in range(len(data_train))]
        self.dataVal = [torch.tensor(data_val[i], dtype=torch.float32) for i in range(len(data_val)) if mask[i]]
        self.dataTest = [torch.tensor(data_test[i], dtype=torch.float32) for i in range(len(data_test)) if mask[i]]

        self.device = device

    def __len__(self):
        return len(self.dataTrain)

    def __getitem__(self, idx):
        return self.dataTrain[idx].to(self.device), \
               self.dataVal[idx].to(self.device), \
               self.dataTest[idx].to(self.device), \
               self.dataInfoCat[idx].to(self.device), \
               idx


class DatasetTS(Dataset):
    """ Data Set Utility for Time Series.

        Args:
            - time_series(numpy 1d array) - array with univariate time series
            - forecast_length(int) - length of forecast window
            - backcast_length(int) - length of backcast window
            - sliding_window_coef(int) - determines how much to adjust sliding window
                by when determining forecast backcast pairs:
                    if sliding_window_coef = 1, this will make it so that backcast
                    windows will be sampled that don't overlap.
                    If sliding_window_coef=2, then backcast windows will overlap
                    by 1/2 of their data. This creates a dataset with more training
                    samples, but can potentially lead to overfitting.
    """

    def __init__(self, time_series, backcast_length, forecast_length, sliding_window_coef=1):
        self.data = time_series
        self.forecast_length, self.backcast_length = forecast_length, backcast_length
        self.sliding_window_coef = sliding_window_coef
        self.sliding_window = int(np.ceil(self.backcast_length / sliding_window_coef))

    def __len__(self):
        """ Return the number of backcast/forecast pairs in the dataset.
        """
        length = int(np.floor((len(self.data) - (self.forecast_length + self.backcast_length)) / self.sliding_window))
        return length

    def __getitem__(self, index):
        """Get a single forecast/backcast pair by index.

            Args:
                index(int) - index of forecast/backcast pair
            raise exception if the index is greater than DatasetTS.__len__()
        """
        if (index > self.__len__()):
            raise IndexError("Index out of Bounds")
        # index = index * self.backcast_length
        index = index * self.sliding_window
        print("Index={}".format(index))
        if index + self.backcast_length:
            backcast_model_input = self.data[index:index + self.backcast_length]
        else:
            backcast_model_input = self.data[index:]
        forecast_actuals_idx = index + self.backcast_length
        forecast_actuals_output = self.data[forecast_actuals_idx:
                                            forecast_actuals_idx + self.forecast_length]
        forecast_actuals_output = np.array(forecast_actuals_output, dtype=np.float32)
        backcast_model_input = np.array(backcast_model_input, dtype=np.float32)
        return backcast_model_input, forecast_actuals_output


def collate_lines(seq_list):
    train_, val_, test_, idx_ = zip(*seq_list)
    train_lens = [len(seq) for seq in train_]
    seq_order = sorted(range(len(train_lens)), key=train_lens.__getitem__, reverse=True)
    train = [train_[i] for i in seq_order]
    val = [val_[i] for i in seq_order]
    test = [test_[i] for i in seq_order]
    idx = [idx_[i] for i in seq_order]
    return train, val, test, idx
