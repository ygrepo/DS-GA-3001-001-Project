import numpy as np
import torch
from torch.utils.data import Dataset


class SeriesDataset(Dataset):

    def __init__(self, data_info_cat_ohe, data_info_cat_headers, data_info_cat, ts_labels, data_train, data_val,
                 data_test, device):
        self.data_info_cat_ohe = data_info_cat_ohe
        self.data_info_cat_headers = data_info_cat_headers
        self.data_info_cat = data_info_cat
        self.data_train = [torch.tensor(data_train[i], dtype=torch.float32) for i in range(len(data_train))]
        self.data_val = [torch.tensor(data_val[i], dtype=torch.float32) for i in range(len(data_val))]
        self.data_test = [torch.tensor(data_test[i], dtype=torch.float32) for i in range(len(data_test))]
        self.ts_labels = dict([reversed(i) for i in ts_labels.items()])

        self.device = device

    def __len__(self):
        return len(self.data_train)

    def __getitem__(self, idx):
        return self.data_train[idx].to(self.device), \
               self.data_val[idx].to(self.device), \
               self.data_test[idx].to(self.device), \
               self.data_info_cat[idx].to(self.device), \
               self.ts_labels[idx], \
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
