import os
import time
from pathlib import Path

import pandas as pd
import torch
from torch.nn import SmoothL1Loss
from torch.utils.data import DataLoader

from ts.es_rnn.config import get_config
from ts.es_rnn.model import ESRNN
from ts.es_rnn.trainer import ESRNNTrainer
from ts.utils.data_loading import SeriesDataset
from ts.utils.helper_funcs import MODEL_TYPE, set_seed, create_datasets, generate_timeseries_length_stats, \
    filter_timeseries

set_seed(0)

run_id = str(int(time.time()))
print("Starting run={}, model={} ".format(run_id, MODEL_TYPE.ESRNN.value))

try:
    user_paths = os.environ["PYTHONPATH"].split(os.pathsep)
    print(user_paths)
except KeyError:
    user_paths = []

BASE_DIR = Path("data/raw/")
LOG_DIR = Path("logs/" + MODEL_TYPE.ESRNN.value)
FIGURE_PATH = Path("figures-temp/" + MODEL_TYPE.ESRNN.value)

print("loading config")
config = get_config("Monthly")
print("Frequency:{}".format(config["variable"]))

print("loading data")
info = pd.read_csv(str(BASE_DIR / "M4info.csv"))

train_path = str(BASE_DIR / "train/%s-train.csv") % (config["variable"])
test_path = str(BASE_DIR / "test/%s-test.csv") % (config["variable"])

sample = config["sample"]
sample_ids = config["sample_ids"] if "sample_ids" in config else []
train, ts_labels, val, test, test_idx = create_datasets(train_path, test_path, config["output_size"],
                                                        sample_ids=sample_ids, sample=sample,
                                                        sampling_size=4)
generate_timeseries_length_stats(train)
print("#.Train before chopping:{}".format(train.shape[0]))
train_before_chopping_count = train.shape[0]
chop_val = config["chop_val"]
print("Chop value:{:6.3f}".format(chop_val))
train, val, test, data_infocat_ohe, data_infocat_headers, data_info_cat = \
    filter_timeseries(info, config["variable"], sample, ts_labels, train, chop_val, val, test)
print("#.Train after chopping:{}, lost:{:5.2f}%".format(len(train),
                                                        (train_before_chopping_count - len(
                                                            train)) / train_before_chopping_count * 100.))
print("#.train:{}, #.validation ts:{}, #.test ts:{}".format(len(train), len(val), len(test)))

dataset = SeriesDataset(data_infocat_ohe, data_infocat_headers, data_info_cat, ts_labels,
                        train, val, test, config["device"])

config["num_of_categories"] = len(dataset.data_info_cat_headers)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

reload = config["reload"]
model = ESRNN(num_series=len(dataset), config=config)
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
add_run_id = config["add_run_id"]
# criterion = PinballLoss(config["training_tau"], config["output_size"] * config["batch_size"], config["device"])
criterion = SmoothL1Loss()
tr = ESRNNTrainer(MODEL_TYPE.ESRNN.value, model, optimizer, criterion, dataloader, run_id, add_run_id, config,
                  ohe_headers=dataset.data_info_cat_headers,
                  csv_path=LOG_DIR,
                  figure_path=FIGURE_PATH, sampling=sample, reload=reload)
tr.train_epochs()
