import os
import time
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from ts.es_rnn.config import get_config
from ts.es_rnn.data_loading import create_datasets, SeriesDataset
from ts.es_rnn.model import ESRNN
from ts.es_rnn.trainer import ESRNNTrainer

print("Start")
try:
    user_paths = os.environ["PYTHONPATH"].split(os.pathsep)
    print(user_paths)
except KeyError:
    user_paths = []

BASE_DIR = Path("data/raw/")
LOG_DIR = Path("logs/esrnn")
print("loading config")
# config = get_config("Quarterly")
config = get_config("Monthly")

print("loading data")
info = pd.read_csv(str(BASE_DIR / "M4info.csv"))

train_path = str(BASE_DIR / "train/%s-train.csv") % (config["variable"])
test_path = str(BASE_DIR / "test/%s-test.csv") % (config["variable"])

sample = True
train, train_idx, val, test, test_idx = create_datasets(train_path, test_path, config["output_size"], sample=sample)
print(train.shape, val.shape, test.shape)

dataset = SeriesDataset(train, val, test, info, config["variable"], config["chop_val"], config["device"],
                        train_idx, sample)
config["num_of_categories"] = len(dataset.dataInfoCatHeaders)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

run_id = str(int(time.time()))
model = ESRNN(num_series=len(dataset), config=config)
tr = ESRNNTrainer(model, dataloader, run_id, config, ohe_headers=dataset.dataInfoCatHeaders, csv_path=LOG_DIR)
tr.train_epochs()
