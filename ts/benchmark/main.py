import time
from pathlib import Path

import gpytorch
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ts.benchmark.config import get_config
from ts.benchmark.model import SpectralMixtureGPModel
from ts.benchmark.trainer import Trainer
from ts.utils.data_loading import SeriesDataset
from ts.utils.helper_funcs import MODEL_TYPE, set_seed, create_datasets, generate_timeseries_length_stats, \
    filter_timeseries, determine_chop_value
from ts.utils.loss_modules import PinballLoss


def main():
    set_seed(0)

    run_id = str(int(time.time()))
    print("Starting run={}, model={} ".format(run_id, MODEL_TYPE.BENCHMARK.value))

    BASE_DIR = Path("data/raw/")
    LOG_DIR = Path("logs/" + MODEL_TYPE.BENCHMARK.value)
    FIGURE_PATH = Path("figures-temp/" + MODEL_TYPE.BENCHMARK.value)

    print("Loading config")
    config = get_config("Daily")
    print("Frequency:{}".format(config["variable"]))
    forecast_length = config["output_size"]
    backcast_length = 1 * forecast_length

    print("loading data")
    info = pd.read_csv(str(BASE_DIR / "M4info.csv"))
    train_path = str(BASE_DIR / "train/%s-train.csv") % (config["variable"])
    test_path = str(BASE_DIR / "test/%s-test.csv") % (config["variable"])

    sample = config["sample"]
    sample_ids = config["sample_ids"] if "sample_ids" in config else []
    train, ts_labels, val, test, test_idx = create_datasets(train_path, test_path, config["output_size"],
                                                            create_val_dataset=True,
                                                            sample_ids=sample_ids, sample=sample,
                                                            sampling_size=4)
    generate_timeseries_length_stats(train)
    print("#.Train before chopping:{}".format(train.shape[0]))
    train_before_chopping_count = train.shape[0]
    chop_val = determine_chop_value(train, backcast_length, forecast_length)
    print("Chop value:{:6.3f}".format(chop_val))
    train, val, test, data_infocat_ohe, data_infocat_headers, data_info_cat = \
        filter_timeseries(info, config["variable"], sample, ts_labels, train, chop_val, val, test)
    print("#.Train after chopping:{}, lost:{:5.2f}%".format(len(train),
                                                            (train_before_chopping_count - len(
                                                                train)) / train_before_chopping_count * 100.))
    print("#.train:{}, #.validation ts:{}, #.test ts:{}".format(len(train), len(val), len(test)))

    dataset = SeriesDataset(data_infocat_ohe, data_infocat_headers, data_info_cat, ts_labels,
                            train, val, test, config["device"])

    # dataloader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collate_lines, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    reload = config["reload"]
    add_run_id = config["add_run_id"]
    criterion = PinballLoss(config["training_tau"], config["output_size"] * config["batch_size"], config["device"])
    trainer = Trainer(MODEL_TYPE.BENCHMARK.value, None, None, criterion, dataloader, run_id, add_run_id, config,
                      forecast_length, backcast_length,
                      ohe_headers=dataset.data_info_cat_headers, csv_path=LOG_DIR, figure_path=FIGURE_PATH,
                      sampling=sample, reload=reload)
    trainer.train_epochs()


if __name__ == "__main__":
    # Training data is 11 points in [0,1] inclusive regularly spaced
    # train_x = torch.linspace(0, 1, 100).view(1, -1, 1).repeat(4, 1, 1)
    # # True function is sin(2*pi*x) with Gaussian noise
    # import math
    # sin_y = torch.sin(train_x[0] * (2 * math.pi)) + 0.5 * torch.rand(1, 100, 1)
    # sin_y_short = torch.sin(train_x[0] * (math.pi)) + 0.5 * torch.rand(1, 100, 1)
    # cos_y = torch.cos(train_x[0] * (2 * math.pi)) + 0.5 * torch.rand(1, 100, 1)
    # cos_y_short = torch.cos(train_x[0] * (math.pi)) + 0.5 * torch.rand(1, 100, 1)
    # train_y = torch.cat((sin_y, sin_y_short, cos_y, cos_y_short)).squeeze(-1)
    #
    # print(train_x.shape, train_y.shape)
    main()
