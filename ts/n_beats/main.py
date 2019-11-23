import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ts.n_beats.config import get_config
from ts.utils.data_loading import SeriesDataset
from ts.n_beats.model import NBeatsNet
from ts.n_beats.trainer import Trainer
from ts.utils.helper_funcs import MODEL_TYPE, set_seed, create_datasets, determine_chop_value, filter_timeseries, \
    generate_timeseries_length_stats
from ts.utils.loss_modules import PinballLoss


def main():
    set_seed(0)

    run_id = str(int(time.time()))
    print("Starting run={}, model={} ".format(run_id, MODEL_TYPE.NBEATS.value))

    BASE_DIR = Path("data/raw/")
    LOG_DIR = Path("logs/" + MODEL_TYPE.NBEATS.value)
    FIGURE_PATH = Path("figures-temp/" + MODEL_TYPE.NBEATS.value)

    print("Loading config")
    config = get_config("Yearly")
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
    model = NBeatsNet(stack_types=config["stack_types"],
                      forecast_length=forecast_length,
                      thetas_dims=config["thetas_dims"],
                      nb_blocks_per_stack=config["nb_blocks_per_stack"],
                      backcast_length=backcast_length,
                      hidden_layer_units=config["hidden_layer_units"],
                      share_weights_in_stack=config["share_weights_in_stack"],
                      device=config["device"])
    reload = config["reload"]
    add_run_id = config["add_run_id"]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = PinballLoss(config["training_tau"], config["output_size"] * config["batch_size"], config["device"])
    trainer = Trainer(MODEL_TYPE.NBEATS.value, model, optimizer, criterion, dataloader, run_id, add_run_id, config,
                      forecast_length, backcast_length,
                      ohe_headers=dataset.data_info_cat_headers, csv_path=LOG_DIR, figure_path=FIGURE_PATH,
                      sampling=sample, reload=reload)
    trainer.train_epochs()


if __name__ == "__main__":
    main()
