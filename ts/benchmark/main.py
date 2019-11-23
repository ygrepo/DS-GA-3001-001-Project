import time
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from ts.n_beats.config import get_config
from ts.utils.data_loading import SeriesDataset
from ts.n_beats.model import NBeatsNet
from ts.n_beats.trainer import Trainer
from ts.utils.helper_funcs import BENCHMARK_MODEL_NAME, set_seed, create_datasets


def main():
    set_seed(0)

    run_id = str(int(time.time()))
    print("Starting run={}, model={} ".format(run_id, BENCHMARK_MODEL_NAME))

    BASE_DIR = Path("data/raw/")
    LOG_DIR = Path("logs/" + BENCHMARK_MODEL_NAME)
    FIGURE_PATH = Path("figures-temp/" + BENCHMARK_MODEL_NAME)

    print("Loading config")
    config = get_config("Daily")
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
    print("#of train ts:{}, dimensions of validation ts:{}, dimensions of test ts:{}".format(train.shape, val.shape,
                                                                                             test.shape))

    dataset = SeriesDataset(info, config["variable"], sample, train, ts_labels, val, test, backcast_length,
                            forecast_length,
                            config["device"])
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
    trainer = Trainer(NBEATS_MODEL_NAME.name, model, dataloader, run_id, add_run_id, config, forecast_length, backcast_length,
                      ohe_headers=dataset.data_info_cat_headers, csv_path=LOG_DIR, figure_path=FIGURE_PATH,
                      sampling=sample, reload=reload)
    trainer.train_epochs()


if __name__ == "__main__":
    main()
