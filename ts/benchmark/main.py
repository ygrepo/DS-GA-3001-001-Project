import time
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from ts.n_beats.config import get_config
from ts.utils.data_loading import SeriesDataset
from ts.n_beats.model import NBeatsNet
from ts.n_beats.trainer import Trainer
from ts.utils.helper_funcs import MODEL_TYPE, set_seed, create_datasets


def main():
    set_seed(0)

    run_id = str(int(time.time()))
    print("Starting run={}, model={} ".format(run_id, MODEL_TYPE.BENCHMARK.value))

    BASE_DIR = Path("data/raw/")
    LOG_DIR = Path("logs/" + MODEL_TYPE.BENCHMARK.value)
    FIGURE_PATH = Path("figures-temp/" + MODEL_TYPE.BENCHMARK.value)

    print("Loading config")
    config = get_config("Monthly")
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
    print("#.train:{}, #.validation ts:{}, #.test ts:{}".format(len(train), len(val), len(test)))



if __name__ == "__main__":
    main()
