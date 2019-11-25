import cmath
import random
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


class MODEL_TYPE(Enum):
    NBEATS = "nbeats"
    ESRNN = "esrnn"
    BENCHMARK = "benchmark"


class SAVE_LOAD_TYPE(Enum):
    NO_ACTION = "NONE"
    MODEL = "MODEL"
    MODEL_PARAMETERS = "MODEL_PARAMETERS"


class BLOCK_TYPE(Enum):
    SEASONALITY = "SEASONALITY"
    TREND = "TREND"
    GENERAL = "GENERAL"


def colwise_batch_mask(target_shape_tuple, target_lens):
    # takes in (seq_len, B) shape and returns mask of same shape with ones up to the target lens
    mask = torch.zeros(target_shape_tuple)
    for i in range(target_shape_tuple[1]):
        mask[:target_lens[i], i] = 1
    return mask


def rowwise_batch_mask(target_shape_tuple, target_lens):
    # takes in (B, seq_len) shape and returns mask of same shape with ones up to the target lens
    mask = torch.zeros(target_shape_tuple)
    for i in range(target_shape_tuple[0]):
        mask[i, :target_lens[i]] = 1
    return mask


def unpad_sequence(padded_sequence, lens):
    seqs = []
    for i in range(padded_sequence.size(1)):
        seqs.append(padded_sequence[:lens[i], i])
    return seqs


def save_model(file_path, model, optimizer, run_id, add_run_id=False):
    file_path.mkdir(parents=True, exist_ok=True)
    if add_run_id:
        model_path = file_path / ("model_" + run_id + ".pyt")
    else:
        model_path = file_path / ("model.pyt")

    torch.save(model, model_path)
    print("Saving whole model and optimizer state dictionary")
    torch.save({
        "model": model,
        "optimizer_state_dict": optimizer.state_dict(),
    }, model_path)


def load_model(file_path, config):
    model_path = file_path / "model.pyt"
    if model_path.exists():
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = "cpu"
        checkpoint = torch.load(model_path, map_location=map_location)
        model = checkpoint["model"]
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Restored checkpoint(whole model and optimizer state dictionary) from {model_path}.")
        return model, optimizer


def save_model_parameters(file_path, model, optimiser, run_id, add_run_id=False):
    file_path.mkdir(parents=True, exist_ok=True)
    if add_run_id:
        model_path = file_path / ("model_" + run_id + ".pyt")
    else:
        model_path = file_path / ("model.pyt")

    print("Saving mode and optimizer state dictionaries")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimiser.state_dict(),
    }, model_path)


def load_model_parameters(file_path, model, optimiser):
    model_path = file_path / "model.pyt"
    if model_path.exists():
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = "cpu"
        checkpoint = torch.load(model_path, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Restored checkpoint(state dictionaries) from {model_path}.")


def read_file(file_location, sample_ids, sampling=False, sample_size=5):
    series = []
    ids = dict()
    with open(file_location, "r") as file:
        data = file.read().split("\n")

    for i in range(1, len(data) - 1):
        row = data[i].replace('"', '').split(',')
        ts_values = np.array([float(j) for j in row[1:] if j != ""])
        series.append(ts_values)
        ids[row[0]] = i - 1
        if sampling and not sample_ids and i == sample_size:
            series = np.asarray(series)
            return series, ids

    series = np.asarray(series)
    return series, ids


def filter_sample_ids(ts, ts_idx, sample_ids):
    sampled_ts = []
    ids = dict()
    for i, id in enumerate(sample_ids):
        if id not in ts_idx:
            print("Could not find ts id:{}".format(id))
            continue
        sampled_ts.append(ts[ts_idx[id]].tolist())
        ids[id] = i
    return np.array(sampled_ts), ids


def create_val_set(train, output_size):
    val = []
    new_train = []
    for i in range(len(train)):
        val.append(train[i][-output_size:])
        new_train.append(train[i][:-output_size])
    return np.array(new_train), np.array(val)


def create_datasets(train_file_location, test_file_location, output_size,
                    create_val_dataset=True,
                    sample_ids=[], sample=False, sampling_size=5):
    train, train_idx = read_file(train_file_location, sample_ids, sample, sampling_size)
    if sample and sample_ids:
        train, train_idx = filter_sample_ids(train, train_idx, sample_ids)
    print("train:{}".format(len(train)))

    test, test_idx = read_file(test_file_location, sample_ids, sample, sampling_size)
    if sample and sample_ids:
        test, test_idx = filter_sample_ids(test, test_idx, sample_ids)

    if create_val_dataset:
        train, val = create_val_set(train, output_size)
    else:
        val = None

    if sample and sample_ids:
        print("Sampling train data for {}".format(sample_ids))
        print("Sampling test data for {}".format(sample_ids))

    elif sample and not sample_ids:
        print("Sampling train data for {}".format(train_idx.keys()))
        print("Sampling test data for {}".format(test_idx.keys()))

    return train, train_idx, val, test, test_idx


def generate_timeseries_length_stats(data):
    df = pd.DataFrame({"data": data.tolist()})
    df["size"] = [len(x) for x in data]
    df.drop(["data"], axis=1, inplace=True)
    print(df.describe())


def determine_chop_value(data, backcast_length, forecast_length):
    ts_lengths = []
    for i in range(len(data)):
        ts = data[i]
        length = len(ts) - (forecast_length + backcast_length)
        if length > 0:
            ts_lengths.append(len(ts))
        # print(len(ts), length)
    if ts_lengths:
        # return np.quantile(ts_lengths, 0.25).astype(dtype=int)
        return np.amin(np.array(ts_lengths)).astype(dtype=int)
    return -1


def chop_series(train, chop_val):
    # CREATE MASK FOR VALUES TO BE CHOPPED
    train_len_mask = [True if len(i) >= chop_val else False for i in train]
    # FILTER AND CHOP TRAIN
    train = [train[i][-chop_val:] for i in range(len(train)) if train_len_mask[i]]
    return train, train_len_mask


def filter_timeseries(info, variable, sample, ts_labels, data_train, chop_val, data_val, data_test):
    data_train, mask = chop_series(data_train, chop_val)
    if sample:
        info = info[(info["M4id"].isin(ts_labels.keys())) & (info["SP"] == variable)]
    data_train = [data_train[i] for i in range(len(data_train))]
    data_val = [data_val[i] for i in range(len(data_val)) if mask[i]]
    data_test = [data_test[i] for i in range(len(data_test)) if mask[i]]
    data_infocat_ohe = pd.get_dummies(info[info["SP"] == variable]["category"])
    data_infocat_headers = np.array([i for i in data_infocat_ohe.columns.values])
    data_info_cat = torch.from_numpy(data_infocat_ohe[mask].values).float()
    return data_train, data_val, data_test, data_infocat_ohe, data_infocat_headers, data_info_cat


# Reproducibility
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def shuffled_arrays(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], p


def plot_stacks(run_id, path, model):
    path.mkdir(parents=True, exist_ok=True)
    num_stacks = len(model.stacks)
    num_blocks = len(model.stacks[0])
    fig, axes = plt.subplots(num_blocks, num_stacks, figsize=(10, 12), sharey=False)
    plt.subplots_adjust(hspace=.4)

    for stack_id in range(num_stacks):
        stack = model.stacks[stack_id]
        for block_id in range(len(stack)):
            block = stack[block_id]
            ax = axes[block_id][stack_id]
            plot_block_ts(ax, block)
            ax.set_xlabel("Time")
            ax.set_ylabel("Observations")
            b_legend_str = ("backcast-{}-{}".format(block.block_type, block.id))
            f_legend_str = ("forecast-{}-{}".format(block.block_type, block.id))
            ax.legend([b_legend_str, f_legend_str], loc="best")

    plt.savefig(path / ("stack_" + run_id + ".png"))

    plt.tight_layout()
    sns.despine()
    plt.show()


def plot_block_ts(ax, block):
    backcasts = (block.backcasts[-1]).squeeze().cpu().detach().numpy()
    forecasts = (block.forecasts[-1]).squeeze().cpu().detach().numpy()
    ts_range = backcasts.shape[1]
    num_segments = len(backcasts)
    start = 0
    stop = ts_range
    x_values = []
    y_backcast_values = []
    y_forecast_values = []
    num_segments = min(10, num_segments)
    for i in range(num_segments):
        x = np.arange(start, stop)
        x_values.extend(x.tolist())
        y = backcasts[i, :]
        y_backcast_values.extend(y.tolist())
        y_forecast_values.extend(forecasts[i, :].tolist())
        start += ts_range
        stop += ts_range
    backcast_color = "b-" if block.block_type == BLOCK_TYPE.GENERAL or block.block_type == BLOCK_TYPE.TREND else "r-"
    forecast_color = "b--" if block.block_type == BLOCK_TYPE.GENERAL or block.block_type == BLOCK_TYPE.TREND else "r--"
    x_values = np.array(x_values)
    y_backcast_values = np.array(y_backcast_values[:-ts_range])
    y_forecast_values = np.array(y_forecast_values[ts_range:])
    ax.plot(x_values[:-ts_range], y_backcast_values, backcast_color, x_values[ts_range:], y_forecast_values,
            forecast_color)
    ax.set_autoscalex_on(True)
    ax.set_autoscaley_on(True)


def plot_ts(original_ts, predicted_ts, ts_labels, cats, path, number_to_plot=1):
    path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, number_to_plot, figsize=(17, 4))
    plt.subplots_adjust(hspace=.3)

    fig_label_iter = iter(cats)
    for i in range(number_to_plot):
        if number_to_plot == 1:
            ax = axes
        else:
            ax = axes[i]
        x = np.arange(len(original_ts[i, :]))
        y = original_ts[i, :]
        y_pred = predicted_ts[i, :]
        ax.plot(x, y_pred, "r-", x, y, "b-")
        ax.set_xlabel("Time")
        ax.set_ylabel("Observations")
        ts_label = next(fig_label_iter)
        ax.set_title(ts_label + " time Series:" + ts_labels[i])
        ax.legend(("predicted", "original"))
    plt.savefig(path / "time_series.png")
    plt.tight_layout()
    sns.despine()
    plt.show()


def plot_levels_seasonalities(train, levels, seasonalities, path=None):
    path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(17, 4))
    x_train = np.arange(train.shape[1])
    x_levels = np.arange(levels.shape[1])
    ax[0].plot(x_train, train[0, :], "b-", x_levels, levels[0, :], "r-")
    x_max = max(np.amax(x_train), np.amax(x_levels))
    ax[0].set_xlim([0, x_max + 1])
    ymin1 = torch.min(train).item()
    ymax1 = torch.max(train).item()
    ymin2 = torch.min(levels).item()
    ymax2 = torch.max(levels).item()
    ymin = min(ymin1, ymin2)
    ymax = max(ymax1, ymax2)
    ax[0].set_ylim([ymin, ymax])
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Observations")
    ax[0].set_title("Levels")
    ax[0].legend(("original", "levels"))
    x_seasonalities = np.arange(seasonalities.shape[1])
    ax[1].plot(x_seasonalities, seasonalities[0, :], "r-")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Observations")
    ax[1].set_title("Seasonality coefficients")
    ymin = torch.min(seasonalities).item()
    ymax = torch.max(seasonalities).item()
    ax[1].set_xlim([0, np.amax(x_seasonalities) + 1])
    ax[1].set_ylim([ymin, ymax])
    plt.savefig(path / "levels_seasonalities.png")
    plt.tight_layout()
    sns.despine()
    plt.show()


def plot_windows(window_input, window_output, path=None):
    path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(17, 4))
    backcasts = window_input.squeeze(1).cpu().detach().numpy()
    forecasts = window_output.squeeze(1).cpu().detach().numpy()
    y_backcast_values = []
    for i in range(backcasts.shape[0]):
        y_backcast_values.extend(backcasts[i, :].tolist())
    y_forecast_values = []
    for i in range(forecasts.shape[0]):
        y_forecast_values.extend(forecasts[i, :].tolist())

    x_values = np.arange(len(y_backcast_values))
    ax[0].plot(x_values, np.array(y_backcast_values), "b-")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Observations")
    ax[0].set_title("Backcast: normalization and deseasonalization")
    x_values = np.arange(len(y_forecast_values))
    ax[1].set_title("Forecast: normalization and deseasonalization")
    ax[1].plot(x_values, np.array(y_forecast_values), "r-")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Observations")
    plt.savefig(path / "windows.png")
    plt.tight_layout()
    sns.despine()
    plt.show()


def isclose(a,
            b,
            rel_tol=1e-9,
            abs_tol=0.0,
            method="weak"):
    """
    returns True if a is close in value to b. False otherwise

    :param a: one of the values to be tested

    :param b: the other value to be tested

    :param rel_tol=1e-8: The relative tolerance -- the amount of error
                         allowed, relative to the magnitude of the input
                         values.

    :param abs_tol=0.0: The minimum absolute tolerance level -- useful for
                        comparisons to zero.

    :param method: The method to use. options are:
                  "asymmetric" : the b value is used for scaling the tolerance
                  "strong" : The tolerance is scaled by the smaller of
                             the two values
                  "weak" : The tolerance is scaled by the larger of
                           the two values
                  "average" : The tolerance is scaled by the average of
                              the two values.

    NOTES:

    -inf, inf and NaN behave similar to the IEEE 754 standard. That
    -is, NaN is not close to anything, even itself. inf and -inf are
    -only close to themselves.

    Complex values are compared based on their absolute value.

    The function can be used with Decimal types, if the tolerance(s) are
    specified as Decimals::

      isclose(a, b, rel_tol=Decimal('1e-9'))

    See PEP-0485 for a detailed description

    """
    if method not in ("asymmetric", "strong", "weak", "average"):
        raise ValueError('method must be one of: "asymmetric",'
                         ' "strong", "weak", "average"')

    if rel_tol < 0.0 or abs_tol < 0.0:
        raise ValueError('error tolerances must be non-negative')

    if a == b:  # short-circuit exact equality
        return True
    # use cmath so it will work with complex or float
    if cmath.isinf(a) or cmath.isinf(b):
        # This includes the case of two infinities of opposite sign, or
        # one infinity and one finite number. Two infinities of opposite sign
        # would otherwise have an infinite relative tolerance.
        return False
    diff = abs(b - a)
    print("Diff:{:8.5f}-{:8.5f}-{:8.5f}".format(diff, abs(rel_tol * b), abs(rel_tol * a)))
    if method == "asymmetric":
        return (diff <= abs(rel_tol * b)) or (diff <= abs_tol)
    elif method == "strong":
        return (((diff <= abs(rel_tol * b)) and
                 (diff <= abs(rel_tol * a))) or
                (diff <= abs_tol))
    elif method == "weak":
        return (((diff <= abs(rel_tol * b)) or
                 (diff <= abs(rel_tol * a))) or
                (diff <= abs_tol))
    elif method == "average":
        return ((diff <= abs(rel_tol * (a + b) / 2) or
                 (diff <= abs_tol)))
    else:
        raise ValueError('method must be one of:'
                         ' "asymmetric", "strong", "weak", "average"')
