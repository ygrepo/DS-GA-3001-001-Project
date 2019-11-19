import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


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


def save(file_path, model, optimiser, run_id, add_run_id=False):
    file_path.mkdir(parents=True, exist_ok=True)
    if add_run_id:
        model_path = file_path / ("model_" + run_id + ".pyt")
    else:
        model_path = file_path / ("model.pyt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimiser.state_dict(),
    }, model_path)


def load(file_path, model, optimiser):
    model_path = file_path / "model.pyt"
    if model_path.exists():
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Restored checkpoint from {model_path}.")


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


def create_datasets(train_file_location, test_file_location, output_size, sample_ids, sample=False, sampling_size=5):
    train, train_idx = read_file(train_file_location, sample_ids, sample, sampling_size)
    if sample and sample_ids:
        train, train_idx = filter_sample_ids(train, train_idx, sample_ids)
    print("After chopping: train:{}".format(len(train)))

    test, test_idx = read_file(test_file_location, sample_ids, sample, sampling_size)
    if sample and sample_ids:
        test, test_idx = filter_sample_ids(test, test_idx, sample_ids)

    train, val = create_val_set(train, output_size)
    if sample and sample_ids:
        print("Sampling train data for {}".format(sample_ids))
        print("Sampling test data for {}".format(sample_ids))

    elif sample and not sample_ids:
        print("Sampling train data for {}".format(train_idx.keys()))
        print("Sampling test data for {}".format(test_idx.keys()))

    return train, train_idx, val, test, test_idx


def chop_series(train, chop_val):
    # CREATE MASK FOR VALUES TO BE CHOPPED
    train_len_mask = [True if len(i) >= chop_val else False for i in train]
    # FILTER AND CHOP TRAIN
    train = [train[i][-chop_val:] for i in range(len(train)) if train_len_mask[i]]
    return train, train_len_mask


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


def plot_bc_fc(run_id, path, bc, fc):
    path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(17, 4))
    plt.subplots_adjust(hspace=.3)

    for i in range(len(bc)):
        #fig_label_iter = iter(cats)
        y = bc[i].squeeze().cpu().detach().numpy()
        for j in range(y.shape[0]):
            x = np.arange(y.shape[1])
            ax.plot(x, y[i], "g-")

    ax.set_xlabel("Time")
    ax.set_ylabel("Observations")
    # ax.set_title(ts_label + " time Series:" + ts_labels[i])
    ax.legend(("original", "predicted"))
    # plt.savefig(path / ("window_" + run_id + " .png"))
    plt.tight_layout()
    sns.despine()
    # if show:
    plt.show()


def plot_ts(run_id, original_ts, predicted_ts, ts_labels, cats, path, number_to_plot=1, show=True):
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
        ax.plot(x, y, "g-", x, y_pred, "b-")
        ax.set_xlabel("Time")
        ax.set_ylabel("Observations")
        ts_label = next(fig_label_iter)
        ax.set_title(ts_label + " time Series:" + ts_labels[i])
        ax.legend(("original", "predicted"))
    plt.savefig(path / ("time_series_" + run_id + " .png"))
    plt.tight_layout()
    sns.despine()
    if show:
        plt.show()
