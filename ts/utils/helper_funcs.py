import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import random

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


def save(file_path, model, optimiser):
    file_path.mkdir(parents=True, exist_ok=True)
    model_path = file_path / "model.pyt"
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
        ax.plot(x, y, "g-", x, y_pred, "b-")
        ax.set_xlabel("Time")
        ax.set_ylabel("Observations")
        ts_label = next(fig_label_iter)
        ax.set_title(ts_label + " time Series:" + ts_labels[i])
        ax.legend(("original", "predicted"))
    plt.savefig(path / "original_time_series.png")
    plt.tight_layout()
    sns.despine()
    plt.show()
