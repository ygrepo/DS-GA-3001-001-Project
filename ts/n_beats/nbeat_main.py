import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ts.n_beats.config import get_config
from ts.n_beats.data_loading import create_datasets, SeriesDataset
from ts.n_beats.model import NBeatsNet
from ts.n_beats.trainer import Trainer


# simple batcher.
def data_generator(x_full, y_full, bs):
    def split(arr, size):
        arrays = []
        while len(arr) > size:
            slice_ = arr[:size]
            arrays.append(slice_)
            arr = arr[size:]
        arrays.append(arr)
        return arrays

    while True:
        for rr in split((x_full, y_full), bs):
            yield rr


def train_100_grad_steps(data, device, net, optimiser, test_losses):
    global_step = 0
    # load(net, optimiser)
    for x_train_batch, y_train_batch in data:
        global_step += 1
        optimiser.zero_grad()
        net.train()
        _, forecast = net(torch.tensor(x_train_batch, dtype=torch.float).to(device))
        loss = F.mse_loss(forecast, torch.tensor(y_train_batch, dtype=torch.float).to(device))
        loss.backward()
        optimiser.step()
        if global_step % 30 == 0:
            print(
                f'grad_step = {str(global_step).zfill(6)}, tr_loss = {loss.item():.6f}, te_loss = {test_losses[-1]:.6f}')
        if global_step > 0 and global_step % 100 == 0:
            with torch.no_grad():
                #    save(net, optimiser, global_step)
                i = 1
            break


def eval_test(backcast_length, forecast_length, net, norm_constant, test_losses, x_test, y_test):
    net.eval()
    _, forecast = net(torch.tensor(x_test, dtype=torch.float))
    test_losses.append(F.mse_loss(forecast, torch.tensor(y_test, dtype=torch.float)).item())
    p = forecast.detach().numpy()
    subplots = [221, 222, 223, 224]
    plt.figure(1)
    for plot_id, i in enumerate(np.random.choice(range(len(x_test)), size=4, replace=False)):
        ff, xx, yy = p[i] * norm_constant, x_test[i] * norm_constant, y_test[i] * norm_constant
        # plt.subplot(subplots[plot_id])
        # plt.grid()
        # plot_scatter(range(0, backcast_length), xx, color='b')
        # plot_scatter(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        # plot_scatter(range(backcast_length, backcast_length + forecast_length), ff, color='r')
    # plt.show()


def train(backcast_length, forecast_length, net, norm_constant, x_test, y_test, data,
          device, optimiser):
    test_losses = []
    for i in range(30):
        eval_test(backcast_length, forecast_length, net, norm_constant, test_losses, x_test, y_test)
        train_100_grad_steps(data, device, net, optimiser, test_losses)
        print(test_losses[-1])


def main_old():
    device = torch.device('cpu')  # use the trainer.py to run on GPU.
    forecast_length = 18
    backcast_length = 3 * forecast_length
    batch_size = 10  # greater than 4 for viz

    milk = pd.read_csv('n_beats/milk.csv', index_col=0, parse_dates=True)

    print(milk.head())
    milk = milk.values  # just keep np array here for simplicity.
    norm_constant = np.max(milk)
    milk = milk / norm_constant  # small leak to the test set here.

    x_train_batch, y = [], []
    for i in range(backcast_length, len(milk) - forecast_length):
        x_train_batch.append(milk[i - backcast_length:i])
        y.append(milk[i:i + forecast_length])

    x_train_batch = np.array(x_train_batch)[..., 0]
    y = np.array(y)[..., 0]

    c = int(len(x_train_batch) * 0.8)
    x_train, y_train = x_train_batch[:c], y[:c]
    x_test, y_test = x_train_batch[c:], y[c:]

    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    # data
    data = data_generator(x_train, y_train, batch_size)
    net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                    forecast_length=forecast_length,
                    thetas_dims=[7, 8],
                    nb_blocks_per_stack=3,
                    backcast_length=backcast_length,
                    hidden_layer_units=128,
                    share_weights_in_stack=True,
                    device=device)
    optimiser = optim.Adam(net.parameters())
    train(backcast_length, forecast_length, net, norm_constant, x_test, y_test, data, device, optimiser)


def main():
    device = torch.device('cpu')  # use the trainer.py to run on GPU.
    print("Start")

    BASE_DIR = Path("/Users/yg/code/github/esrnn-gpu/data/raw/")
    print('loading config')
    config = get_config('Monthly')
    forecast_length = config["output_size"]
    backcast_length = 1 * forecast_length

    print('loading data')
    info = pd.read_csv(str(BASE_DIR / "M4info.csv"))
    train_path = str(BASE_DIR / "train/%s-train.csv") % (config['variable'])
    test_path = str(BASE_DIR / "test/%s-test.csv") % (config['variable'])

    sample = False
    train, train_idx, val, test, test_idx = create_datasets(train_path, test_path, config['output_size'],
                                                            sample=sample, sampling_size=4)
    print("#of train ts:{}, dimensions of validation ts:{}, dimensions of test ts:{}".format(train.shape, val.shape,
                                                                                             test.shape))

    dataset = SeriesDataset(info, config["variable"], sample, train, train_idx, val, test, backcast_length,
                            forecast_length,
                            config['device'])
    # dataloader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=collate_lines, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    model = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                      forecast_length=forecast_length,
                      thetas_dims=[7, 8],
                      nb_blocks_per_stack=3,
                      backcast_length=backcast_length,
                      hidden_layer_units=128,
                      share_weights_in_stack=True,
                      device=device)
    run_id = str(int(time.time()))
    trainer = Trainer(device, model, dataloader, run_id, config, forecast_length, backcast_length,
                      ohe_headers=dataset.dataInfoCatHeaders)
    trainer.train_epochs()


if __name__ == '__main__':
    main()
