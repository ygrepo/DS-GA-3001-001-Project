import os
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.nn import functional as F

from data import get_data
from n_beats.model import NBeatsNet

CHECKPOINT_NAME = 'nbeats-training-checkpoint.th'


def get_script_arguments():
    parser = ArgumentParser(description='N-Beats')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--disable-plot', action='store_true', help='Disable interactive plots')
    args = parser.parse_args()
    return args

def read_file(file_location):
    series = []
    ids = []
    with open(file_location, 'r') as file:
        data = file.read().split("\n")

    for i in range(1, len(data) - 1):
    # for i in range(1, len(data)):
        row = data[i].replace('"', '').split(',')
        series.append(np.array([float(j) for j in row[1:] if j != ""]))
        ids.append(row[0])

    series = np.array(series)
    return series

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


def train(ind):
    args = get_script_arguments()
    device = torch.device('cuda') if not args.disable_cuda and torch.cuda.is_available() else torch.device('cpu')
    forecast_length = 2
    backcast_length = 3 * forecast_length
    batch_size = 6  # greater than 4 for viz

    train_read=read_file('M4Dataset/Quarterly-train.csv')
    test_read=read_file('M-test-set/Quarterly-test.csv')

    merged=np.concatenate((train_read[ind] , test_read[ind]), axis=0)
    merged= merged/np.max(np.abs(merged))
    #train_to_use=train_read[ind]
    #train_to_use= train_to_use/np.max(np.abs(train_to_use))

    quarterly_xbatch,quarterly_ybatch = [], []
    for i in range(backcast_length, len(merged) - forecast_length+1):
        quarterly_xbatch.append(merged[i - backcast_length:i])
        quarterly_ybatch.append(merged[i:i + forecast_length])

    #quarterly_test_xbatch,quarterly_test_ybatch = [], []
    #for i in range(backcast_length, len(test_read[1]) - forecast_length+1):
       # quarterly_test_xbatch.append(test_read[1][i - backcast_length:i])
       # quarterly_test_ybatch.append(test_read[1][i:i + forecast_length])

    data_gen=data_generator(quarterly_xbatch, quarterly_ybatch, batch_size)



    q=ind
    print('--- Model ---')
    net = NBeatsNet(device=device,
                    stack_types=[NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK],
                    forecast_length=forecast_length,
                    thetas_dims=[2, 8],
                    nb_blocks_per_stack=3,
                    backcast_length=backcast_length,
                    hidden_layer_units=1024,
                    share_weights_in_stack=False)

    # net = NBeatsNet(device=device,
    #                 stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
    #                 forecast_length=forecast_length,
    #                 thetas_dims=[7, 8],
    #                 nb_blocks_per_stack=3,
    #                 backcast_length=backcast_length,
    #                 hidden_layer_units=128,
    #                 share_weights_in_stack=False)

    optimiser = optim.Adam(net.parameters())

    def plot_model(x, target, grad_step,q):
        if not args.disable_plot:
            print('plot()')
            plot(net, x, target, backcast_length, forecast_length, grad_step,q)

    simple_fit(net, optimiser, data_gen, plot_model, device,q)


def simple_fit(net, optimiser, data_generator, on_save_callback, device,q,max_grad_steps=10000):
    print('--- Training ---')
    initial_grad_step = load(net, optimiser)
    for grad_step, (x, target) in enumerate(data_generator):
        grad_step += initial_grad_step
        optimiser.zero_grad()
        net.train()
        backcast, forecast = net(torch.tensor(x, dtype=torch.float).to(device))
        loss = F.mse_loss(forecast, torch.tensor(target, dtype=torch.float).to(device))
        loss.backward()
        optimiser.step()
        print(f'grad_step = {str(grad_step).zfill(6)}, loss = {loss.item():.6f}')
        if grad_step % 1000 == 0 or (grad_step < 1000 and grad_step % 100 == 0):
            with torch.no_grad():
                save(net, optimiser, grad_step)
                if on_save_callback is not None:
                    on_save_callback(x, target, grad_step,q)
        if grad_step > max_grad_steps:
            print('Finished.')
            break


def save(model, optimiser, grad_step):
    torch.save({
        'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, CHECKPOINT_NAME)


def load(model, optimiser):
    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        grad_step = checkpoint['grad_step']
        print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
        return grad_step
    return 0


def plot(net, x, target, backcast_length, forecast_length, grad_step,q):
    net.eval()
    _, f = net(torch.tensor(x, dtype=torch.float))
    subplots = [221, 222, 223, 224]

    plt.figure(1)
    plt.subplots_adjust(top=0.88)
    for i in range(4):
        ff, xx, yy = f.cpu().numpy()[i], x[i], target[i]
        plt.subplot(subplots[i])
        plt.plot(range(0, backcast_length), xx, color='b')
        plt.plot(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plt.plot(range(backcast_length, backcast_length + forecast_length), ff, color='r')
        # plt.title(f'step #{grad_step} ({i})')

    output = 'n_beats_{}_{}.png'.format(grad_step,q)
    plt.savefig(output)
    plt.clf()
    print('Saved image to {}.'.format(output))


if __name__ == '__main__':
    for i in range(20):
        train(i)
        if os.path.exists(CHECKPOINT_NAME):
            os.remove(CHECKPOINT_NAME)


