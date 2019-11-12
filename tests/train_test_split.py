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


def train():
    args = get_script_arguments()
    device = torch.device('cuda') if not args.disable_cuda and torch.cuda.is_available() else torch.device('cpu')
    forecast_length = 1
    backcast_length = 3 * forecast_length
    batch_size = 6  # greater than 4 for viz

    train_read=read_file('M4Dataset/Quarterly-train.csv')
    test_read=read_file('M-test-set/Quarterly-test.csv')

    train_to_use= train_read[1]/np.max(np.abs(train_read[1]))
    test_to_use=test_read[1]/np.max(np.abs(test_read[1]))
    #test_to_use= test_read[1]/np.max(np.abs(test_read[1])
    
    

    quarterly_train_xbatch,quarterly_train_ybatch = [], []
    for i in range(backcast_length, len(train_to_use) - forecast_length+1):
        quarterly_train_xbatch.append(train_to_use[i - backcast_length:i])
        quarterly_train_ybatch.append(train_to_use[i:i + forecast_length])

    quarterly_test_xbatch,quarterly_test_ybatch = [], []
    for i in range(backcast_length, len(test_to_use) - forecast_length+1):
       quarterly_test_xbatch.append(test_to_use[i - backcast_length:i])
       quarterly_test_ybatch.append(test_to_use[i:i + forecast_length])

    data_gen=data_generator(quarterly_train_xbatch, quarterly_train_ybatch, batch_size)




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

    def plot_model(x_test, target_test,test_losses, device, grad_step):
        if not args.disable_plot:
            print('plot()')
            plot(net, x_test, target_test, test_losses, backcast_length, forecast_length, device, grad_step)

    simple_fit(net, optimiser, data_gen, plot_model, device, quarterly_test_xbatch, quarterly_test_ybatch)


def simple_fit(net, optimiser, data_generator, on_save_callback, device, x_test, target_test, max_grad_steps=10000):
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
        #print(f'grad_step = {str(grad_step).zfill(6)}, loss = {loss.item():.6f}')
        if grad_step % 1000 == 0 or (grad_step < 1000 and grad_step % 100 == 0):
            with torch.no_grad():
                save(net, optimiser, grad_step)
                if on_save_callback is not None:
                    test_losses=[]
                    on_save_callback(x_test, target_test, test_losses,device, grad_step)
        print(f'grad_step = {str(grad_step).zfill(6)}, loss = {loss.item():.6f}, te_loss = {test_losses[-1]:.6f}')
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


def plot(net, x_test, target_test, test_losses, backcast_length, forecast_length, device, grad_step):
    net.eval()
    _, f = net(torch.tensor(x_test, dtype=torch.float).to(device))
    subplots = [221, 222, 223, 224]
    test_losses.append(F.mse_loss(f, torch.tensor(target_test, dtype=torch.float).to(device)))
    print(len(x_test),len(target_test))
    print(len(f.cpu().numpy()))
   
    plt.figure(1)
    plt.subplots_adjust(top=0.88)
    for i in range(4):
        ff, xx, yy = f.cpu().numpy()[i], x_test[i], target_test[i]
        plt.subplot(subplots[i])
        plt.plot(range(0, backcast_length), xx, color='b')
        plt.plot(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plt.plot(range(backcast_length, backcast_length + forecast_length), ff, color='r')
        # plt.title(f'step #{grad_step} ({i})')

    output = 'n_beats_{}.png'.format(grad_step)
    plt.savefig(output)
    plt.clf()
    print('Saved image to {}.'.format(output))


if __name__ == '__main__':
    train()
