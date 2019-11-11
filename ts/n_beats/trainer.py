import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from ts.n_beats.model import NBeatsNet
from ts.utils.helper_funcs import load, save
from ts.utils.logger import Logger
from ts.utils.loss_modules import PinballLoss, np_sMAPE


class Trainer(nn.Module):
    def __init__(self, device, model, dataloader, run_id, config, forecast_length, backcast_length, ohe_headers,
                 csv_path, reload):
        super(Trainer, self).__init__()
        self.device = device
        self.model = model.to(config["device"])
        self.config = config
        self.data_loader = dataloader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=config["lr_anneal_step"],
                                                         gamma=config["lr_anneal_rate"])
        self.criterion = PinballLoss(self.config["training_tau"],
                                     self.config["output_size"] * self.config["batch_size"], self.config["device"])
        # self.criterion = nn.MSELoss()
        self.criterion = nn.SmoothL1Loss()
        self.epochs = 0
        self.max_epochs = config["num_of_train_epochs"]
        self.run_id = str(run_id)
        self.prod_str = "prod" if config["prod"] else "dev"
        logger_path = str(
            csv_path / "tensorboard/nbeat" / ("train%s%s%s" % (self.config['variable'], self.prod_str, self.run_id)))
        self.log = Logger(logger_path)
        self.csv_save_path = csv_path
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.ohe_headers = ohe_headers
        self.reload = reload

    def train_epochs(self):
        max_loss = 1e8
        start_time = time.time()
        file_path = Path(".") / "models/nbeats"
        if self.reload:
            load(file_path, self.model, self.optimizer)
        for e in range(self.max_epochs):
            epoch_loss = self.train()
            if epoch_loss < max_loss:
                print("Loss decreased, saving model!")
                file_path = Path(".") / "models/nbeats"
                save(file_path, self.model, self.optimizer)
                max_loss = epoch_loss
            file_path = self.csv_save_path / "grouped_results" / self.run_id / self.prod_str
            file_path_validation_loss = file_path / "validation_losses.csv"
            if e == 0:
                file_path.mkdir(parents=True, exist_ok=True)
                with open(file_path_validation_loss, "w") as f:
                    f.write("epoch,training_loss,validation_loss\n")
            epoch_val_loss = self.val(file_path, testing=True)
            with open(file_path_validation_loss, "a") as f:
                f.write(",".join([str(e), str(epoch_loss), str(epoch_val_loss)]) + "\n")
            self.epochs += 1
        print("Total Training Mins: %5.2f" % ((time.time() - start_time) / 60))

    def train(self):
        self.model.train()
        epoch_loss = 0
        for batch_num, (train, val, test, info_cat, idx) in enumerate(self.data_loader):
            start = time.time()
            print("Train_batch: %d" % (batch_num + 1))
            loss = self.train_batch(train)
            epoch_loss += loss
            end = time.time()
            self.log.log_scalar("Iteration time", end - start, batch_num + 1 * (self.epochs + 1))
        epoch_loss = epoch_loss / (batch_num + 1)

        # LOG EPOCH LEVEL INFORMATION
        print("[TRAIN]  Epoch [%d/%d]   Loss: %.4f" % (
            self.epochs, self.max_epochs, epoch_loss))
        info = {"loss": epoch_loss}

        self.log_values(info)
        # self.log_hists()

        return epoch_loss

    def train_batch(self, train):
        self.optimizer.zero_grad()

        window_input_list = []
        window_output_list = []
        ts_len = train.shape[1]
        for j in range(self.backcast_length, ts_len - self.forecast_length):
            window_input_list.append(train[:, j - self.backcast_length:j])
            window_output_list.append(train[:, j:j + self.forecast_length])

        window_input = torch.cat([i.unsqueeze(0) for i in window_input_list], dim=0)
        window_output = torch.cat([i.unsqueeze(0) for i in window_output_list], dim=0)
        # network_pred = self.series_forward(window_input[:-self.config["output_size"]])
        backcast, forecast = self.model(window_input)
        loss = self.criterion(forecast, window_output)
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), self.config["gradient_clipping"])
        self.optimizer.step()
        self.scheduler.step()
        return float(loss)

    def val(self, file_path, testing=False):
        self.model.eval()
        with torch.no_grad():
            acts = []
            preds = []
            info_cats = []
            hold_out_loss = 0
            for batch_num, (train, val, test, info_cat, idx) in enumerate(self.data_loader):
                target = test if testing else val
                if testing:
                    train = torch.cat((train, val), dim=1)
                ts_len = train.shape[1]
                input = train[:, ts_len - self.backcast_length:ts_len][np.newaxis, ...]
                backcast, forecast = self.model(input)
                hold_out_loss += self.criterion(forecast, target[np.newaxis, ...])
                acts.extend(target.view(-1).cpu().detach().numpy())
                preds.extend(forecast.view(-1).cpu().detach().numpy())
                info_cats.append(info_cat.cpu().detach().numpy())

            hold_out_loss = hold_out_loss / (batch_num + 1)

            info_cat_overall = np.concatenate(info_cats, axis=0)
            _hold_out_df = pd.DataFrame({"acts": acts, "preds": preds})
            cats = [val for val in self.ohe_headers[info_cat_overall.argmax(axis=1)] for _ in
                    range(self.config["output_size"])]
            _hold_out_df["category"] = cats

            overall_hold_out_df = copy.copy(_hold_out_df)
            overall_hold_out_df["category"] = ["Overall" for _ in cats]

            overall_hold_out_df = pd.concat((_hold_out_df, overall_hold_out_df))
            grouped_results = overall_hold_out_df.groupby(["category"]).apply(
                lambda x: np_sMAPE(x.preds, x.acts, x.shape[0]))

            results = grouped_results.to_dict()
            results["hold_out_loss"] = float(hold_out_loss.detach().cpu())
            print(results)

            self.log_values(results)

            grouped_path = file_path / ("grouped_results-{}.csv".format(self.epochs))
            grouped_results.to_csv(grouped_path, header=True)

        return hold_out_loss.detach().cpu().item()

    def log_values(self, info):

        # SCALAR
        for tag, value in info.items():
            self.log.log_scalar(tag, value, self.epochs + 1)

    def log_hists(self):
        # HISTS
        batch_params = dict()
        for tag, value in self.model.named_parameters():
            if value.grad is not None:
                if "init" in tag:
                    name, _ = tag.split(".")
                    if name not in batch_params.keys() or "%s/grad" % name not in batch_params.keys():
                        batch_params[name] = []
                        batch_params["%s/grad" % name] = []
                    batch_params[name].append(value.data.cpu().numpy())
                    batch_params["%s/grad" % name].append(value.grad.cpu().numpy())
                else:
                    tag = tag.replace(".", "/")
                    self.log.log_histogram(tag, value.data.cpu().numpy(), self.epochs + 1)
                    self.log.log_histogram(tag + "/grad", value.grad.data.cpu().numpy(), self.epochs + 1)
            else:
                print("Not printing %s because it\"s not updating" % tag)

        for tag, v in batch_params.items():
            vals = np.concatenate(np.array(v))
            self.log.log_histogram(tag, vals, self.epochs + 1)


def train():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    forecast_length = 10
    backcast_length = 5 * forecast_length
    batch_size = 100  # greater than 4 for viz

    print("--- Model ---")
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

    # def plot_model(x, target, grad_step):
    #    if not args.disable_plot:
    #        print("plot()")
    #        plot(net, x, target, backcast_length, forecast_length, grad_step)

    # simple_fit(net, optimiser, data_gen, plot_model, device)


# def simple_fit(net, optimiser, data_generator, on_save_callback, device, max_grad_steps=10000):
#     print("--- Training ---")
#     initial_grad_step = load(net, optimiser)
#     for grad_step, (x, target) in enumerate(data_generator):
#         grad_step += initial_grad_step
#         optimiser.zero_grad()
#         net.train()
#         backcast, forecast = net(torch.tensor(x, dtype=torch.float).to(device))
#         loss = F.mse_loss(forecast, torch.tensor(target, dtype=torch.float).to(device))
#         loss.backward()
#         optimiser.step()
#         print(f"grad_step = {str(grad_step).zfill(6)}, loss = {loss.item():.6f}")
#         if grad_step % 1000 == 0 or (grad_step < 1000 and grad_step % 100 == 0):
#             with torch.no_grad():
#                 save(net, optimiser, grad_step)
#                 if on_save_callback is not None:
#                     on_save_callback(x, target, grad_step)
#         if grad_step > max_grad_steps:
#             print("Finished.")
#             break


def plot(net, x, target, backcast_length, forecast_length, grad_step):
    net.eval()
    _, f = net(torch.tensor(x, dtype=torch.float))
    subplots = [221, 222, 223, 224]

    plt.figure(1)
    plt.subplots_adjust(top=0.88)
    for i in range(4):
        ff, xx, yy = f.cpu().numpy()[i], x[i], target[i]
        plt.subplot(subplots[i])
        plt.plot(range(0, backcast_length), xx, color="b")
        plt.plot(range(backcast_length, backcast_length + forecast_length), yy, color="g")
        plt.plot(range(backcast_length, backcast_length + forecast_length), ff, color="r")
        # plt.title(f"step #{grad_step} ({i})")

    output = "n_beats_{}.png".format(grad_step)
    plt.savefig(output)
    plt.clf()
    print("Saved image to {}.".format(output))


if __name__ == "__main__":
    train()
