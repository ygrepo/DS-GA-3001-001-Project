import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ts.utils.helper_funcs import MODEL_TYPE, load_model_parameters, load_model, save_model, \
    save_model_parameters, plot_stacks, SAVE_LOAD_TYPE, isclose
from ts.utils.logger import Logger


class BaseTrainer(nn.Module):
    def __init__(self, model_name, model, optimizer, criterion, dataloader, run_id, add_run_id, config, ohe_headers,
                 csv_path, figure_path,
                 sampling=False, reload=SAVE_LOAD_TYPE.NO_ACTION):
        super(BaseTrainer, self).__init__()
        self.model_name = model_name
        if model:
            self.model = model.to(config["device"])
        else:
            self.model = None
        self.config = config
        self.data_loader = dataloader
        self.sampling = sampling
        self.ohe_headers = ohe_headers
        self.optimizer = optimizer
        # self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=config["learning_rate"])
        if optimizer:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                              step_size=config["lr_anneal_step"],
                                                              gamma=config["lr_anneal_rate"])
            #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True)
        else:
            self.scheduler = None
        self.criterion = criterion
        self.epochs = 0
        self.max_epochs = config["num_of_train_epochs"]
        if sampling:
            self.max_epochs = config["num_of_train_epochs_sampling"]
        self.run_id = str(run_id)
        self.add_run_id = add_run_id
        self.prod_str = "prod" if config["prod"] else "dev"
        self.csv_save_path = csv_path
        self.figure_path = figure_path
        logger_path = str(csv_path / ("tensorboard/" + self.model_name) / (
                "train%s%s%s" % (self.config["variable"], self.prod_str, self.run_id)))
        self.log = Logger(logger_path)
        self.reload = reload

    def plot_ts_enabled(self):
        return self.config["plot_ts"] and self.config["sample"]

    def save_model_enabled(self):
        return self.config["save_model"] == SAVE_LOAD_TYPE.MODEL or self.config[
            "save_model"] == SAVE_LOAD_TYPE.MODEL_PARAMETERS

    def train_epochs(self):
        max_loss = 1e8
        start_time = time.time()
        file_path = Path(".") / ("models/" + self.model_name)
        if self.reload == SAVE_LOAD_TYPE.MODEL:
            model, optimizer = load_model(file_path, self.config)
            self.model = model
            self.optimizer = optimizer
        if self.reload == SAVE_LOAD_TYPE.MODEL_PARAMETERS:
            load_model_parameters(file_path, self.model, self.optimizer)
        max_loss_repeat = 3
        loss_repeat_counter = 1
        prev_loss = float("-inf")
        for e in range(self.max_epochs):

            if self.reload != SAVE_LOAD_TYPE.NO_ACTION:
                epoch_loss = 0
            else:
                epoch_loss = self.train()

            if self.save_model_enabled() and epoch_loss < max_loss:
                print("Loss decreased, saving model!")
                file_path = Path(".") / ("models/" + self.model_name)
                if self.config["save_model"] == SAVE_LOAD_TYPE.MODEL:
                    save_model(file_path, self.model, self.optimizer, self.run_id, self.add_run_id)
                elif self.config["save_model"] == SAVE_LOAD_TYPE.MODEL_PARAMETERS:
                    save_model_parameters(file_path, self.model, self.optimizer, self.run_id, self.add_run_id)
                max_loss = epoch_loss

            file_path = self.csv_save_path / "grouped_results" / self.run_id / self.prod_str
            file_path_validation_loss = file_path / "validation_losses.csv"

            if e == 0:
                file_path.mkdir(parents=True, exist_ok=True)
                with open(file_path_validation_loss, "w") as f:
                    f.write("epoch,training_loss,validation_loss\n")
            if e == self.max_epochs - 1 and self.model_name == MODEL_TYPE.NBEATS.value and self.plot_ts_enabled():
                plot_stacks(self.run_id, self.figure_path, self.model)

            epoch_val_loss = self.val(file_path, testing=True, debugging=False, figure_path=self.figure_path)
            with open(file_path_validation_loss, "a") as f:
                f.write(",".join([str(e), str(epoch_loss), str(epoch_val_loss)]) + "\n")
            self.scheduler.step()

            if isclose(epoch_val_loss, prev_loss, rel_tol=1e-4):
                loss_repeat_counter += 1
                if loss_repeat_counter >= max_loss_repeat:
                    print("Loss not decreasing for last {} times".format(loss_repeat_counter))
                    if self.model_name == MODEL_TYPE.NBEATS.value and self.plot_ts_enabled() \
                            and self.config["sample_ids"]:
                        plot_stacks(self.run_id, self.figure_path, self.model)

                    if self.model_name == MODEL_TYPE.ESRNN.value and self.plot_ts_enabled() \
                            and self.config["sample_ids"]:
                        self.val(file_path, testing=True, debugging=True, figure_path=self.figure_path)
                    break
                else:
                    loss_repeat_counter += 1
            prev_loss = epoch_val_loss

            self.epochs += 1

        if self.plot_ts_enabled():
            self.plot(testing=True)
        print("Total Training in mins: %5.2f" % ((time.time() - start_time) / 60))

    def train(self):
        if self.model:
            self.model.train()
        epoch_loss = 0
        for batch_num, (train, val, test, info_cat, _, idx) in enumerate(self.data_loader):
            start = time.time()
            print("Train_batch: %d" % (batch_num + 1))
            loss = self.train_batch(train, val, test, info_cat, idx)
            epoch_loss += loss
            end = time.time()
            self.log.log_scalar("Iteration time", end - start, batch_num + 1 * (self.epochs + 1))
        epoch_loss = epoch_loss / (batch_num + 1)

        # LOG EPOCH LEVEL INFORMATION
        print("[TRAIN]  Epoch [%d/%d]   Loss: %.4f" % (self.epochs, self.max_epochs, epoch_loss))
        info = {"loss": epoch_loss}

        self.log_values(info)
        self.log_hists()

        return epoch_loss

    def train_batch(self, train, val, test, info_cat, idx):
        pass

    def val(self, file_path, testing, debugging=False, figure_path=None):
        return 0

    def plot(self, filepath=None, testing=True):
        pass

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
