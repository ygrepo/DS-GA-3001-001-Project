import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ts.abstract_trainer import BaseTrainer
from ts.utils.loss_modules import np_sMAPE, np_MASE, np_mase
from gpytorch import mlls
import gpytorch
from ts.benchmark.model import SpectralMixtureGPModel


class Trainer(BaseTrainer):
    def __init__(self, model_name, model, optimizer, criterion, dataloader, run_id, add_run_id, config, ohe_headers,
                 csv_path, figure_path,
                 sampling, reload):
        super().__init__(model_name, model, optimizer, criterion, dataloader, run_id, add_run_id, config, ohe_headers,
                         csv_path, figure_path, sampling, reload)

    def train_batch(self, train, val, test, info_cat, idx):

        #self.model.covar_module.initialize_from_data(train, val)
        y_train = train.view(1,-1,1)
        x_train = np.array(range(0, train.shape[1]))
        #val = val.view(1,-1,1)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=y_train.shape[1])
        self.model = SpectralMixtureGPModel(x_train, y_train, likelihood, num_outputs=y_train.shape[1])
        self.model.train()
        self.model.likelihood.train()
        self.mll = mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.config["lr_anneal_step"],
                                                         gamma=self.config["lr_anneal_rate"])

        self.optimizer.zero_grad()
        print(x_train.shape, y_train.shape)
        forecast = self.model(train)
        loss = -self.mll(forecast, val)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return float(loss)

    def val(self, file_path, testing=False):
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            acts = []
            preds = []
            total_acts = []
            info_cats = []
            hold_out_loss = 0
            for batch_num, (train, val, test, info_cat, _, idx) in enumerate(self.data_loader):
                target = test if testing else val
                if testing:
                    train = torch.cat((train, val), dim=1)
                ts_len = train.shape[1]
                input = train[:, ts_len - self.backcast_length:ts_len][np.newaxis, ...]
                backcast, forecast = self.model(input)
                hold_out_loss += self.criterion(forecast, target[np.newaxis, ...])
                acts.extend(target.view(-1).cpu().detach().numpy())
                preds.extend(forecast.view(-1).cpu().detach().numpy())
                total_act = torch.cat((train, forecast.view(forecast.shape[1], -1)), dim=1)
                total_acts.extend(total_act.view(-1).cpu().detach().numpy())
                info_cats.append(info_cat.cpu().detach().numpy())

            hold_out_loss = hold_out_loss / (batch_num + 1)

            info_cat_overall = np.concatenate(info_cats, axis=0)
            _hold_out_df = pd.DataFrame({"acts": acts, "preds": preds})
            cats = [val for val in self.ohe_headers[info_cat_overall.argmax(axis=1)] for _ in
                    range(self.config["output_size"])]
            _hold_out_df["category"] = cats

            overall_hold_out_df = copy.copy(_hold_out_df)
            overall_hold_out_df["category"] = ["Overall" for _ in cats]

            overall_hold_out_df = pd.concat((_hold_out_df, overall_hold_out_df), sort=False)

            mase = np_mase(total_acts, self.config["output_size"])
            grouped_results = overall_hold_out_df.groupby(["category"]).apply(
                lambda x: np_MASE(x.preds, x.acts, mase, x.shape[0]))
            results = grouped_results.to_dict()
            print("============== MASE ==============")
            print(results)

            grouped_results = overall_hold_out_df.groupby(["category"]).apply(
                lambda x: np_sMAPE(x.preds, x.acts, x.shape[0]))
            results = grouped_results.to_dict()

            print("============== sMAPE ==============")
            print(results)

            hold_out_loss = float(hold_out_loss.detach().cpu())
            print("============== HOLD-OUT-LOSS ==============")
            print("hold_out_loss:{:5.2f}".format(hold_out_loss))

            results["hold_out_loss"] = hold_out_loss
            self.log_values(results)

            grouped_path = file_path / ("grouped_results-{}.csv".format(self.epochs))
            grouped_results.to_csv(grouped_path, header=True)

        return hold_out_loss
