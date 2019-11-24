import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ts.abstract_trainer import BaseTrainer
from ts.utils.helper_funcs import plot_ts
from ts.utils.loss_modules import np_sMAPE, np_MASE, np_mase


class ESRNNTrainer(BaseTrainer):
    def __init__(self, model_name, model, optimizer, criterion, dataloader, run_id, add_run_id, config, ohe_headers,
                 csv_path, figure_path,
                 sampling, reload):
        super().__init__(model_name, model, optimizer, criterion, dataloader, run_id, add_run_id, config, ohe_headers,
                         csv_path, figure_path, sampling, reload)

    def train_batch(self, train, val, test, info_cat, idx):
        self.optimizer.zero_grad()
        network_pred, network_act, _, _, loss_mean_sq_log_diff_level = self.model(train, val,
                                                                                  test, info_cat,
                                                                                  idx)

        # Computing loss between predicted and truth training data deseasonalized and normalized.
        loss = self.criterion(network_pred, network_act)
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), self.config["gradient_clipping"])
        self.optimizer.step()
        self.scheduler.step()
        return float(loss)

    def val(self, file_path, testing, debugging, figure_path):
        print("Validation")
        self.model.eval()
        with torch.no_grad():
            acts = []
            total_acts = []
            preds = []
            info_cats = []

            hold_out_loss = 0
            for batch_num, (train, val, test, info_cat, _, idx) in enumerate(self.data_loader):
                _, _, (hold_out_pred, network_output_non_train), \
                (hold_out_act, hold_out_act_deseas_norm), _ = self.model(train, val, test, info_cat, idx,
                                                                         testing=testing,
                                                                         debugging=debugging, figure_path=self.figure_path)
                # Compute loss between normalized and deseasonalized predictions and
                # either validation or test data depending the value of the flag testing
                # hold_out_loss += self.criterion(network_output_non_train.unsqueeze(0).float(),
                #                                 hold_out_act_deseas_norm.unsqueeze(0).float())
                hold_out_loss += self.criterion(hold_out_pred, hold_out_act)
                acts.extend(hold_out_act.view(-1).cpu().detach().numpy())
                total_act = torch.cat((train, hold_out_pred), dim=1)
                total_acts.extend(total_act.view(-1).cpu().detach().numpy())
                preds.extend(hold_out_pred.view(-1).cpu().detach().numpy())
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
            print("hold_out_loss:{:8.4f}".format(hold_out_loss))

            results["hold_out_loss"] = hold_out_loss
            self.log_values(results)

            grouped_path = file_path / ("grouped_results-{}.csv".format(self.epochs))
            grouped_results.to_csv(grouped_path, header=True)

        return hold_out_loss

    def plot(self, testing=False):
        self.model.eval()
        with torch.no_grad():
            (train, val, test, info_cat, ts_labels, idx) = next(iter(self.data_loader))
            info_cats = info_cat.cpu().detach().numpy()
            cats = [val for val in self.ohe_headers[info_cats.argmax(axis=1)]]

            if testing:
                train = torch.cat((train, val), dim=1)
            _, _, (hold_out_pred, _), (hold_out_act, _), _ = self.model(train, val, test, info_cat, idx,
                                                                        testing=testing)
            original_ts = torch.cat((train, hold_out_act), axis=1)
            predicted_ts = torch.cat((train, hold_out_pred), axis=1)
            plot_ts(self.run_id, original_ts, predicted_ts, ts_labels, cats, self.figure_path,
                    number_to_plot=train.shape[0])
