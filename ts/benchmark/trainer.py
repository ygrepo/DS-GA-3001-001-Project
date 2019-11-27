import time

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from gpytorch import mlls

from ts.benchmark.model import SpectralMixtureGPModel
from ts.utils.loss_modules import sMAPE


class Trainer:
    def __init__(self, model_name, dataloader, run_id, add_run_id, config, csv_path, figure_path):
        super(Trainer, self).__init__()
        self.model_name = model_name
        self.config = config
        self.data_loader = dataloader
        self.run_id = str(run_id)
        self.add_run_id = add_run_id
        self.csv_save_path = csv_path
        self.figure_path = figure_path

    def train(self):
        self.figure_path.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        ts_label = self.config["sample_ids"][0]
        (train, val, test, info_cat, ts_labels, idx) = next(iter(self.data_loader))

        data_y = np.squeeze(torch.cat((train, val), dim=1))
        N_samples = min(self.config["min_samples"], data_y.shape[0])
        sample_indices = np.random.choice(data_y.shape[0], N_samples, replace=False)
        data_x = np.arange(data_y.shape[0], dtype=float)
        train_x = torch.from_numpy(data_x[sample_indices]).float()
        train_y = data_y[sample_indices]

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = SpectralMixtureGPModel(train_x, train_y, likelihood)
        model.train()
        likelihood.train()
        mll = mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["learning_rate"])

        training_iter = 100
        for i in range(training_iter):
            optimizer.zero_grad()
            forecast = model(train_x)
            loss = - mll(forecast, train_y)
            loss.backward()
            optimizer.step()
            print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.likelihood.noise.item()
            ))

        test_data_y = np.squeeze(torch.cat((train, val, test), axis=1))
        test_data_x = np.arange(test_data_y.shape[0], dtype=float)
        test_x = torch.from_numpy(test_data_x).float()

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Make predictions


            observed_pred = likelihood(model(test_x))

            # Initialize plot
            f, ax = plt.subplots(figsize=(17, 4))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            # Plot training data
            ax.plot(test_data_x, test_data_y, "r")
            ax.plot(data_x, data_y, "k")
            ax.plot(train_x.numpy(), train_y.numpy(), "k*")
            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), "b")
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.legend(["Forecast", "Truth", "Samples", "Mean", "Confidence"])
            ax.set_xlabel("Time")
            ax.set_ylabel("Observations")
            mape = sMAPE(observed_pred.mean[-self.config["output_size"]:],
                         test_data_y[-self.config["output_size"]:], self.config["output_size"])
            ax.set_title("Time Series:{}, MAPE:{:8.2f}, N samples:{}".format(ts_label, mape, N_samples))
            #plt.show()
            plt.tight_layout()
            sns.despine()
            plt.savefig(self.figure_path / (ts_label + "_time_series.png"), bbox_inches="tight")

        print("Total Training in mins: %5.2f" % ((time.time() - start_time) / 60))
