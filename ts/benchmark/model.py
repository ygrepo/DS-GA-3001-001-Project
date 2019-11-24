import math
import torch
import gpytorch


class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood, num_outputs=1, num_mixtures=4):
        super(SpectralMixtureGPModel, self).__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_size=num_outputs)
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures, batch_size=num_outputs)
        self.likelihood = likelihood
        self.num_outputs = num_outputs

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
