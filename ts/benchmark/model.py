import math
import torch
import gpytorch


class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, batch_size=1, num_mixtures=4):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.Mean(batch_size=batch_size)
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures, batch_size=batch_size)
        #self.covar_module.initialize_from_data(train_x, train_y)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
