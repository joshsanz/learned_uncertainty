import numpy as np


class PredictionModel(object):
    """
    Models used to predict asset values based on past observations.
    """

    def __init__(self):
        pass

    def predict(self, samples):
        pass


class UnbiasEstimator(PredictionModel):

    def __init__(self):
        super(PredictionModel, UnbiasEstimator).__init__(self)

    def sample_mean(self, samples):
        return np.mean(samples, axis=0)

    def sample_covariance(self, samples):
        return np.cov(samples, rowvar=False)

    def predict(self, samples, assume_diag=True):
        covar = self.sample_covariance(samples)
        if assume_diag:
            covar = np.diag(np.diag(covar))
        return self.sample_mean(samples), covar


if __name__ == "__main__":
    from data_models import GaussianNoise

    num_samples = 1000
    num_assets = 3

    mu_truth = np.ones(num_assets)
    sigma_truth = np.diag([0.5, 0.3, 0.2])
    sampler = GaussianNoise()

    data = np.zeros(shape=(num_samples, num_assets))
    for i in range(num_samples):
        data[i] = sampler.sample((mu_truth, sigma_truth))

    sample_mean, sample_variance = UnbiasEstimator().predict(data)
    for i in range(num_assets):
        print(sample_mean[i], sample_variance[i])
