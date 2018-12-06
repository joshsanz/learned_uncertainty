import numpy as np


class DataModel(object):
    """
    Models used to generate asset values over time.
    """

    def sample(self, data):
        pass


class GaussianNoise(DataModel):
    """
    Samples are drawn from a multivariate normal distribution with given means and covariance
    """
    def __init__(self, seed=1):
        self.seed = seed
        self.r = np.random.RandomState(seed=self.seed)

    def sample(self, data):
        mu, sigma = data
        return self.r.multivariate_normal(mu, sigma)


if __name__ == "__main__":

    data = GaussianNoise()
    mu_truth = np.ones(3)
    sigma_truth = np.diag([0.5, 0.3, 0.2])
    mu, sigma = data.sample((mu_truth, sigma_truth))
    for i in range(mu.shape[0]):
        print(mu[i], sigma[i])
