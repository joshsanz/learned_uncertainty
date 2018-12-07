import numpy as np
import pickle


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


class NoisySine(DataModel):
    """
    Samples are drawn from a multivariate normal distribution with given means and covariance
    """
    def __init__(self, seed=1):
        self.seed = seed
        self.r = np.random.RandomState(seed=self.seed)

    def sample(self, data):
        phases, noise, num_samples = data
        result = np.zeros(shape=(num_samples, phases.shape[0]))
        for i in range(len(phases)):
            result[:, i] = np.sin(np.linspace(phases[i], 2*np.pi, num_samples)) + self.r.normal(0, noise[i])
        return result


class RealData(object):
    """
    Real data pulled from the web. 8 tech assets.
    """

    def sample(self, compute_returns=True):
        with open("./alpha_vantage_data/data/real_data.pickle", "rb") as fh:
            data = pickle.load(fh)

        if compute_returns:
            print(data.shape)
            r = np.ones_like(data)
            # data = timestep x asset
            for j in range(r.shape[1]):
                for i in range(1, r.shape[0]):
                    r[i][j] = data[i][j]/data[i-1][j]
            return r
        return data

    def labels(self):
        with open("./alpha_vantage_data/data/real_data_symbols.pickle", "rb") as fh:
            data_labels = pickle.load(fh)
        return data_labels

    def dates(self):
        with open("./alpha_vantage_data/data/real_data_dates.pickle", "rb") as fh:
            data_dates = pickle.load(fh)
        return data_dates


if __name__ == "__main__":

    # data = GaussianNoise()
    # mu_truth = np.ones(3)
    # sigma_truth = np.diag([0.5, 0.3, 0.2])
    # mu, sigma = data.sample((mu_truth, sigma_truth))
    # for i in range(mu.shape[0]):
    #     print(mu[i], sigma[i])

    # sampler = RealData()
    # data = sampler.sample()
    # labels = sampler.labels()
    # for i in range(data.T.shape[0]):
    #     print(labels[i], data.T[i])

    data = NoisySine()
    phase = np.array([1., .5, 2.])
    noise = np.array([0.5, 0.3, 0.2])
    samples = data.sample((phase, noise, 20))
    for i in range(samples.shape[1]):
        print(samples.T[i])
