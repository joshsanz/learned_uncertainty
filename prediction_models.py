import autograd as ag
import autograd.numpy as np
import matplotlib.pyplot as plt


class PredictionModel(object):
    """
    Models used to predict asset values based on past observations.
    """

    def __init__(self):
        pass

    def predict(self, samples):
        pass


class UnbiasGaussianEstimator(PredictionModel):

    def __init__(self):
        super(PredictionModel, UnbiasGaussianEstimator).__init__(self)

    def sample_mean(self, samples):
        return np.mean(samples, axis=0)

    def sample_covariance(self, samples):
        return np.cov(samples, rowvar=False)

    def predict(self, samples, assume_diag=True, n=1):
        projected_means = np.empty(shape=(n, samples.shape[1]))
        projected_covariance = np.empty(shape=(n, samples.shape[1], samples.shape[1]))
        projected_samples = samples
        for i in range(n):
            projected_means[i] = self.sample_mean(projected_samples)
            covar = self.sample_covariance(projected_samples)
            if covar.shape == ():
                covar = covar.reshape(1, 1)
            if assume_diag:
                covar = np.diag(np.diag(covar))
            projected_covariance[i, :, :] = covar
            projected_samples = np.vstack((projected_samples, projected_means[i]))

        if n == 1:
            return projected_means[0], projected_covariance[0]
        else:
            return projected_means, projected_covariance


class AutoRegression(PredictionModel):

    def __init__(self, p, regularizer=0.001):
        self.p = p
        self.data = None
        self.models = None
        self.regularizer = regularizer

    def fit(self, samples):
        self.data = samples.T
        self.models = []
        for i in range(self.data.shape[0]):
            model = OneDimensionalAutoRegression(self.p, self.regularizer)
            model.fit(self.data[i])
            self.models.append(model)

    def predict(self, samples, n):
        predictions = np.empty(shape=(n, samples.shape[1]))
        for i in range(samples.shape[1]):
            predictions[:, i] = self.models[i].predict(samples.T[i], n)
        return predictions


class OneDimensionalAutoRegression(object):

    def __init__(self, p, regularizer=0.001):
        self.w = None
        self.p = p
        self.regularizer = regularizer

    def fit(self, samples):
        assert(samples.shape[0] > self.p), "Number of samples are less than autoregression length"
        X = np.zeros([samples.shape[0]-self.p, self.p])
        self.regularizer = 0.001
        for i in range(samples.shape[0] - self.p):
            X[i] = samples[i:i+self.p]
        y = samples[self.p:]
        self.w = np.matmul(\
                np.linalg.inv(np.matmul(X.transpose(),X) + self.regularizer * np.identity(X.shape[1])),\
                np.matmul(X.transpose(),y))

    def predict(self, r, n):
        plen = self.w.shape[0]
        pred = np.zeros([n + plen])
        pred[0:plen] = r[-plen:]
        for i in range(n):
            pred[plen + i] = np.dot(pred[i:i+plen], self.w)
        return pred[-n:]

if __name__ == "__main__":
    from data_models import GaussianNoise

    # num_samples = 100
    # sine_samples = np.sin(np.linspace(0, 10 * np.pi, num_samples))
    # reg_len = 20
    #
    # ar = OneDimensionalAutoRegression(reg_len)
    # ar.fit(sine_samples)
    #
    # num_project = 100
    # pred_sine = ar.predict(sine_samples, num_project)
    # plt.plot(np.arange(num_samples), sine_samples)
    # plt.plot(np.arange(num_samples-1, num_samples+num_project-1), pred_sine, linewidth = 10, alpha = 0.5)
    # plt.show()

    num_samples = 1000
    num_assets = 3

    mu_truth = np.ones(num_assets)
    sigma_truth = np.diag([0.5, 0.3, 0.2])
    sampler = GaussianNoise()

    data = np.zeros(shape=(num_samples, num_assets))
    for i in range(num_samples):
        data[i] = sampler.sample((mu_truth, sigma_truth))

    L = 3
    ar = AutoRegression(L)
    ar.fit(data)
    prediction = ar.predict(data, L)
    for l in range(L):
        print("\n prediction", l, prediction[l])

    # num_samples = 1000
    # num_assets = 3
    #
    # mu_truth = np.ones(num_assets)
    # sigma_truth = np.diag([0.5, 0.3, 0.2])
    # sampler = GaussianNoise()
    #
    # data = np.zeros(shape=(num_samples, num_assets))
    # for i in range(num_samples):
    #     data[i] = sampler.sample((mu_truth, sigma_truth))
    #
    # L = 3
    # sample_mean, sample_variance = UnbiasGaussianEstimator().predict(data, True, L)
    # for l in range(L):
    #     print("\n projection", l)
    #     for i in range(num_assets):
    #         print(sample_mean[l][i], sample_variance[l][i])
