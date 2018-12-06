import numpy as np
import cvxpy as cvx
from matplotlib import pyplot as plt


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
        if covar.shape == ():
            covar = covar.reshape(1, 1)
        if assume_diag:
            covar = np.diag(np.diag(covar))
        return self.sample_mean(samples), covar


class AutoregressiveModel(PredictionModel):

    def __init__(self, order):
        super(PredictionModel, AutoregressiveModel).__init__(self)
        self.order = order

        self.w = cvx.Variable(order)

    def predict(self, samples):
        k = samples.shape[0]

        # Determine the weights in the autoregressive model.
        objective = 0.
        for t in range(k):
            s = 0.

            ub = min(self.order, t)
            for i in range(ub):
                s += self.w[i] * samples[t - ub + i]

            objective += cvx.power(s - samples[t], 2)

        problem = cvx.Problem(cvx.Minimize(objective))
        optim = problem.solve()
        # print("weights = {}".format(self.w.value))

        # Make the prediction using these weights.
        pred = 0.
        for i in range(self.order):
            pred += self.w[i].value * samples[-self.order + i]

        return pred


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

    # Simple example.
    # ar = AutoregressiveModel(2)
    # samples = np.array([2, 4, 8, 16, 32])

    ar = AutoregressiveModel(5)
    # Generate some samples.
    r = np.random.RandomState(seed=1)
    samples = []
    sample_ts = []
    preds = []
    pred_ts = []
    t = 0.
    dt = 0.25
    amplitude = 1.
    for k in range(100):
        samples.append(r.normal(loc=amplitude*np.sin(2*t), scale=0.1))
        sample_ts.append(t)

        if k > 10:
            # Just take the last few samples.
            pred = ar.predict(np.asarray(samples[-10:]))
            preds.append(pred)
            pred_ts.append(t + dt)

        t += dt

    plt.plot(sample_ts, samples, label='samples')
    plt.plot(pred_ts, preds, label='predictions')
    plt.legend(loc='best')
    plt.show()
