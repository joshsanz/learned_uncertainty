import cvxpy as cvx
import numpy as np


class ControlModel(object):
    """
    Models used to find optimal investment strategies.
    """

    def __init__(self):
        pass

    def run(self, data):
        pass

    def optima(self):
        pass

    def variables(self):
        pass


class NormModel(ControlModel):

    def __init__(self, num_assets, gamma=1.0, regularization=1):
        super(ControlModel, CovarianceModel).__init__(self)
        self.num_assets = num_assets
        self.gamma = gamma
        self.regularization = regularization
        self.x = None
        self.problem = None
        self._optima = None

    def run(self, data):
        mu, sigma = data

        self.x = cvx.Variable(self.num_assets)

        objective = self.x.T*mu - self.gamma*cvx.norm(self.x, self.regularization)

        self.problem = cvx.Problem(cvx.Maximize(objective),
                           [
                               cvx.norm(self.x, 1) <= 1,
                               self.x >= 0
                           ])
        self._optima = self.problem.solve()

    def optima(self):
        return self._optima

    def variables(self):
        return self.x.value.flatten()


class CovarianceModel(ControlModel):

    def __init__(self, num_assets, gamma=1.0):
        super(ControlModel, CovarianceModel).__init__(self)
        self.num_assets = num_assets
        self.gamma = gamma
        self.x = None
        self.problem = None
        self._optima = None

    def run(self, data):
        mu, sigma = data

        self.x = cvx.Variable(self.num_assets)

        objective = self.x.T*mu - self.gamma*cvx.quad_form(self.x, sigma)

        self.problem = cvx.Problem(cvx.Maximize(objective),
                           [
                               cvx.norm(self.x, 1) <= 1,
                               self.x >= 0
                           ])
        self._optima = self.problem.solve()

    def optima(self):
        return self._optima

    def variables(self):
        return self.x.value.flatten()


class MultiPeriodModel(ControlModel):
    """
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.116.559&rep=rep1&type=pdf
    page 7
    """
    def __init__(self, num_assets, L, theta, nu):
        self.L = L # planning horizon
        self.theta = theta # safety margin on std dev
        self.nu = nu # transaction cost
        self.num_assets = num_assets
        self.R = None
        self.zeta = cvx.Variable((num_assets, L+1))
        self.xi = cvx.Variable((num_assets, L))
        self.eta = cvx.Variable((num_assets, L))
        self.omega = cvx.Variable()
        self.problem = None
        self._optima = None

    def run(self, data):
        # x0 n x 1 initial state of portfolio,
        # returns n x L expected return at each time step,
        # sigmas n x n x L variance at each time step
        x0, returns, _ = data
        self.R = np.cumprod(returns, axis=1)
        print("R:",self.R)
        objective = cvx.Maximize(self.omega)
        constraints = [self.omega <= self.R[:,self.L].T @ self.zeta[:,-1],
                       self.zeta >= 0, self.xi >= 0, self.eta >= 0,
                       self.zeta[:,0] == np.divide(x0, self.R[:,0])]
        A = (1 - self.nu) * self.R
        B = (1 + self.nu) * self.R
        for l in range(1, self.L + 1):
            # Equation 1.9
            constraints += [0 == -self.zeta[:,l] + self.zeta[:,l-1] - self.eta[:,l-1] + self.xi[:,l-1],
                            0 <= A[:,l-1].T @ self.eta[:,l-1] - B[:,l-1] @ self.xi[:,l-1]]
        self.problem = cvx.Problem(objective, constraints)
        self._optima = self.problem.solve()
        print(self.problem.status)

    def optima(self):
        return self._optima

    def variables(self):
        zeta = self.zeta.value
        eta = self.eta.value
        xi = self.xi.value
        R = self.R
        return zeta * R, eta * R[:,1:], xi * R[:, 1:]



if __name__ == "__main__":
    from data_models import GaussianNoise
    from prediction_models import UnbiasEstimator

    num_samples = 1000
    num_assets = 3

    mu_truth = np.ones(num_assets)
    sigma_truth = np.diag([0.5, 0.3, 0.2])

    sampler = GaussianNoise()
    data = np.zeros(shape=(num_samples, num_assets))

    for i in range(num_samples):
        data[i] = sampler.sample((mu_truth, sigma_truth))

    sample_mean, sample_covar = UnbiasEstimator().predict(data)

    for i in range(num_assets):
        print(sample_mean[i], sample_covar[i])

    # cov_model = CovarianceModel(num_assets=num_assets)
    # cov_model.run(data=(sample_mean, sample_covar), gamma=1.0)

    # print(cov_model.variables())
    # print(cov_model.optima())

    mpc = MultiPeriodModel(num_assets, 2, 2, .1)
    x0 = np.ones((num_assets,)) / num_assets
    sample_mean[0] = 1.1
    sample_mean[1] = 0.9
    means = np.repeat(sample_mean.reshape(-1,1), 3, 1)
    covs = np.repeat(sample_covar.reshape(-1,1), 3, 1)
    mpc.run(data=(x0, means, covs))

    x, y, z = mpc.variables()
    print("x:",x)
    print("y:",y)
    print("z:",z)
    print(mpc.optima())
