import cvxpy as cvx
import numpy as np

import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
plt.rc('figure', figsize=[10, 6])

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
    page 4
    """
    def __init__(self, num_assets, L, theta, nu):
        self.L = L # planning horizon
        self.theta = theta # safety margin on std dev
        self.nu = nu # transaction cost
        self.num_assets = num_assets
        self.R = None
        self.xi = cvx.Variable((num_assets+1, L+1))
        self.eta = cvx.Variable((num_assets+1, L))
        self.zeta = cvx.Variable((num_assets+1, L))
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
        constraints = [self.omega <= self.R[:,self.L].T @ self.xi[:,-1],
                       self.zeta >= 0, self.xi >= 0, self.eta >= 0,
                       self.xi[:,0] == np.divide(x0, self.R[:,0]),
                       self.xi[-1,1:] == 0]
        A = (1 - self.nu) * self.R
        B = (1 + self.nu) * self.R
        for l in range(1, self.L + 1):
            # Equation 1.9
            constraints += [0 == -self.xi[:,l] + self.xi[:,l-1] - self.eta[:,l-1] + self.zeta[:,l-1],
                            0 <= A[:,l-1].T @ self.eta[:,l-1] - B[:,l-1] @ self.zeta[:,l-1]]
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
        return xi * R, eta * R[:,1:], zeta * R[:, 1:]

class RobustMultiPeriodModel(ControlModel):
    """
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.116.559&rep=rep1&type=pdf
    page 15

    Note: Here we use theta as a scaling parameter on the variance, xi^T V xi,
    rather than on the standard deviation sqrt(xi^T V xi) to ease implementation in CVX
    """
    def __init__(self, num_assets, L, theta, nu):
        self.L = L # planning horizon
        self.theta = theta # safety margin on std dev
        self.nu = nu # transaction cost
        self.num_assets = num_assets
        self.R = None
        self.xi = cvx.Variable((num_assets+1, L+1))
        self.zeta = cvx.Variable((num_assets+1, L))
        self.eta = cvx.Variable((num_assets+1, L))
        self.omega = cvx.Variable()
        self.problem = None
        self._optima = None

    def run(self, data):
        # x0 n x 1 initial state of portfolio,
        # log_returns n x L expected log return at each time step,
        # sigmas n x L expected log variance at each time step
        x0, log_returns, log_vars = data

        # Expectations
        ExpR = np.exp(np.cumsum(log_returns + 0.5 * log_vars, axis=1))
        self.R = ExpR
        print("R:",self.R)
        pl = [ np.concatenate([(1-self.nu) * ExpR[:,l], -(1+self.nu) * ExpR[:,l]]) for l in range(self.L) ]
        pLp1 = ExpR[:,-1]

        # Covariances
        VarR = (np.exp(np.cumsum(2 * log_returns + log_vars, axis=1)) * np.exp(np.cumsum(log_vars, axis=1)) -
                np.exp(np.cumsum(2 * log_returns + log_vars, axis=1)))
        print("VarR:",VarR)
        print(VarR.shape)
        Cl = [ np.diag(VarR[:,l]) for l in range(1, self.L+1) ]
        Vl = [ np.bmat( [[(1 - self.nu) ** 2 * C, -(1-self.nu) * (1 + self.nu) * C],
                         [-(1-self.nu) * (1 + self.nu) * C, (1 + self.nu) ** 2 * C]] ) for C in Cl ]
        print(len(Vl))
        VLp1 = Cl[-1]


        objective = cvx.Maximize(self.omega)
        constraints = [self.omega <= pLp1 @ self.xi[:,self.L] - self.theta * cvx.quad_form(self.xi[:,-1], VLp1),
                       self.xi >= 0, self.eta >= 0, self.zeta >= 0,
                       self.xi[:,0] == np.divide(x0, self.R[:,0]),
                       self.xi[-1,1:] == 0]
        alpha = (1 - self.nu) * self.R
        beta = (1 + self.nu) * self.R
        for l in range(1, self.L + 1):
            # Equation 1.9
            constraints += [0 == -self.xi[:,l] + self.xi[:,l-1] - self.eta[:,l-1] + self.zeta[:,l-1],
                            0 <= (alpha[:,l-1].T @ self.eta[:,l-1] - beta[:,l-1] @ self.zeta[:,l-1] -
                                  self.theta * cvx.quad_form(cvx.bmat([[self.eta[:,l-1], self.zeta[:,l-1]]]).T, Vl[l-1]))
                            ]
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
        return xi * R, eta * R[:,1:], zeta * R[:, 1:]


# class MultiPeriodModelSimple(ControlModel):
#     """
#     Equation 1.5
#     """
#     def __init__(self, num_assets, L, mu, v):
#         self.L = L # planning horizon
#         self.mu = mu
#         self.v = v
#         self.num_assets = num_assets
#         self.x = cvx.Variable((num_assets, L + 1))
#         self.y = cvx.Variable((num_assets, L))
#         self.z = cvx.Variable((num_assets, L))
#         self.problem = None
#         self._optima = None
#
#     def run(self, data):
#         # TODO (hme): Finish imp.
#         x0, r, _ = data
#         assert r.shape == (self.num_assets, self.L + 1)
#
#         objective = cvx.Maximize(r[:, self.L].T * self.x[:, self.L])
#         constraints = [
#             self.x >= 0, self.z >= 0, self.y >= 0,
#        ]
#         for l in range(1, self.L + 1):
#             for i in range(1, self.num_assets - 1):
#                 # Equation 1.5
#                 constraints += [
#                     self.x[i, l] == r[i, l-1] @ self.x[i, l-1] - self.y[i, l] + self.z[i, l],
#                 ]
#         # self.x[n+1, l] <= self.x[:, l - 1] + A[:, l - 1].T @ self.y[:, l - 1] - B[:, l - 1] @ self.z[:, l - 1]
#
#         self.problem = cvx.Problem(objective, constraints)
#         self._optima = self.problem.solve()
#         print(self.problem.status)
#
#     def optima(self):
#         return self._optima
#
#     def variables(self):
#         x = self.x.value
#         y = self.y.value
#         z = self.z.value
#         return x, y, z


if __name__ == "__main__":
    from data_models import GaussianNoise, NoisySine
    from prediction_models import UnbiasGaussianEstimator, AutoRegression

    num_samples = 1000
    num_assets = 3

    # mu_truth = np.ones(num_assets)
    # sigma_truth = np.diag([0.5, 0.3, 0.2])
    #
    # sampler = GaussianNoise()
    # data = np.zeros(shape=(num_samples, num_assets))
    #
    # for i in range(num_samples):
    #     data[i] = sampler.sample((mu_truth, sigma_truth))
    #
    # sample_mean, sample_covar = UnbiasGaussianEstimator().predict(data)
    #
    # for i in range(num_assets):
    #     print(sample_mean[i], sample_covar[i])

    # mpc = MultiPeriodModel(num_assets, 2, 2, .1)
    # x0 = np.ones((num_assets,)) / num_assets
    # sample_mean[0] = 1.1
    # sample_mean[1] = 0.9
    # means = np.repeat(sample_mean.reshape(-1, 1), 3, 1)
    # covs = np.repeat(sample_covar.reshape(-1, 1), 3, 1)
    # mpc.run(data=(x0, means, covs))
    #
    # x, y, z = mpc.variables()
    # print("x:", x)
    # print("y:", y)
    # print("z:", z)
    # print(mpc.optima())
    #
    # cov_model = CovarianceModel(num_assets=num_assets)
    # cov_model.run(data=(sample_mean, sample_covar), gamma=1.0)

    # print(cov_model.variables())
    # print(cov_model.optima())

    data = NoisySine()
    phase = np.array([1., .5, 2.])
    noise = np.array([0.1, 0.03, 0.2])
    samples = data.sample((phase, noise, 20))
    for i in range(samples.shape[1]):
        print(samples.T[i])

    L = 5
    ar = AutoRegression(L)
    ar.fit(samples)
    ar_projections, ar_errors = ar.predict(samples, L)
    print("Projections:",ar_projections)
    print(ar_projections.shape)
    print("Errors:",ar_errors)
    print(ar_errors.shape)
    ar_variances = np.zeros((num_assets, L))
    ar_variances[:,1:] = np.repeat(ar_errors.reshape(-1,1), L-1, axis=1)

    projections = np.ones((num_assets+1, L))
    projections[:-1,:] = ar_projections.T
    print(projections)
    variances = np.zeros((num_assets+1, L))
    variances[:-1,:] = ar_variances
    print(variances)

    # Run models
    mpc = MultiPeriodModel(num_assets, L=4, theta=2, nu=.5)
    rmpc = RobustMultiPeriodModel(num_assets, L=4, theta=0, nu=.5)
    x0 = np.zeros((num_assets+1,))
    x0[-1] = 1.0

    mpc.run(data=(x0, projections, None))

    x, y, z = mpc.variables()
    print("x:",x)
    print("y:",y)
    print("z:",z)
    print(mpc.optima())

    rmpc.run(data=(x0, np.log(projections), variances))

    rx, ry, rz = rmpc.variables()
    print("rx:",rx)
    print("ry:",ry)
    print("rz:",rz)
    print(rmpc.optima())

    plt.plot(x.T,label='x')
    plt.plot(rx.T,':',label='rx')
    plt.legend()
    plt.show()

    plt.plot(y.T,label='y')
    plt.plot(ry.T,':',label='ry')
    plt.legend()
    plt.show()

    plt.plot(z.T,label='z')
    plt.plot(rz.T,':',label='rz')
    plt.legend()
    plt.show()
