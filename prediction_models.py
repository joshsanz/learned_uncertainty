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

    def predict(self, samples, assume_diag=True, p=1):
        projected_means = np.empty(shape=(p, samples.shape[1]))
        projected_covariance = np.empty(shape=(p, samples.shape[1], samples.shape[1]))
        projected_samples = samples
        for i in range(p):
            projected_means[i] = self.sample_mean(projected_samples)
            covar = self.sample_covariance(projected_samples)
            if covar.shape == ():
                covar = covar.reshape(1, 1)
            if assume_diag:
                covar = np.diag(np.diag(covar))
            projected_covariance[i, :, :] = covar
            projected_samples = np.vstack((projected_samples, projected_means[i]))

        if p == 1:
            return projected_means[0], projected_covariance[0]
        else:
            return projected_means, projected_covariance


class OneDimensionalAutoRegression(PredictionModel):

    def __init__(self, p, learning_rate=0.1, seed=1):
        super(PredictionModel, OneDimensionalAutoRegression).__init__(self)
        self.r = np.random.RandomState(seed=seed)
        self.w = None
        self.p = p
        self.samples = None
        self.grad_loss = ag.grad(self.loss, 0)
        self.learning_rate = learning_rate

    def loss(self, w, r, p):
        k = r.shape[0]
        sum_outer = 0
        for t in range(k):
            sum_inner = - r[t]
            for i in range(1, p):
                if i < t:
                    continue
                sum_inner += w[i] * r[t-i]
            sum_inner = sum_inner**2
            sum_outer += sum_inner
        return sum_outer

    def fit(self, samples):
        if len(samples.shape) != 1:
            assert len(samples.shape) == 2
            assert samples.shape[1] == 1
        self.samples = samples.flatten()
        self.w = self.r.rand(self.p)
        for i in range(10):
            self.w -= self.learning_rate*self.grad_loss(self.w, self.samples, self.p)
            print(self.loss(self.w, self.samples, self.p))

    #def toToepMat(x, L):
    #    mat = np.zeros(x.shape[0]-L, L)
    #    for i in range(L,x.shape[0]):
    #        
    #def fit(self, samples):
    
    #def predict(self, r):
    #    return np.dot(r, self.w)

    def predict(self, r, n):
        plen = self.w.shape[0]
        pred = np.zeros([n + plen])
        pred[0:plen] = r[-plen:]
        for i in range(n):
            pred[plen + i] = np.dot(pred[i:i+plen], self.w)
        return pred[-n:] 
        #for 

if __name__ == "__main__":
    num_samples = 100
    sine_samples = np.sin(np.linspace(0, 10 * np.pi, num_samples))
    reg_len = 10

    ar = OneDimensionalAutoRegression(reg_len)
    ar.fit(sine_samples)

    pred_sine = [0 for i in range(len(sine_samples))]
    pred_sine = ar.predict(sine_samples, 5) 
    import pdb; pdb.set_trace()
    plt.plot(np.arange(num_samples), sine_samples)
    plt.plot(np.arange(reg_len,num_samples), pred_sine)
    plt.savefig('autograd_test.png')
    #plt.show()

    # from data_models import GaussianNoise
    #
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
