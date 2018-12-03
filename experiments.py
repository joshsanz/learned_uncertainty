from data_models import *
from prediction_models import *
from control_models import *


# TODO: Add abstraction for experiments with code to generate plots.

def run_gaussian_mean_variance():
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

    cov_model = CovarianceModel(num_assets=num_assets)
    cov_model.run(data=(sample_mean, sample_covar), gamma=1.0)

    print(cov_model.variables())
    print(cov_model.optima())


def run_experiments():
    run_gaussian_mean_variance()


if __name__ == "__main__":
    run_experiments()
