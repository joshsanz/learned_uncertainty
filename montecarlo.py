import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt

import time
import numpy as np
import multiprocessing as mproc
NCPU = mproc.cpu_count()
print("Running on {} cores".format(NCPU))

from experiments import run_simple_gaussian_experiments, run_ltv_gaussian_experiments, run_wiener_experiments


def mp_simple_gauss(args):
    print("Doing a simple gauss run")
    return run_simple_gaussian_experiments(args, False, np.random.randint(0, 1e9))


def mp_ltv_gauss(args):
    print("Doing a LTV gauss run")
    return run_ltv_gaussian_experiments(args, False, np.random.randint(0, 1e9))


def mp_wiener(args):
    print("Doing a Wiener run")
    return run_wiener_experiments(args, False, np.random.randint(0, 1e9))


def construct_params(val, cov, delta, gamma):
    return {'asset_value': val, 'asset_covariance': cov, 'asset_delta': delta, 'gamma': gamma}


def main(nassets, nsimulations):
    pool = mproc.Pool(NCPU)

    print("Building parameter sets")

    val_mean = np.ones((nassets,))
    val_cov = np.diag(np.ones((nassets,)) * 0.01)
    values = [np.random.multivariate_normal(val_mean, val_cov) for i in range(nsimulations)]

    cov_mean = np.zeros((nassets,))
    cov_cov = np.diag(np.ones((nassets,)) * 0.01)
    covariances = [np.diag(np.abs(np.random.multivariate_normal(cov_mean, cov_cov))) for i in range(nsimulations)]
    print(covariances[0])

    delta_mean = np.zeros((nassets,))
    delta_cov = np.diag(np.ones((nassets,)) * 0.001)
    deltas = [np.random.multivariate_normal(delta_mean, delta_cov) for i in range(nsimulations)]

    gammas = [0.5 for i in range(nsimulations)]

    params = [construct_params(values[i], covariances[i], deltas[i], gammas[i]) for i in range(nsimulations)]

    print("Starting runs")

    results = list(pool.imap_unordered(mp_simple_gauss, params))

    print(results[0])


if __name__ == "__main__":
    main(3, 100)