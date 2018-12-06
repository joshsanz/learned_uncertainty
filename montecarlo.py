import pickle
import time
import numpy as np
import multiprocessing as mproc
NCPU = mproc.cpu_count()
print("Running on {} cores".format(NCPU))

from experiments import run_simple_gaussian_experiments, run_ltv_gaussian_experiments, run_wiener_experiments


def mp_simple_gauss(args):
    return run_simple_gaussian_experiments(args, False, np.random.randint(0, 1e9))


def mp_ltv_gauss(args):
    return run_ltv_gaussian_experiments(args, False, np.random.randint(0, 1e9))


def mp_wiener(args):
    return run_wiener_experiments(args, False, np.random.randint(0, 1e9))


def construct_params(val, cov, delta, gamma, window):
    return {'asset_value': val, 'asset_covariance': cov, 'asset_delta': delta, 'gamma': gamma, 'window': window}


def master_mp_run(func, nassets, nsimulations):
    with mproc.Pool(NCPU) as pool:
        print("Building parameter sets")

        val_mean = np.ones((nassets,))
        val_cov = np.diag(np.ones((nassets,)) * 0.01)
        values = [np.random.multivariate_normal(val_mean, val_cov) for i in range(nsimulations)]

        cov_mean = np.zeros((nassets,))
        cov_cov = np.diag(np.ones((nassets,)) * 0.005)
        covariances = [np.diag(np.abs(np.random.multivariate_normal(cov_mean, cov_cov))) for i in range(nsimulations)]

        delta_mean = np.zeros((nassets,))
        delta_cov = np.diag(np.ones((nassets,)) * 0.001)
        deltas = [np.random.multivariate_normal(delta_mean, delta_cov).reshape(1, -1) for i in range(nsimulations)]

        gammas = [0.5 for i in range(nsimulations)]
        windows = [10 for i in range(nsimulations)]

        params = [construct_params(values[i], covariances[i], deltas[i], gammas[i], windows[i]) for i in range(nsimulations)]

        print("Starting runs")
        results = list(pool.imap_unordered(func, params))
        print("Finished runs")

        return results


def main(nassets, nsimulations):
    ltv_gauss = master_mp_run(mp_ltv_gauss, nassets, nsimulations)
    with open("ltv_gauss.pkl", "wb") as f:
        pickle.dump(ltv_gauss, f, -1)
    wiener = master_mp_run(mp_wiener, nassets, nsimulations)
    with open("wiener.pkl", "wb") as f:
        pickle.dump(wiener, f, -1)
    simple_gauss = master_mp_run(mp_simple_gauss, nassets, nsimulations)
    with open("simple_gauss.pkl", "wb") as f:
        pickle.dump(simple_gauss, f, -1)

if __name__ == "__main__":
    main(3, 100)