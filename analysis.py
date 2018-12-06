import pickle
import numpy as np

import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
plt.rc('figure', figsize=[10, 6])


def plot_residuals(runs, model_name):
    # Plot residual of predicted vs actual return
    guc = np.zeros((100,))
    gu1 = np.zeros((100,))
    gu2 = np.zeros((100,))
    gwc = np.zeros((100,))
    gw1 = np.zeros((100,))
    gw2 = np.zeros((100,))
    for r in runs:
        guc += r['gaussian_unbiased_covar']['predicted_return'] - r['gaussian_unbiased_covar']['true_return']
        gu1 += r['gaussian_unbiased_l1']['predicted_return'] - r['gaussian_unbiased_l1']['true_return']
        gu2 += r['gaussian_unbiased_l2']['predicted_return'] - r['gaussian_unbiased_l2']['true_return']
        gwc += r['gaussian_windowed_covar']['predicted_return'] - r['gaussian_windowed_covar']['true_return']
        gw1 += r['gaussian_windowed_l1']['predicted_return'] - r['gaussian_windowed_l1']['true_return']
        gw2 += r['gaussian_windowed_l2']['predicted_return'] - r['gaussian_windowed_l2']['true_return']
        plt.plot(r['gaussian_unbiased_covar']['predicted_return'] - r['gaussian_unbiased_covar']['true_return'], label='gaussian_unbiased_covar', alpha=0.25)
        plt.plot(r['gaussian_unbiased_l1']['predicted_return'] - r['gaussian_unbiased_l1']['true_return'], label='gaussian_unbiased_l1', alpha=0.25)
        plt.plot(r['gaussian_unbiased_l2']['predicted_return'] - r['gaussian_unbiased_l2']['true_return'], label='gaussian_unbiased_l2', alpha=0.25)
        plt.plot(r['gaussian_windowed_covar']['predicted_return'] - r['gaussian_windowed_covar']['true_return'], label='gaussian_windowed_covar', alpha=0.25)
        plt.plot(r['gaussian_windowed_l1']['predicted_return'] - r['gaussian_windowed_l1']['true_return'], label='gaussian_windowed_l1', alpha=0.25)
        plt.plot(r['gaussian_windowed_l2']['predicted_return'] - r['gaussian_windowed_l2']['true_return'], label='gaussian_windowed_l2', alpha=0.25)
    plt.ylabel("Predicted - Actual Return")
    plt.xlabel("Time Step")
    plt.title("Return Residual for {} Model".format(model_name))
    plt.savefig("out/return_residual-{}_model.png".format(model_name))
    plt.close()

    plt.plot(guc/len(runs), label='gaussian_unbiased_covar', alpha=0.5)
    plt.plot(gu1/len(runs), label='gaussian_unbiased_l1', alpha=0.5)
    plt.plot(gu2/len(runs), label='gaussian_unbiased_l2', alpha=0.5)
    plt.plot(gwc/len(runs), label='gaussian_windowed_covar', alpha=0.5)
    plt.plot(gw1/len(runs), label='gaussian_windowed_l1', alpha=0.5)
    plt.plot(gw2/len(runs), label='gaussian_windowed_l2', alpha=0.5)
    plt.legend()
    plt.ylabel("Mean Predicted - Actual Return")
    plt.xlabel("Time Step")
    plt.title("Mean Return Residual for {} Model".format(model_name))
    plt.savefig("out/mean_return_residual-{}_model.png".format(model_name))
    plt.show()


def plot_runs(runs, model_name):
    plot_residuals(runs, model_name)
    # plot_gamma_sweep(runs, model_name)


def main():
    with open("base_run/simple_gauss.pkl", 'rb') as f:
        simple_gauss = pickle.load(f)
    plot_residuals(simple_gauss, 'Gaussian')
    with open("base_run/ltv_gauss.pkl", 'rb') as f:
        ltv_gauss = pickle.load(f)
    plot_residuals(ltv_gauss, 'Trending-Gaussian')
    with open("base_run/wiener.pkl", 'rb') as f:
        wiener = pickle.load(f)
    plot_residuals(wiener, 'Wiener-Process')


if __name__ == "__main__":
    main()