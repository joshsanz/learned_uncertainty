import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
plt.rc('figure', figsize=[10, 6])

import time

from data_models import *
from prediction_models import *
from control_models import *


def error(predicted_return, true_return):
    return (predicted_return - true_return)


def get_gaussian_data(num_samples, true_asset_value, asset_covariance, seed=1):
    num_assets = asset_covariance.shape[0]
    sampler = GaussianNoise(seed)
    data = np.zeros(shape=(num_samples, num_assets))
    for t in range(num_samples):
        sampler_input = (true_asset_value, asset_covariance)
        data[t] = sampler.sample(sampler_input)
    return data


def get_wiener_data(num_samples, true_asset_value, asset_covariance, seed=1):
    num_assets = asset_covariance.shape[0]
    steps = get_gaussian_data(num_samples, np.zeros((num_assets,)), asset_covariance, seed)
    return np.cumsum(steps, axis=0) + true_asset_value


def get_real_data():
    sampler = RealData()
    return sampler.labels(), sampler.dates(), sampler.sample()


def get_returns(data, investment_strategies, asset_predictions):
    augmented_data = np.hstack((data, np.ones((data.shape[0], 1))))
    augmented_predictions = np.hstack((asset_predictions, np.ones((asset_predictions.shape[0], 1))))

    num_samples = investment_strategies.shape[0]
    predicted_return = np.zeros(shape=(num_samples+1,))
    true_return = np.zeros(shape=(num_samples+1,))
    for t in range(num_samples):
        if t <= 2:
            continue
        observed_asset_value = augmented_data[t]
        predicted_asset_value = augmented_predictions[t]
        investment_strategy = investment_strategies[t]
        true_return[t] = investment_strategy.dot(observed_asset_value)
        predicted_return[t] = investment_strategy.dot(predicted_asset_value)
    return predicted_return, true_return


def run_gaussian_norm(data, num_samples, num_assets, pred_params, control_params):
    gamma = control_params['gamma']
    regularization = control_params['regularization']

    prediction_model = UnbiasGaussianEstimator()
    window = pred_params['window']
    cov_model = NormModel(num_assets=num_assets, gamma=gamma, regularization=regularization)

    predicted_asset_values = np.zeros(shape=(num_samples, num_assets))
    investment_strategies = np.zeros(shape=(num_samples, num_assets))
    for t in range(num_samples):
        if t <= 2:
            continue
        if window is None:
            past_data = data[:t]
        else:
            past_data = data[max(0, t-window):t]
        predicted_asset_value, predicted_asset_variance = prediction_model.predict(past_data)
        predicted_asset_values[t] = predicted_asset_value

        control_input = (predicted_asset_value, predicted_asset_variance)
        cov_model.run(control_input)
        investment_strategy = cov_model.variables()
        investment_strategies[t] = investment_strategy

    return predicted_asset_values, investment_strategies


def run_gaussian_covar(data, num_samples, num_assets, pred_params, control_params):
    gamma = control_params['gamma']

    prediction_model = UnbiasGaussianEstimator()
    window = pred_params['window']
    cov_model = CovarianceModel(num_assets=num_assets, gamma=gamma)

    predicted_asset_values = np.zeros(shape=(num_samples, num_assets))
    investment_strategies = np.zeros(shape=(num_samples, num_assets))
    for t in range(num_samples):
        if t <= 2:
            continue
        if window is None:
            past_data = data[:t]
        else:
            past_data = data[max(0, t-window):t]
        predicted_asset_value, predicted_asset_variance = prediction_model.predict(past_data)
        predicted_asset_values[t] = predicted_asset_value

        control_input = (predicted_asset_value, predicted_asset_variance)
        cov_model.run(control_input)
        investment_strategy = cov_model.variables()
        investment_strategies[t] = investment_strategy

    return predicted_asset_values, investment_strategies


def run_real_ar_mpc(data, num_samples, num_assets, pred_params, control_params):

    prediction_model = AutoRegression(**pred_params)
    L = control_params["L"]

    mpm_params = control_params.copy()
    mpm_params["num_assets"] = num_assets
    mpm_params["L"] = L-1
    mpm = MultiPeriodModel(**mpm_params)

    predicted_returns = np.zeros(shape=(num_samples, num_assets))
    investment_strategies = np.zeros(shape=(num_samples, num_assets+1))
    current_investments = np.zeros(shape=(num_assets + 1,))
    current_investments[-1] = 1.0
    for t in range(num_samples):
        if t <= L+20:
            continue
        print("running timestep", t)
        past_data = data[:t]
        prediction_model.fit(past_data)
        predicted_return, predicted_return_error = prediction_model.predict(past_data, L)
        control_inputs = mpm.get_input(L, past_data, predicted_return, predicted_return_error)

        control_inputs = (current_investments, control_inputs[0], control_inputs[1])
        mpm.run(control_inputs)

        current_investments = mpm.apply_model_results(current_investments, mpm.variables())
        investment_strategies[t] = current_investments

    return predicted_returns, investment_strategies


def run_simple_gaussian_experiments(params, real_data=False, plot=False, seed=1):
    if not real_data:
        num_samples = 100
        true_asset_value = params['asset_value']
        asset_covariance = params['asset_covariance']
        data = get_gaussian_data(num_samples, true_asset_value, asset_covariance, seed)
        data = np.clip(data, 1e-3, None)
    else:
        data_labels, data_dates, data = get_real_data()
        print("date range:", data_dates[0][0], "-", data_dates[0][-1])
        num_samples = data.shape[0]

    gamma = params['gamma']
    window = params['window']
    num_assets = data.shape[1]

    if plot:
        if real_data:
            for i in range(num_assets):
                plt.plot(data.T[i], label=data_labels[i])
        else:
            plt.plot(data, label='Asset Values')
        plt.legend()
        plt.title('Input Data')
        plt.show()

    # Add experiments to run here.
    experiments = [
        ("gaussian_unbiased_covar", run_gaussian_covar, {'window': None}, {"gamma": gamma}),
        ("gaussian_unbiased_l1", run_gaussian_norm, {'window': None}, {"gamma": gamma, "regularization": 1}),
        ("gaussian_unbiased_l2", run_gaussian_norm, {'window': None}, {"gamma": gamma, "regularization": 2}),
        ("gaussian_windowed_covar", run_gaussian_covar, {'window': window}, {"gamma": gamma}),
        ("gaussian_windowed_l1", run_gaussian_norm, {'window': window}, {"gamma": gamma, "regularization": 1}),
        ("gaussian_windowed_l2", run_gaussian_norm, {'window': window}, {"gamma": gamma, "regularization": 2}),
    ]

    bar_plot_mean = []
    bar_plot_std = []

    results = {}
    results['true_values'] = data
    for name, experiment_func, pred_params, control_params in experiments:
        predicted_asset_values, investment_strategies = experiment_func(data,
                                                                        num_samples,
                                                                        num_assets,
                                                                        pred_params,
                                                                        control_params)
        predicted_return, true_return = get_returns(data, investment_strategies, predicted_asset_values)
        results[name] = {}
        results[name]['predicted_return'] = predicted_return
        results[name]['strategies'] = investment_strategies
        results[name]['predicted_values'] = predicted_asset_values
        results[name]['true_return'] = true_return
        print(name, np.sum(true_return))
        bar_plot_mean.append(np.mean(true_return))
        bar_plot_std.append(np.std(true_return))
        # all_error = error(predicted_return, true_return)
        # window = 10
        # for i in range(0, num_samples-window, window):
        #     print(name, np.mean(all_error[i:i + window]))
        if plot:
            # We really just care about how well the investment strategies actually do,
            # which is given by true_return.
            plt.plot(np.arange(3, num_samples), true_return[3:], label=name + ' true return', alpha=0.5)
            # In final plots, predicted return may not be relevant.
            # plt.plot(np.arange(3, num_samples), predicted_return[3:], label=name + ' predicted return')
    if plot:
        plt.legend()
        plt.show()

        plt.bar(np.arange(len(experiments)), height=bar_plot_mean, yerr=bar_plot_std)
        plt.show()

    return results


def run_ltv_gaussian_experiments(params, plot=False, seed=1):
    num_samples = 100
    true_asset_v0 = params['asset_value']
    true_asset_delta = params['asset_delta']
    asset_covariance = params['asset_covariance']
    gamma = params['gamma']
    window = params['window']
    true_asset_value = true_asset_v0 + (true_asset_delta.T @ np.arange(0,num_samples).reshape(-1,1).T).T
    data = get_gaussian_data(num_samples, np.zeros((3,)), asset_covariance, seed) + true_asset_value
    data = np.clip(data, 1e-3, None)
    num_assets = data.shape[1]

    if plot:
        plt.plot(data, label='Asset Values')
        plt.legend()
        plt.title('Input Data')
        plt.show()

    # Add experiments to run here.
    experiments = [
        ("gaussian_unbiased_covar", run_gaussian_covar, {'window': None}, {"gamma": gamma}),
        ("gaussian_unbiased_l1", run_gaussian_norm, {'window': None}, {"gamma": gamma, "regularization": 1}),
        ("gaussian_unbiased_l2", run_gaussian_norm, {'window': None}, {"gamma": gamma, "regularization": 2}),
        ("gaussian_windowed_covar", run_gaussian_covar, {'window': window}, {"gamma": gamma}),
        ("gaussian_windowed_l1", run_gaussian_norm, {'window': window}, {"gamma": gamma, "regularization": 1}),
        ("gaussian_windowed_l2", run_gaussian_norm, {'window': window}, {"gamma": gamma, "regularization": 2}),
    ]

    bar_plot_mean = []
    bar_plot_std = []

    results = {}
    results['true_values'] = data
    for name, experiment_func, pred_params, control_params in experiments:
        predicted_asset_values, investment_strategies = experiment_func(data,
                                                                        num_samples,
                                                                        num_assets,
                                                                        pred_params,
                                                                        control_params)
        predicted_return, true_return = get_returns(data, investment_strategies, predicted_asset_values)
        results[name] = {}
        results[name]['predicted_return'] = predicted_return
        results[name]['strategies'] = investment_strategies
        results[name]['predicted_values'] = predicted_asset_values
        results[name]['true_return'] = true_return
        print(name, np.sum(true_return))
        bar_plot_mean.append(np.mean(true_return))
        bar_plot_std.append(np.std(true_return))
        # all_error = error(predicted_return, true_return)
        # window = 10
        # for i in range(0, num_samples-window, window):
        #     print(name, np.mean(all_error[i:i + window]))
        if plot:
            # We really just care about how well the investment strategies actually do,
            # which is given by true_return.
            plt.plot(np.arange(3, num_samples), true_return[3:], label=name + ' true return', alpha=0.33)
            # In final plots, predicted return may not be relevant.
            plt.plot(np.arange(3, num_samples), predicted_return[3:], label=name + ' predicted return')
    if plot:
        plt.legend()
        plt.show()

        plt.bar(np.arange(len(experiments)), height=bar_plot_mean, yerr=bar_plot_std)
        plt.show()

    return results


def run_wiener_experiments(params, plot=False, seed=1):
    num_samples = 100
    true_asset_v0 = params['asset_value']
    asset_covariance = params['asset_covariance']
    gamma = params['gamma']
    window = params['window']
    data = get_wiener_data(num_samples, true_asset_v0, asset_covariance, seed)
    data = np.clip(data, 1e-3, None)
    num_assets = data.shape[1]

    if plot:
        plt.plot(data, label='Asset Values')
        plt.legend()
        plt.title('Input Data')
        plt.show()

    # Add experiments to run here.
    experiments = [
        ("gaussian_unbiased_covar", run_gaussian_covar, {'window': None}, {"gamma": gamma}),
        ("gaussian_unbiased_l1", run_gaussian_norm, {'window': None}, {"gamma": gamma, "regularization": 1}),
        ("gaussian_unbiased_l2", run_gaussian_norm, {'window': None}, {"gamma": gamma, "regularization": 2}),
        ("gaussian_windowed_covar", run_gaussian_covar, {'window': window}, {"gamma": gamma}),
        ("gaussian_windowed_l1", run_gaussian_norm, {'window': window}, {"gamma": gamma, "regularization": 1}),
        ("gaussian_windowed_l2", run_gaussian_norm, {'window': window}, {"gamma": gamma, "regularization": 2}),
    ]

    bar_plot_mean = []
    bar_plot_std = []
    results = {}
    results['true_values'] = data
    for name, experiment_func, pred_params, control_params in experiments:
        predicted_asset_values, investment_strategies = experiment_func(data,
                                                                        num_samples,
                                                                        num_assets,
                                                                        pred_params,
                                                                        control_params)
        predicted_return, true_return = get_returns(data, investment_strategies, predicted_asset_values)
        results[name] = {}
        results[name]['predicted_return'] = predicted_return
        results[name]['strategies'] = investment_strategies
        results[name]['predicted_values'] = predicted_asset_values
        results[name]['true_return'] = true_return
        print(name, np.sum(true_return))
        bar_plot_mean.append(np.mean(true_return))
        bar_plot_std.append(np.std(true_return))
        # all_error = error(predicted_return, true_return)
        # window = 10
        # for i in range(0, num_samples-window, window):
        #     print(name, np.mean(all_error[i:i + window]))
        if plot:
            # We really just care about how well the investment strategies actually do,
            # which is given by true_return.
            plt.plot(np.arange(3, num_samples), true_return[3:], label=name + ' true return', alpha=0.33)
            # In final plots, predicted return may not be relevant.
            plt.plot(np.arange(3, num_samples), predicted_return[3:], label=name + ' predicted return')
    if plot:
        plt.legend()
        plt.show()

        plt.bar(np.arange(len(experiments)), height=bar_plot_mean, yerr=bar_plot_std)
        plt.show()

    return results


def run_real_mpc_experiments(pred_params, control_params, plot=False, seed=1):
    data_labels, data_dates, data = get_real_data()
    print("date range:", data_dates[0][0], "-", data_dates[0][-1])
    num_samples = data.shape[0]
    num_assets = data.shape[1]

    if plot:
        for i in range(num_assets):
            plt.plot(data.T[i], label=data_labels[i])
        plt.legend()
        plt.title('Input Data')
        plt.show()

    # Add experiments to run here.
    experiments = [
        ("run_real_ar_mpc", run_real_ar_mpc, pred_params, control_params),
    ]

    bar_plot_mean = []
    bar_plot_std = []

    results = {}
    results['true_values'] = data
    for name, experiment_func, pred_params, control_params in experiments:
        predicted_asset_values, investment_strategies = experiment_func(data,
                                                                        num_samples,
                                                                        num_assets,
                                                                        pred_params,
                                                                        control_params)
        predicted_return, true_return = get_returns(data, investment_strategies, predicted_asset_values)
        results[name] = {}
        results[name]['predicted_return'] = predicted_return
        results[name]['strategies'] = investment_strategies
        results[name]['predicted_values'] = predicted_asset_values
        results[name]['true_return'] = true_return
        print(name, np.sum(true_return))
        bar_plot_mean.append(np.mean(true_return))
        bar_plot_std.append(np.std(true_return))
        # all_error = error(predicted_return, true_return)
        # window = 10
        # for i in range(0, num_samples-window, window):
        #     print(name, np.mean(all_error[i:i + window]))
        if plot:
            # We really just care about how well the investment strategies actually do,
            # which is given by true_return.
            plt.plot(np.arange(3, num_samples), true_return[3:], label=name + ' true return', alpha=0.5)
            # In final plots, predicted return may not be relevant.
            # plt.plot(np.arange(3, num_samples), predicted_return[3:], label=name + ' predicted return')
    if plot:
        plt.legend()
        plt.show()

        plt.bar(np.arange(len(experiments)), height=bar_plot_mean, yerr=bar_plot_std)
        plt.show()

    return results


if __name__ == "__main__":
    # run_simple_gaussian_experiments(params={'gamma': 1,
    #                                         'window': 10},
    #                                 real_data=True,
    #                                 plot=True, seed=int(time.time()))
    # run_simple_gaussian_experiments(params={'asset_value': np.array([0.8, 1.0, 1.1]),
    #                                         'asset_covariance': np.diag([0.02, 0.01, 0.03]),
    #                                         'gamma': 1,
    #                                         'window': 10},
    #                                 plot=True, seed=int(time.time()))
    # run_ltv_gaussian_experiments(params={'asset_value': np.array([0.9, 1.2, 1.0]),
    #                                      'asset_covariance': np.diag([1.0, 1.0, 0.2]) * 0.02,
    #                                      'asset_delta': np.array([[0.002, -0.003, 0.001]]),
    #                                      'gamma': 1,
    #                                      'window': 10},
    #                              plot=True, seed=int(time.time()))
    # run_wiener_experiments(params={'asset_value': np.array([0.9, 1.2, 1.0]),
    #                                'asset_covariance': np.diag([1.0, 1.0, 0.2]) * 0.02,
    #                                'gamma': 1,
    #                                'window': 10},
    #                        plot=True, seed=int(time.time()))

    run_real_mpc_experiments(pred_params={"p": 10,
                                          "regularizer": 0.001},
                             control_params={"L": 10,
                                             "theta": 2.0,
                                             "nu": 0.1},
                             plot=False, seed=int(time.time()))
