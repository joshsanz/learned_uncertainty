from matplotlib import pyplot as plt

from data_models import *
from prediction_models import *
from control_models import *


def error(predicted_return, true_return):
    return (predicted_return - true_return)


def get_data(num_samples, num_assets):
    true_asset_value = np.array([0.9, 1.0, 1.1])
    asset_covariance = np.diag([0.3, 0.2, 0.1])
    sampler = GaussianNoise()
    data = np.zeros(shape=(num_samples, num_assets))
    for t in range(num_samples):
        sampler_input = (true_asset_value, asset_covariance)
        data[t] = sampler.sample(sampler_input)
    return data


def get_returns(data, investment_strategies, asset_predictions):
    num_samples = investment_strategies.shape[0]
    predicted_return = np.zeros(shape=(num_samples,))
    true_return = np.zeros(shape=(num_samples,))
    for t in range(num_samples):
        if t <= 2:
            continue
        observed_asset_value = data[t]
        predicted_asset_value = asset_predictions[t]
        investment_strategy = investment_strategies[t]
        true_return[t] = investment_strategy.dot(observed_asset_value)
        predicted_return[t] = investment_strategy.dot(predicted_asset_value)
    return predicted_return, true_return


def run_gaussian_mean_variance(data, num_samples, num_assets):
    gamma = 1.0

    prediction_model = UnbiasEstimator()
    cov_model = CovarianceModel(num_assets=num_assets)

    predicted_asset_values = np.zeros(shape=(num_samples, num_assets))
    investment_strategies = np.zeros(shape=(num_samples, num_assets))
    for t in range(num_samples):
        if t <= 2:
            continue
        past_data = data[:t]
        predicted_asset_value, predicted_asset_variance = prediction_model.predict(past_data)
        predicted_asset_values[t] = predicted_asset_value

        control_input = (predicted_asset_value, predicted_asset_variance)
        cov_model.run(control_input, gamma)
        investment_strategy = cov_model.variables()
        investment_strategies[t] = investment_strategy

    return predicted_asset_values, investment_strategies


def run_experiments():
    num_samples = 200
    num_assets = 3
    data = get_data(num_samples, num_assets)

    # Add experiments to run here.
    experiments = [
        ("gaussian_mean_variance", run_gaussian_mean_variance),
    ]

    for name, experiment_func in experiments:
        predicted_asset_values, investment_strategies = experiment_func(data, num_samples, num_assets)
        predicted_return, true_return = get_returns(data, investment_strategies, predicted_asset_values)
        all_error = error(predicted_return, true_return)
        window = 10
        for i in range(0, num_samples-window, window):
            print(name, np.mean(all_error[i:i + window]))
        # We really just care about how well the investment strategies actually do,
        # which is given by true_return.
        plt.plot(np.arange(3, num_samples), true_return[3:], label=name + ' true return')
        # In final plots, predicted return may not be relevant.
        plt.plot(np.arange(3, num_samples), predicted_return[3:], label=name + ' predicted return')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    run_experiments()
