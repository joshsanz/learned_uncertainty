from urllib import request
import re
import json
import csv
import os
import pickle
import numpy as np
import datetime
import time


# Documentation: https://www.alphavantage.co/documentation/
api_key = "QCI3RT39F1VOUR21"
stocks = ["MSFT", "AAPL", "GOOGL", "AMZN", "FB", "NFLX", "IBM", "TSLA"]


def daily_value_estimate(daily_value_dict):
    open_val = float(daily_value_dict["1. open"])
    high_val = float(daily_value_dict["2. high"])
    low_val = float(daily_value_dict["3. low"])
    return np.mean([low_val, high_val])


def get_time_series(symbol):
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=" + symbol + "&apikey=" + api_key
    htmltext = request.urlopen(url)
    content = htmltext.read()
    data = json.loads(content)
    dates = np.array([])
    result = np.array([])
    try:
        time_dict = data["Time Series (Daily)"]
        dates = time_dict.keys()
        dates = sorted(dates, key=lambda date_str: datetime.datetime.strptime(date_str, "%Y-%m-%d").timestamp())
        result = np.zeros(shape=(len(dates),))
        for i in range(result.shape[0]):
            result[i] = daily_value_estimate(time_dict[dates[i]])
    except Exception as e:
        print(symbol, "failed:", e)
    time.sleep(2.0)
    return dates, result

all_data = []
all_dates = []
for stock in stocks:
    single_dates, single_data = get_time_series(stock)
    print(stock, single_data.shape)
    all_data.append(single_data)
    all_dates.append(single_dates)


all_data = np.array(all_data).T
print("num_samples=", all_data.shape[0])
print("num_assets=", all_data.shape[1])

# ensure dates are the same
for i in range(all_data.shape[1]):
    for j in range(i, all_data.shape[1]):
        assert all_dates[i] == all_dates[j]
print("date range:", all_dates[0][0], "-", all_dates[0][-1])

with open("./data/real_data_symbols.pickle", "wb") as fh:
    pickle.dump(stocks, fh)

with open("./data/real_data_dates.pickle", "wb") as fh:
    pickle.dump(all_dates, fh)

with open("./data/real_data.pickle", "wb") as fh:
    pickle.dump(all_data, fh)
