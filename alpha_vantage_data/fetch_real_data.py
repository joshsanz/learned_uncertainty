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
api_key_1 = "QCI3RT39F1VOUR21"
api_key_2 = "BVS30FU4LEK4Q948"
tech_stocks = ["MSFT", "AAPL", "GOOGL", "AMZN", "FB", "NFLX", "IBM", "TSLA"]
DJIA = ["MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DIS", "DWDP", "XOM",
        "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "MCD", "MRK", "MSFT", "NKE", "PFE",
        "PG", "TRV", "UTX", "UNH", "VZ", "V", "WMT", "WBA"]
SP100 = ["AAPL", "ABBV", "ABT", "ACN", "AGN", "AIG", "ALL", "AMGN", "AMZN", "AXP", "BA", "BAC", "BIIB",
        "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CELG", "CHTR", "CL", "CMCSA", "COF", "COP",
        "COST", "CSCO", "CVS", "CVX", "DHR", "DIS", "DUK", "DWDP", "EMR", "EXC", "F", "FB", "FDX", "FOX",
        "FOXA", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HAL", "HD", "HON", "IBM", "INTC", "JNJ",
        "JPM", "KHC", "KMI", "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO",
        "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "OXY", "PEP", "PFE", "PG", "PM", "PYPL",
        "QCOM", "RTN", "SBUX", "SLB", "SO", "SPG", "T", "TGT", "TXN", "UNH", "UNP", "UPS", "USB", "UTX",
        "V", "VZ", "WBA", "WFC", "WMT", "XOM"]
SP500 = ["A", "AAL", "AAP", "AAPL", "ABBV", "ABC", "ABMD", "ABT", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADS",
        "ADSK", "AEE", "AEP", "AES", "AFL", "AGN", "AIG", "AIV", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALK",
        "ALL", "ALLE", "ALXN", "AMAT", "AMD", "AME", "AMG", "AMGN", "AMP", "AMT", "AMZN", "ANET", "ANSS",
        "ANTM", "AON", "AOS", "APA", "APC", "APD", "APH", "APTV", "ARE", "ARNC", "ATVI", "AVB", "AVGO",
        "AVY", "AWK", "AXP", "AZO", "BA", "BAC", "BAX", "BBT", "BBY", "BDX", "BEN", "BF", "BHF", "BHGE",
        "BIIB", "BK", "BKNG", "BLK", "BLL", "BMY", "BR", "BRK", "BSX", "BWA", "BXP", "C", "CAG", "CAH",
        "CAT", "CB", "CBOE", "CBRE", "CBS", "CCI", "CCL", "CDNS", "CELG", "CERN", "CF", "CFG", "CHD", "CHRW",
        "CHTR", "CI", "CINF", "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF",
        "COG", "COO", "COP", "COST", "COTY", "CPB", "CPRT", "CRM", "CSCO", "CSX", "CTAS", "CTL", "CTSH",
        "CTXS", "CVS", "CVX", "CXO", "D", "DAL", "DE", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DISCA",
        "DISCK", "DISH", "DLR", "DLTR", "DOV", "DRE", "DRI", "DTE", "DUK", "DVA", "DVN", "DWDP", "DXC", "EA",
        "EBAY", "ECL", "ED", "EFX", "EIX", "EL", "EMN", "EMR", "EOG", "EQIX", "EQR", "ES", "ESRX", "ESS",
        "ETFC", "ETN", "ETR", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", "FB", "FBHS",
        "FCX", "FDX", "FE", "FFIV", "FIS", "FISV", "FITB", "FL", "FLIR", "FLR", "FLS", "FLT", "FMC", "FOX",
        "FOXA", "FRT", "FTI", "FTNT", "FTV", "GD", "GE", "GILD", "GIS", "GLW", "GM", "GOOG", "GOOGL",
        "GPC", "GPN", "GPS", "GRMN", "GS", "GT", "GWW", "HAL", "HAS", "HBAN", "HBI", "HCA", "HCP", "HD",
        "HES", "HFC", "HIG", "HII", "HLT", "HOG", "HOLX", "HON", "HP", "HPE", "HPQ", "HRB", "HRL", "HRS",
        "HSIC", "HST", "HSY", "HUM", "IBM", "ICE", "IDXX", "IFF", "ILMN", "INCY", "INFO", "INTC", "INTU",
        "IP", "IPG", "IPGP", "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "JBHT", "JCI", "JEC", "JEF",
        "JKHY", "JNJ", "JNPR", "JPM", "JWN", "K", "KEY", "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI",
        "KMX", "KO", "KORS", "KR", "KSS", "KSU", "L", "LB", "LEG", "LEN", "LH", "LIN", "LKQ", "LLL",
        "LLY", "LMT", "LNC", "LNT", "LOW", "LRCX", "LUV", "LW", "LYB", "M", "MA", "MAA", "MAC", "MAR",
        "MAS", "MAT", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "MGM", "MHK", "MKC", "MLM",
        "MMC", "MMM", "MNST", "MO", "MOS", "MPC", "MRK", "MRO", "MS", "MSCI", "MSFT", "MSI", "MTB",
        "MTD", "MU", "MXIM", "MYL", "NBL", "NCLH", "NDAQ", "NEE", "NEM", "NFLX", "NFX", "NI", "NKE", "NKTR",
        "NLSN", "NOC", "NOV", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NWL", "NWS", "NWSA", "O",
        "OKE", "OMC", "ORCL", "ORLY", "OXY", "PAYX", "PBCT", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG",
        "PG", "PGR", "PH", "PHM", "PKG", "PKI", "PLD", "PM", "PNC", "PNR", "PNW", "PPG", "PPL", "PRGO",
        "PRU", "PSA", "PSX", "PVH", "PWR", "PXD", "PYPL", "QCOM", "QRVO", "RCL", "RE", "REG", "REGN",
        "RF", "RHI", "RHT", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTN", "SBAC", "SBUX",
        "SCG", "SCHW", "SEE", "SHW", "SIVB", "SJM", "SLB", "SLG", "SNA", "SNPS", "SO", "SPG", "SPGI", "SRE",
        "STI", "STT", "STX", "STZ", "SWK", "SWKS", "SYF", "SYK", "SYMC", "SYY", "T", "TAP", "TDG", "TEL",
        "TGT", "TIF", "TJX", "TMK", "TMO", "TPR", "TRIP", "TROW", "TRV", "TSCO", "TSN", "TSS", "TTWO",
        "TWTR", "TXN", "TXT", "UA", "UAA", "UAL", "UDR", "UHS", "ULTA", "UNH", "UNM", "UNP", "UPS", "URI",
        "USB", "UTX", "V", "VAR", "VFC", "VIAB", "VLO", "VMC", "VNO", "VRSK", "VRSN", "VRTX", "VTR",
        "VZ", "WAT", "WBA", "WCG", "WDC", "WEC", "WELL", "WFC", "WHR", "WLTW", "WM", "WMB", "WMT", "WRK",
        "WU", "WY", "WYNN", "XEC", "XEL", "XLNX", "XOM", "XRAY", "XRX", "XYL", "YUM", "ZBH", "ZION", "ZTS", ]

stocks = SP100

def daily_value_estimate(daily_value_dict):
    open_val = float(daily_value_dict["1. open"])
    high_val = float(daily_value_dict["2. high"])
    low_val = float(daily_value_dict["3. low"])
    close_val = float(daily_value_dict["4. close"])
    volume = float(daily_value_dict["5. volume"])
    return close_val


def get_time_series(symbol, api_key):
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
        print(symbol, "failed:", e, url)
    return dates, result

all_data = []
all_dates = []
keys = [api_key_1, api_key_2] * len(stocks)
for i, stock in enumerate(stocks):
    single_dates, single_data = get_time_series(stock, keys[i])
    # Throttled to 5 api calls per minutes per key, split between two keys (shhhh)
    time.sleep(12)
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
