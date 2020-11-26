import time
import numpy as np
import scipy as sp
from scipy import stats
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel(r'C:\Users\grlef\PycharmProjects\Binance\symbols\symbols.xlsx')
symbols = [None]*len(df)
for i in range(len(df)):
    symbols[i] = df.loc[i][0]
# time.sleep(1)

# alpha = np.zeros(len(df))
# beta = np.zeros(len(df))
# r_value = np.zeros(len(df))
# p_value = np.zeros(len(df))
# std_err = np.zeros(len(df))

time_units=1
time_spec='d'
period=str(time_units)+time_spec

##############################     MARKET INDEX    ###################################
def market_index(ticker):
    stocks = {ticker: yf.Ticker(ticker).history(period=period)}
    returns = {ticker: np.diff(stocks[ticker]['Close'])/stocks[ticker]['Close'][:-1]}
    mean_returns = {ticker: np.mean(returns[ticker])}
    volatility = {ticker: np.std(returns[ticker])}

    return stocks, returns, mean_returns, volatility
##############################     MARKET INDEX    ###################################


def stock_stats(symbols, start, end): # "yyyy-mm-dd"

    stocks = yf.download(tickers=symbols, start=start, end=end,group_by='ticker')

    return stocks


def benchmark_index(symbol, start, end): # "yyyy-mm-dd"

    index = yf.download(tickers=symbol, start=start, end=end,group_by='ticker')

    h = np.array(index['High'])
    l = np.array(index['Low'])
    c = np.array(index['Close'])
    m = (h+l)/2
    Rx = np.divide(c[1:] - c[0:-1], c[0:-1])

    return index, c, Rx

def stock_returns(stocks):

    # hls = {}  # Today's High minus Yesterday's Low
    # mms = {}  # Today's Medium (random) minus Yesterday's Medium (random)
    ccs = {}  # Today's Close minus Yesterday's Close
    # oos = {}  # Today's Open minus Yesterday's Open
    # cos = {}  # Today's Close minus Yesterday's Open
    # ocs = {}  # Today's Open minus Yesterday's Close

    for i in range(int(len(list(stocks))/6)):
        # h = np.array(stocks[symbols[i]]['High'])
        # l = np.array(stocks[symbols[i]]['Low'])
        c = np.array(stocks[symbols[i]]['Close'])
        # o = np.array(stocks[symbols[i]]['Open'])
        # m = (h + l)/2

        # hls.update({symbols[i]: np.divide(h[1:] - l[0:-1], l[0:-1])})
        # mms.update({symbols[i]: np.divide(m[1:] - m[0:-1], m[0:-1])})
        ccs.update({symbols[i]: np.divide(c[1:] - c[0:-1], c[0:-1])})
        # oos.update({symbols[i]: np.divide(o[1:] - o[0:-1], o[0:-1])})
        # cos.update({symbols[i]: np.divide(c[1:] - o[0:-1], o[0:-1])})
        # ocs.update({symbols[i]: np.divide(o[1:] - c[0:-1], c[0:-1])})

    return ccs#, hls, mms, oos, cos, ocs, c

def stocks_capm(stock_returns,rx):

    alpha = {}
    beta = {}
    epsilon = {}
    beta_pvalue ={}
    alpha_tstat = {}

    for i in range(len(stock_returns)):
        ry = stock_returns[list(stock_returns)[i]]
        beta_, alpha_, r_value_, p_value_, std_err_ = stats.linregress(rx, ry)

        ############### calculate statistical significance of alpha ######################
        sigma2 = np.sum(np.square(ry-alpha_-beta_*rx))/(len(rx)-2)
        Syy = np.sum(np.square(rx-np.mean(rx)))
        std_err_alpha = np.sqrt(sigma2*((1/len(rx)) + (np.square(np.mean(rx))/Syy)) )
        T0 = alpha_/std_err_alpha
        Tt = stats.t.ppf(0.975, len(rx))
        ############### calculate statistical significance of alpha ######################

        epsilon_ = stock_returns[list(stock_returns)[i]] - (alpha_ + beta_*rx)

        alpha.update({list(stock_returns)[i]: alpha_})
        beta.update({list(stock_returns)[i]: beta_})
        epsilon.update({list(stock_returns)[i]: epsilon_})
        beta_pvalue.update({list(stock_returns)[i]: p_value_})
        alpha_tstat.update({list(stock_returns)[i]: T0-Tt})

    return alpha, beta, epsilon, beta_pvalue, alpha_tstat


# def stock_stats(df,period):
#     stocks = {}
#     returns = {}
#     mean_returns = {}
#     volatility = {}
#     minmax_spread = {}
#     for i in range(len(df)):
#         stocks.update({df.loc[i][0]: yf.Ticker(df.loc[i][0]).history(period=period)})
#         h = np.array(stocks[df.loc[i][0]]['High'])
#         l = np.array(stocks[df.loc[i][0]]['Low'])
#         minmax_spread.update({df.loc[i][0]: np.divide(h[1:]-l[0:-1], l[0:-1])})
#         # returns.update({df.loc[i][0]: np.diff(stocks[df.loc[i][0]]['Close'])/stocks[df.loc[i][0]]['Close'][:-1]})
#         # mean_returns.update({df.loc[i][0]: np.mean(returns[df.loc[i][0]])})
#         # volatility.update({df.loc[i][0]: np.std(returns[df.loc[i][0]])})
#         # if len(returns[df.loc[i][0]]) == len(stocks[df.loc[0][0]]['Close'])-1:
#         #     slope, intercept, r_value_, p_value_, std_err_ = stats.linregress(returns['SPY'], returns[df.loc[i][0]])
#         #     beta[i], alpha[i], r_value[i], p_value[i], std_err[i] = slope, intercept, r_value_, p_value_, std_err_
#
#     return minmax_spread

def find_nonegative_distributions(minmax_spread, left_limit):
    dist = -10*np.ones(len(minmax_spread))
    for j in range(len(minmax_spread)):
        print(j)
        if not np.isnan(minmax_spread[symbols[j]][0]):
            x = plt.hist(minmax_spread[symbols[j]])
            dist[j] = x[1][0] #leftmost value of distribution
    ind = np.where(dist > left_limit)

    return ind







# exec(open(r'C:\Users\grlef\PycharmProjects\Binance\stock_data.py').read())