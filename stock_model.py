import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA

# TRADING DAY: 15:00-3:00 = 12 HOURS = 720 MINUTES



def stock_stats(symbol):

    x1 = r'C:\Users\grlef\PycharmProjects\Binance'
    symbol = '\\'+symbol
    x2 = r'.xlsx'

    ############################ INITIAL DATA SAMPLE ############################
    sample_percentage = 100  # percentage of sample to use
    df = pd.read_excel(x1+symbol+x2)
    df = df[-int(np.round((sample_percentage/100)*len(df))):]
    ############################ INITIAL DATA SAMPLE ############################


    ################# BASIC STOCK DATA VECTORS AND TIME CONFIG ##################
    c = np.array(df['Close'])
    h = np.array(df['High'])
    l = np.array(df['Low'])
    v = np.array(df['Volume'])
    u = np.array(df['Up'])
    d = np.array(df['Down'])
    T = np.array(df['Time'])
    D = np.array(df['Date'])

    umd = u-d
    rc = c[1:]/c[0:-1]-1

    for i in range(len(D)):
        T[i] = T[i].strftime('%H.%M')
        if not isinstance(D[i],str):
            D[i] = D[i].strftime('%d/%m/%Y')
    ################# BASIC STOCK DATA VECTORS AND TIME CONFIG ##################


    ##################### MEAN AND STD EVOLUTION #####################
    def cumulative_meanstd(sample_percentage):
        start = -int(np.round((sample_percentage) * len(c)))
        vsize = len(c[start:len(c)])
        mu = np.zeros(vsize)
        mu[0]=c[start]
        sigma = np.zeros(vsize)
        j=1
        for i in np.arange(start+1,0):
            mu[j]=np.mean(c[start:i])
            sigma[j] = np.std(c[start:i])
            j=j+1
        return mu, sigma
    ##################### MEAN AND STD EVOLUTION #####################


    ########################################### DAILY DATA CLUSTERING #############################################
    Dind = np.zeros(1)
    xD = D[0]
    for i in range(len(D)-1):
         if D[i+1] != D[i]:
             Dind = np.append(Dind, i+1)

    # trading_days_ind = np.where(T == '15.01')
    # no_of_trading_days = np.shape(trading_days_ind)[1]
    no_of_trading_days = len(Dind)
    trading_days_ind = [None]*(len(Dind))
    for i in range(no_of_trading_days):
        trading_days_ind[i] = int(Dind[i])

    cday = [None]*no_of_trading_days
    vday = [None]*no_of_trading_days
    uday = [None]*no_of_trading_days
    dday = [None]*no_of_trading_days
    min_cday = np.zeros(no_of_trading_days)
    max_cday = np.zeros(no_of_trading_days)
    close_cday = np.zeros(no_of_trading_days)
    mean_cday = np.zeros(no_of_trading_days)
    std_cday = np.zeros(no_of_trading_days)
    cum_vday = np.zeros(no_of_trading_days)
    cum_uday = np.zeros(no_of_trading_days)
    cum_dday = np.zeros(no_of_trading_days)
    for i in range(no_of_trading_days-1):
        cday[i] = c[trading_days_ind[i]:trading_days_ind[i + 1]]
        vday[i] = v[trading_days_ind[i]:trading_days_ind[i + 1]]
        uday[i] = u[trading_days_ind[i]:trading_days_ind[i + 1]]
        dday[i] = d[trading_days_ind[i]:trading_days_ind[i + 1]]
        min_cday[i] = np.min(cday[i])
        max_cday[i] = np.max(cday[i])
        close_cday[i] = cday[i][-1] # closing price at 00:00 NOT 23:00!!!
        mean_cday[i] = np.mean(cday[i])
        std_cday[i] = np.std(cday[i])
        cum_vday[i] = np.sum(vday[i])
        cum_uday[i] = np.sum(uday[i])
        cum_dday[i] = np.sum(dday[i])
    ################################## LAST DAY #####################################
    cday[no_of_trading_days-1] = c[trading_days_ind[no_of_trading_days-1]:len(df)]
    vday[no_of_trading_days-1] = v[trading_days_ind[no_of_trading_days-1]:len(df)]
    uday[no_of_trading_days-1] = u[trading_days_ind[no_of_trading_days-1]:len(df)]
    dday[no_of_trading_days-1] = d[trading_days_ind[no_of_trading_days-1]:len(df)]
    min_cday[no_of_trading_days-1] = np.min(cday[no_of_trading_days-1])
    max_cday[no_of_trading_days-1] = np.max(cday[no_of_trading_days-1])
    close_cday[no_of_trading_days-1] = cday[no_of_trading_days-1][-1]
    mean_cday[no_of_trading_days-1] = np.mean(cday[no_of_trading_days-1])
    std_cday[no_of_trading_days-1] = np.std(cday[no_of_trading_days-1])
    cum_vday[no_of_trading_days-1] = np.sum(vday[no_of_trading_days-1])
    cum_uday[no_of_trading_days-1] = np.sum(uday[no_of_trading_days-1])
    cum_dday[no_of_trading_days-1] = np.sum(dday[no_of_trading_days-1])
    ################################## LAST DAY #####################################

    true_range = max_cday - min_cday

    def theoretical_maxmin_spread(start_day,days_frequency):
        spread = max_cday[start_day:len(max_cday):days_frequency]-min_cday[start_day-1:len(max_cday)-1:days_frequency]
        returns = np.divide(spread,min_cday[start_day-1:len(max_cday)-1:days_frequency])
        return spread, returns

    def maxmin_visual():
        plt.plot(min_cday[0:-1], '-b*')
        plt.plot(max_cday[1:], '-k*')
        return

    return no_of_trading_days, c, min_cday, max_cday, mean_cday, close_cday
    ########################################### DAILY DATA CLUSTERING ############################################# ## #
#returns no_of_trading_days, c, min_cday, max_cday, mean_cday


############################ DAILY MIN MAX PREDICTORS (LINEAR REGRESSION) #############################
############################           OUT OF SAMPLE PREDICTION           #############################
def linpredict(in_sample_percentage_min,in_sample_percentage_max):
    # in_sample_percentage_min = 0.5
    # in_sample_percentage_max = 0.8
    end_day_min = int(np.floor(in_sample_percentage_min*no_of_trading_days))
    end_day_max = int(np.floor(in_sample_percentage_max*no_of_trading_days))

    beta_min, alpha_min, r_v_min, p_v_min, std_err_min = stats.linregress(min_cday[0:end_day_min], min_cday[1:end_day_min+1])
    beta_max, alpha_max, r_v_max, p_v_max, std_err_max = stats.linregress(max_cday[0:end_day_max], max_cday[1:end_day_max+1])
    def predict_minc(x):
        y = alpha_min + beta_min*x
        return y

    def predict_maxc(x):
        y = alpha_max + beta_max*x
        return y

    min_predicted = np.zeros(no_of_trading_days-1)
    max_predicted = np.zeros(no_of_trading_days-1)
    for i in np.arange(1,no_of_trading_days):
        min_predicted[i-1] = predict_minc(min_cday[i - 1])
        max_predicted[i-1] = predict_maxc(max_cday[i - 1])

    oos_error_min = min_cday[end_day_min+1:]-min_predicted[end_day_min:]
    oos_error_max = max_cday[end_day_max+1:]-max_predicted[end_day_max:]
    abs_oos_error_min = np.abs(oos_error_min)
    abs_oos_error_max = np.abs(oos_error_max)
    mean_oos_error_min = np.mean(oos_error_min)
    mean_oos_error_max = np.mean(oos_error_max)
    std_oos_error_min = np.std(oos_error_min)
    std_oos_error_max = np.std(oos_error_max)
    insample_error_min = min_cday[:end_day_min]-min_predicted[:end_day_min:]
    insample_error_max = max_cday[:end_day_max]-max_predicted[:end_day_max:]
    mean_insample_error_min = np.mean(insample_error_min)
    mean_insample_error_max = np.mean(insample_error_max)

    return min_predicted, max_predicted, mean_oos_error_min, mean_oos_error_max, std_oos_error_min, std_oos_error_max

def predict_arima(no_of_trading_days,data_set,end_day_min,p,d,q):
    if end_day_min == no_of_trading_days:
        predictions = np.zeros(1)
        history = [x for x in data_set[0:end_day_min]]
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        predictions[0] = output[0]
        error = None
        mean_error = None
        std_error = None
    else:
        predictions = np.zeros(len(data_set[end_day_min:]))
        history = [x for x in data_set[0:end_day_min]]
        for t in range(len(data_set[end_day_min:])):
            model = ARIMA(history, order=(p, d, q))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            predictions[t] = output[0]
            history.append(data_set[end_day_min + t])

        error = data_set[end_day_min:] - predictions
        mean_error = np.mean(error)
        std_error = np.std(error)

    return predictions, error, mean_error, std_error

def oos_backtest(no_of_trading_days,end_day,min_cday,max_cday,p,d,q):
    rmin = min_cday[end_day:-1]
    pmin, e, me, se = predict_arima(no_of_trading_days,min_cday, end_day , p, d, q)
    pmin = pmin[0:-1]
    rmax = max_cday[end_day+1:]
    frmin = min_cday[end_day+1:]
    ind = np.where(rmin <= pmin)
    ind_fail = np.where(np.logical_and(rmin <= pmin, pmin >= rmax))
    rmin = rmin[ind]
    rmax = rmax[ind]
    pmin = pmin[ind]
    indf = ind+np.ones(len(ind[0]))
    frmin = frmin[indf.astype(int)]
    r = np.divide(rmax, pmin)-1
    x = plt.hist(r)
    rr = np.zeros(len(x[1]) - 1)
    for k in range(len(x[1]) - 1):
        rr[k] = (x[1][k + 1] + x[1][k]) / 2

    dpr = pmin - rmin
    mdist = np.mean(dpr) # mean difference between predicted and real minimum
    # plt.figure(2)
    # plt.plot(dpr,'-*')

    max_risk = np.divide(frmin - pmin,pmin)
    mean_max_risk = np.mean(max_risk) # maximum risk defined as selling at next day's minimum
    # plt.figure(3)
    # plt.plot(max_risk, '-*')

    expected_return = np.dot(x[0]/np.sum(x[0]), rr)
    no_of_openings = len(ind[0])
    no_of_fail_trades = len(ind_fail[0])
    openings_percentage = no_of_openings/(no_of_trading_days-end_day-1)
    success_percentage = (no_of_openings - no_of_fail_trades)/no_of_openings
    fail_percentage = no_of_fail_trades/no_of_openings



    return expected_return, openings_percentage, success_percentage, dpr, max_risk

def backtest_visual(min_cday,max_cday,end_day,p,d,q):
    plt.cla()
    pmin, e, me, se = predict_arima(min_cday, end_day, p, d, q)
    plt.plot(pmin,'-r*')
    plt.plot(min_cday[end_day:], '-b*')
    plt.plot(max_cday[end_day + 1:], '-k*')
    return

def position_construction(no_of_trading_days,min_cday,p,d,q,mdist):
    pnew, e, me, se = predict_arima(min_cday, no_of_trading_days, p, d, q)




# def actual_maxmin_spread(predicted_set,start_day,days_frequency):
#     spread = max_cday[start_day:len(max_cday):days_frequency]-predicted_set[start_day-1:len(max_cday)-1:days_frequency]
#     returns = np.divide(spread,predicted_set[start_day-1:len(max_cday)-1:days_frequency])
#     return spread, returns
############################ DAILY MIN MAX PREDICTORS (LINEAR REGRESSION) #############################
############################           OUT OF SAMPLE PREDICTION           #############################

# def intraday_minmax_timing(end_day):
#     cday = cday[end_day:-1]
#     rmin = min_cday[end_day:-1]
#     rmax = max_cday[end_day:-1]
#     pmin, e, me, se = predict_arima(min_cday, end_day, p, d, q)
#     pmin = pmin[0:-1]
#     ind = np.where(rmin <= pmin)
#     y = np.zeros(len(cday))
#     rmin = rmin[ind]
#     rmax = rmax[ind]
#     cday = cday[ind]
#     pmin = pmin[ind]
#     for i in range(len(cday)):
#         indmin = np.where(cday[i] == min_cday[i])
#         indmax = np.where(cday[i] == max_cday[i])
#         indmin = indmin[0][0]
#         indmax = indmax[0][0]
#         if indmin < indmax:
#             y[i] = 1
#     y = np.sum(y)
#     minmax_timing_percent = y/len(cday)
#     return minmax_timing_percent




def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


# exec(open(r'C:\Users\grlef\PycharmProjects\Binance\stock_model.py').read())













def gbm(So,T,mu,sigma,W):
    S = So*np.exp((mu-0.5*(sigma**2))*T+sigma*W)
    return S

def mc_price(So,T,mu,sigma,N):
    c = np.zeros(N)
    for i in range(N):
        c[i] = gbm(So,T,mu,sigma,np.sqrt(T)*np.random.rand())
    c = np.sum(c)/N
    return c