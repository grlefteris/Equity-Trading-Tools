from stock_model import *
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution

# Define linear function
def lin_func(x, a, b):
    return a + b*x
# Define linear function

# Construct Generalized Regression Indicator
def func(x, a, b, c):
    return a + b*(x**2) + c*x

def GRI(ydata,func,window):
    xdata = np.linspace(0, len(ydata), len(ydata))
    indw = int(np.floor(len(ydata)/window))
    out = []
    for i in range(indw):
        x = xdata[i*window:(i+1)*window]
        y = ydata[i*window:(i+1)*window]
        popt, pcov = curve_fit(func, x, y)
        out = np.append(out, func(x, *popt))

    return out

################ Compute Fast-Slow Series #################
def trend_data(no_of_trading_days,no_of_sample_days,c,slow_weight,fast_weight,fs_diff_tol,diff_cross_tol):

    # fs_diff_tol: refers to the numerical tolerance between the fast and the slow moving averages
    # in order to assess whether there is an actual crossing

    # diff_cross_tol: given fs_diff_tol various crossings arise localy; this parameter establishes
    # a tolerance in order to eliminate all but one crossing value


    dlen = int(np.floor(len(c)/no_of_trading_days)) # length of day price sample
    slen = no_of_sample_days*dlen # length of sample
    cslow = moving_average(c[0:slen],periods=slow_weight*dlen)
    cfast = GRI(c[0:slen],func,fast_weight*dlen)
    cslow = np.append(np.ones(len(c[0:slen])-len(cslow))*c[0], cslow)
    ecut = len(cslow)-len(cfast)
    cslow = cslow[0:len(cslow)-ecut]

    fs_diff = np.abs(cslow - cfast)
    fs_cross_ind = np.where(fs_diff < fs_diff_tol)
    fs_cross_ind = fs_cross_ind[0]

    dcfast_crossed = np.abs(np.diff(cfast[fs_cross_ind]))
    dcfast_ind = np.where(dcfast_crossed > diff_cross_tol)
    fs_cross_ind = fs_cross_ind[dcfast_ind]

    signal_open_ind = np.where(cfast[fs_cross_ind] > cslow[fs_cross_ind])
    signal_open_ind = signal_open_ind[0]
    signal_open_ind = fs_cross_ind[signal_open_ind]
    signal_close_ind = np.where(cfast[fs_cross_ind] < cslow[fs_cross_ind])
    signal_close_ind = signal_close_ind[0]
    signal_close_ind = fs_cross_ind[signal_close_ind]

    return cslow, cfast, fs_cross_ind, signal_open_ind, signal_close_ind


def trend_vis(c,cslow,cfast,fs_cross_ind,signal_open_ind, signal_close_ind):
    plt.cla()
    plt.plot(c, '-b')
    plt.plot(cslow, '-m')
    plt.plot(cfast, '-k')
    plt.plot(signal_open_ind, c[signal_open_ind], 'g*')
    plt.plot(signal_close_ind, c[signal_close_ind], 'r*')
    return


def positions(signal_open_ind,signal_close_ind):
    pos_length = np.min([len(signal_open_ind), len(signal_close_ind)])
    pos = np.zeros(2*pos_length)
    pos_ind = np.zeros(2*pos_length)
    time_ind = 0
    k = 0
    j = 0
    for i in range(pos_length):
        if signal_open_ind[i] > time_ind:
            pos_ind[k] = signal_open_ind[i]
            time_ind = signal_open_ind[i]
            k = k + 1
            flag = k
            while flag < k + 1:
                if signal_close_ind[j] > time_ind:
                    pos_ind[k] = signal_close_ind[j]
                    time_ind = signal_close_ind[j]
                    flag = flag + 1
                j = j + 1
            k = k + 1
    pos_ind = pos_ind[0:np.where(pos_ind == 0)[0][0]]
    pind = [None]*len(pos_ind)
    for i in range(len(pos_ind)):
        pind[i] = int(pos_ind[i])

    return pind

def trend_performance(no_of_trading_days,c,pind):
    dlen = int(np.floor(len(c) / no_of_trading_days))  # length of day price sample
    cp = c[pind]
    gains = np.zeros(int(len(pind) / 2))
    trend_length = np.zeros(int(len(pind) / 2))
    j = 0
    for i in range(1, len(pind), 2):
        gains[j] = cp[i] / cp[i - 1] - 1
        trend_length[j] = (pind[i]-pind[i-1])/dlen
        j = j + 1

    return gains, trend_length



################ Compute Fast-Slow Series #################

###### DETERMINE TREND SIGNALS BASED ON PRICE SPEED-ACCELERATION PROFILE ######
def price_speed_acc_profile(cslow, shift_index, acc_lbound):
    slow_ind = np.where(cslow == cslow[0])[0]
    # cs = cslow[slow_ind[-1] + shift_index:]
    cs = cslow
    dcs = np.diff(cs)
    ddcs = np.diff(dcs)
    cddcs = np.zeros(len(ddcs)) # cummulative price acceleration
    for i in range(len(cddcs)):
        cddcs[i] = np.sum(ddcs[0:i])
    for i in range(len(cddcs)):
        if cddcs[i] < 0:
            cddcs[i] = 0
    s = np.where(cddcs >= acc_lbound)[0]
    o = np.where(cddcs < acc_lbound)[0]

    return cs, cddcs, s, o








