import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytrends.request import TrendReq

df = pd.read_excel(r'C:\Users\grlef\PycharmProjects\Binance\watchlist.xlsx')

def get_trends(period):
    pytrends = TrendReq(hl='en-US', tz=360)
    trends = {}
    dft = [None]*len(df)
    for i in range(len(df)):

        pytrends.build_payload([df.iloc[i][0]+' stock'], cat=1163, timeframe=period, geo='', gprop='')
        temp = pytrends.interest_over_time()
        trends.update({df.iloc[i][0]: temp})
        dft[i] = pd.DataFrame(trends[df.iloc[i][0]])
        if not dft[i].empty:
            dft[i] = dft[i].drop(['isPartial'], axis=1)


    all_trends = pd.concat(dft[0:len(dft)],axis=1)
    all_trends.to_excel('watchlist_trends.xlsx')
    st = pd.read_excel (r'C:\Users\grlef\PycharmProjects\Binance\watchlist_trends.xlsx')

    return st

##################### LOW TRENDING STOCKS #####################
def low_trends(st,interest):
    dst = np.zeros(st.shape[1])
    for i in np.arange(1,st.shape[1]):
        dst[i] = np.max(np.diff(st[st.columns.values[i]]))
    low_trend_ind = dst < interest
    return low_trend_ind
##################### LOW TRENDING STOCKS #####################

################### STARTING TO TREND STOCKS ##################
def hot_trends(st, weight):
    hot_trend_ind = [False]*st.shape[1]
    for i in np.arange(1,st.shape[1]):
        a = np.mean(st[st.columns.values[i]][0:143])
        b = np.mean(st[st.columns.values[i]][144:st.shape[0]])
        if b > weight*a:
            hot_trend_ind[i] = True

    return hot_trend_ind
################### STARTING TO TREND STOCKS ##################


####################### LAST-HOURS TRENDS #######################
def last_hours_trends(st, weight):
    last_hours_trend_ind = [False]*st.shape[1]
    for i in np.arange(1,st.shape[1]):
        a = np.mean(st[st.columns.values[i]][0:143])
        b = np.mean(st[st.columns.values[i]][144:st.shape[0]])
        if b > weight*a:
            last_hours_trend_ind[i] = True

    return last_hours_trend_ind
####################### LAST-HOURS TRENDS #######################