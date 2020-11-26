from stock_data import *
from CDB_Assistant import DB_Tool, Groups_watchlist

start = '2018-01-03'
end = '2018-11-01'


stocks = stock_stats(symbols,start,end)
ccs = stock_returns(stocks)
qqq, cx, Rx = benchmark_index('QQQ',start,end)

alpha, beta, epsilon, beta_p = stocks_capm(ccs, Rx)


# def extract_list(beta_tol,p_value_tol,output_filename):
#     b  = np.zeros(len(beta))
#     bp = np.zeros(len(beta))
#     for i in range(len(beta)):
#         b[i]=beta[list(beta)[i]]
#         bp[i] = beta_p[list(beta)[i]]
#
#     bind = np.where(np.logical_and(b < beta_tol, bp < p_value_tol))[0]
#     fstocks=[]
#     for i in range(len(bind)):
#         fstocks.append([list(beta)[bind[i]],beta[list(beta)[bind[i]]],beta_p[list(beta)[bind[i]]]])
#     fstocks = pd.DataFrame(fstocks)
#     fstocks.to_excel(output_filename + '.xlsx')

b  = np.zeros(len(beta))
bp = np.zeros(len(beta))
for i in range(len(beta)):
    b[i]=beta[list(beta)[i]]
    bp[i] = beta_p[list(beta)[i]]

bind = np.where(np.logical_and(b<-0.5, bp<0.00001))[0]
fstocks=[]
for i in range(len(bind)):
    fstocks.append([list(beta)[bind[i]],beta[list(beta)[bind[i]]],
                    beta_p[list(beta)[bind[i]]],alpha[list(beta)[bind[i]]]])
fstocks = pd.DataFrame(fstocks)
fstocks.to_excel('yfinance_filtering_nyse1.xlsx')

