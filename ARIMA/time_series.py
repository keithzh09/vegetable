# coding: utf-8
# @author: lin
# @date: 18-7-22


from save_to_MongoDB import get_one_vegetable_data,get_vegetable_list
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
import sys
from statsmodels.tsa.arima_model import ARMA, ARIMA

pd.set_option('max_colwidth',5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.max_rows', 5000)


pd.set_option('display.max_rows', 1000)


def draw_trend(timeSeries, size):
    """
    画出原数据曲线图及移动平均图
    :param timeSeries:
    :param size:
    :return:
    """
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    #rol_weighted_mean = pd.ewma(timeSeries, span=size)
    plt.plot(timeSeries, color='blue', label='Original')
    plt.plot(rol_mean, color='red', label='Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.xlabel('date')
    plt.ylabel('price')
    plt.show()

def draw_ts(timeSeries):
    """
    画出曲线图
    :param timeSeries:
    :return:
    """
    plt.plot(timeSeries, color='blue', label='price')
    plt.title('price/date')
    plt.xlabel('date')
    plt.ylabel('price')
    plt.show()

def testStationarity(timeSeries):
    """
    测试平稳性
    :param timeSeries:
    :return:
    """
    dftest = adfuller(timeSeries)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput


def draw_acf_pacf(ts, lags=31):
    """
    # 自相关和偏相关图，默认阶数为31阶，即
    :param ts:
    :param lags:
    :return:
    """
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=31, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=31, ax=ax2)
    plt.show()

def change_to_list(data):
    """
    将dataframe结构转化为价格和date两种, 这个函数之后没有什么必要了，因为画曲线用Series而不是传入Dataframe
    :param data:
    :return:
    """
    date = []
    price = []
    for index,row in data.iterrows():
        index = str(index)[0:10]
        date.append(datetime.strptime(index, '%Y-%m-%d').date())
        price.append(row['price'])
    return price,date

def draw_predict(original_data, predict_data):
    """
    画出原曲线和拟合曲线
    :param original_data:
    :param predict_data:
    :return:
    """
    try:
        original_data = original_data[predict_data.index]  # 过滤没有预测的记录
    except:
        print("Fuck")
    print(original_data)
    plt.plot(original_data, color='blue', label='Original')
    plt.plot(predict_data, color='red', label='Predict')
    plt.legend(loc='best')
    print(sum((predict_data - original_data)**2)/original_data.size)
    plt.title('RMSE: %.4f' % np.sqrt(sum((predict_data - original_data)**2)/original_data.size))
    plt.xlabel('date')
    plt.ylabel('price')
    plt.show()

def proper_model(data_ts, maxLag):
    """
    BIC准则测试自相关和偏自相关系数
    :param data_ts:
    :param maxLag:
    :return:
    """
    init_bic = sys.maxsize
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARMA(data_ts, order=(p, q))
            try:
                results_ARMA = model.fit(disp=-1, method='css')
            except:
                continue
            bic = results_ARMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_ARMA
                init_bic = bic
    return init_bic, init_p, init_q, init_properModel


"""
以下为测试代码，仅供参考
"""

data = get_one_vegetable_data("大蒜")
# print(data)
data.drop([0], inplace=True)
data.set_index('date',inplace=True)
data.index = pd.to_datetime(data.index, format='%Y-%m-%d')    #变为以timestamp为索引的
data = data['price']
draw_trend(data, 12)
print(data)
#
# draw_trend(data,30)
# draw_ts(data)
# log_data = np.log(data)
# # draw_trend(log_data,31)
# # print(testStationarity(data))
# # print(testStationarity(log_data))
rol_mean = data.rolling(window=30).mean()
rol_mean.dropna(inplace=True)
# print(testStationarity(rol_mean))

ts_diff_1 = data.diff(1)
ts_diff_1.dropna(inplace=True)
print(testStationarity(ts_diff_1))
# draw_predict(data, rol_mean)

draw_acf_pacf(ts_diff_1)
print(ts_diff_1)

# a,p,q,b = proper_model(ts_diff_1, 10)
# print(a,p,q,b)
# #
# from statsmodels.tsa.arima_model import ARIMA
# model = ARIMA(data, order=(p,1,  q))
# result_arma = model.fit( disp=-1, method='css')
#
# predict_ts = result_arma.predict()
# # print(predict_ts)
# # 一阶差分还原
# diff_shift_ts = data.shift(1)
# # print(diff_shift_ts)
#
# diff_recover_1 = predict_ts.add(diff_shift_ts)
#
# diff_recover_1.dropna(inplace=True)
# draw_predict(data, diff_recover_1)

# print(results_AR.fittedvalues)
# print(results_AR.fittedvalues.to_frame(name=None))
# draw_predict(data, results_AR.fittedvalues.to_frame(name='price'))

# print(ts_diff_1['2018-07-29'])

# from statsmodels.tsa.seasonal import seasonal_decompose
# decomposition = seasonal_decompose(data, model="additive")
#
# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid