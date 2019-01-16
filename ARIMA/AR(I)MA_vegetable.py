import pandas as pd
import sys
import csv
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA 

#使图像显示在juypter里面
%matplotlib inline 

# 单位根检验
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



#处理不平稳数据
def best_diff(ts, maxdiff = 8):
    for i in range(0, maxdiff):
        temp = ts.copy() #每次循环前，重置
        if i == 0:
            temp_diff = temp
        else:
            temp_diff = temp.diff(i)
            temp_diff.dropna(inplace=True)
        p_value = testStationarity(temp_diff)['p-value']
        if p_value < 0.99:
            return i

#获取合适的ARMA模型系数
def proper_model_ARMA(data_ts, maxLag):
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




#获取合适的ARIMA模型系数
def proper_model_ARIMA(data_ts, maxLag):
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
            model = ARIMA(data_ts, order=(p, 1, q))
            try:
                results_ARIMA = model.fit(disp=-1, method='css')
            except:
                print("")
                continue
            bic = results_ARIMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_ARIMA
                init_bic = bic
    return init_bic, init_p, init_q, init_properModel



  
#提取所需序列

#读取已统计的蔬菜集合
df = pd.read_csv('/Users/huangguobin/Desktop/statistical_result.csv', encoding='utf-8', index_col='name')
sta_vegetable = set(df.index)
print(len(sta_vegetable))

#读取全部蔬菜集合
df = pd.read_csv('/Users/huangguobin/Desktop/vegetable.csv', encoding='utf-8', index_col='name')
all_vegetable = set(df.index)
print(len(all_vegetable))

#得到未统计的蔬菜集合
vegetable = all_vegetable - sta_vegetable
print(len(vegetable))
print(vegetable)


for i in vegetable:
    print(i)
    vegetable_name = i
    data = df.loc[i]
    data.set_index(['date'],inplace = True)
    data = data['price']
    
    if(len(data) < 800):   #数据小于800个的不加入统计
        print("a")
        continue
        
    number = 700
    predict_datas = 0
    percent1 = 0
    percent5 = 0
    percent10 = 0
    
 #每100个数据一组，统计700个组的误差率
    for i in range(number):     
        ts = data.iloc[i:(i+100)]
        
        k = best_diff(ts, maxdiff = 15)     #差分的最优阶
        if k == None:
            print(i)
        elif k > 0:
            print(i,k)
            
    for i in range(number):
        print(i)
        ts = data.iloc[i:(i+100)]
        p_value =  testStationarity(ts)['p-value']
        
        #检验平稳性，选取适当模型和模型参数，并获取第101的预测值
        if p_value < 0.99:           
            bic, p, q, properModel = proper_model_ARMA(ts, 5)
            print(bic, p, q, properModel)
            predict_day_count = 2
            predict_datas = properModel.forecast(predict_day_count,alpha=0.05)[0][0]
        else:                               
            bic, p, q, properModel = proper_model_ARIMA(ts, 5)
            print(bic, p, q, properModel)
            predict_day_count = 2
            predict_datas = properModel.forecast(predict_day_count, alpha=0.05)[0][0] 
            print(predict_datas)
        
        #统计误差率
        percent = abs(data.iloc[i+100] - predict_datas)/data.iloc[i+100]
        print(percent)
        if percent < 0.01:
            percent1 = percent1 + 1
        elif percent < 0.05:
            percent5 = percent5 + 1 
        elif percent < 0.1:
            percent10 = percent10 + 1
            
    per_percent1 = percent1 / number
    per_percent5 = percent5 / number
    per_percent10 = percent10 / number
    print("number:" + str(number))
    print("error:" + str(error))
    print("0~1%:" + str(percent1) + "  " + str(per_percent1))
    print("1~5%:" + str(percent5) + "  " + str(per_percent5))
    print("5~10%:" + str(percent10) + "  " + str(per_percent10))
    
    with open('/Users/huangguobin/Desktop/statistical_result.csv','a+') as f:
        writer = csv.writer(f)
        writer.writerow((vegetable_name,number,percent1,per_percent1,percent5,per_percent5,percent10,per_percent10))
            