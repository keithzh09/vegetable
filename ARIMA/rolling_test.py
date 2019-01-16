# coding: utf-8
# @author: lin
# @date: 2018/8/3

from vegetable_price_predict.save_to_MongoDB import get_one_vegetable_data,get_vegetable_list
import pandas as pd
import matplotlib.pyplot as plt


from statsmodels.tsa.arima_model import ARMA, ARIMA
pd.set_option('max_colwidth',5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.max_rows', 5000)


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


def draw_trend(original_series, predict_series):
    """
    画出两条曲线
    :param original_series:
    :param predict_series:
    :return:
    """
    plt.plot(original_series, color='blue', label='Original')
    plt.plot(predict_series, color='red', label='Predict')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.xlabel('date')
    plt.ylabel('price')
    plt.show()


def predict(data, p=2, d=0, q=2, predict_day_count=1):
    """
    实现滚动预测，并画图
    :param data:一段序列
    :param p: AR系数
    :param d: 差分数
    :param q: MA系数
    :param predict_day_count: 每次预测的天数
    :return:
    """
    length = int(len(data) * 0.7)
    if_first = True
    while(True):
        train_data = data.iloc[0:length]
        if len(train_data) == len(data):
            break
        model = ARIMA(train_data, order=(p, d, q))
        result_arma = model.fit(disp=-1, method='css')
        predict_datas = result_arma.forecast(predict_day_count)

        data1 = data.iloc[length: length + predict_day_count]
        original_data = pd.DataFrame({'original_data': data1})
        predict_data = pd.DataFrame({'predict_data': data1})    # 借个时间轴为索引，确保两个索引一致
        i = 0
        for index, row in predict_data.iterrows():
            row['predict_data'] = predict_datas[0][i]
            i += 1
        length += predict_day_count
        if if_first:
            all_data = pd.concat([original_data, predict_data], axis=1)
            if_first = False
        else:
            all_data = pd.concat([all_data, pd.concat([original_data, predict_data], axis=1)])
    cal_difference(all_data)
    draw_trend(all_data['original_data'], all_data['predict_data'])


def cal_difference(all_data):
    difference = all_data['predict_data']-all_data['original_data']
    all_data['difference_data'] = difference
    difference_ratio = abs(difference) / all_data['original_data'] * 100
    all_data['difference_ratio'] = difference_ratio
    print(all_data)
    print('max_difference: ', max(abs(all_data['difference_data'])))
    print('max_difference_ratio: ', max(all_data['difference_ratio']), '%')
    # print([all_data['difference_ratio'].idxmax()])
    try:
        index1 = all_data['difference_ratio'].idxmax()
        index2 = all_data['difference_data'].idxmax()
        all_data.drop([index1], inplace=True)
        all_data.drop([index2], inplace=True)
    except Exception as error:
        pass
    print('max_ed_difference: ', max(abs(all_data['difference_data'])))
    print('max_ed_difference_ratio: ', max(all_data['difference_ratio']), '%')


data = get_one_vegetable_data("大蒜")
# print(data)
data.drop([0], inplace=True)
data.set_index('date', inplace=True)
data.index = pd.to_datetime(data.index, format='%Y-%m-%d')    # 变为以timestamp为索引的

data = data['price']
predict(data, p=2, d=1, q=2, predict_day_count=20)


