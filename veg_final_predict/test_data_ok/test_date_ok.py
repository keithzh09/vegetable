# -*- coding: utf-8 -*-
# @Time    : 18-11-30
# @Author  : lin

"""
这个文件是要用来处理蔬菜数据的，要做的有以下几个步骤：
1. 把完全相同的数据删掉，即日期和价格都相同的，删掉
2. 同一天有两个数据，保存第一个
3. 对两个日期之间进行比较，日期大于1的，需要添加数据，先得出相差几天，
   然后两个价格数据相减除以天数，再加上第一个数的值

"""

import datetime
import pandas as pd
import time
import os
from veg_final_predict.test_data_ok.solve_data_tools import SolveData

pd.set_option('max_colwidth', 5000)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 1000)
# 得到预期数据长度应为1065
# d1 = datetime.date(2016, 1, 1)
# d2 = datetime.date(2018, 12, 1)
# print(str(d2 - d1).split(' ')[0])


def get_two_date_days(date1, date2):
    """
    得到两个日期之间的天数
    :param date1: eg.2018-01-01
    :param date2:
    :return:
    """
    date1 = date1.split('-')
    date2 = date2.split('-')
    if len(date1) == 1:
        date1 = date1[0].split('/')
        date2 = date2[0].split('/')  # 两种情况，一种是日期格式为2018-01-01，一种是2018/01/01
    date1 = datetime.date(int(date1[0]), int(date1[1]), int(date1[2]))
    date2 = datetime.date(int(date2[0]), int(date2[1]), int(date2[2]))
    days = int(str(date2 - date1).split(' ')[0])
    return days


def get_between_data1_2(date1, date2):
    """
    获得两个日期间的所有间隔的日期
    :param date1:
    :param date2:
    :return:
    """
    date_list = []
    date_between = date1
    while date_between < date2:
        # 得到日期加一
        date_between = SolveData.day_increase(date_between, 1)[:10]
        date_list.append(date_between)
    date_list.pop()
    return date_list


def get_between_price1_2(price1, price2, number):
    """
    获得两个价格间按递增顺序
    :param price1:前一日期的价格
    :param price2:后一日期的价格
    :param number: 表示这两个价格是间隔几天的数据
    :return:
    """
    price_list = []
    add_price = (price2 - price1) / number
    price_mid = price1
    for i in range(number):
        price_mid += add_price
        price_list.append(round(price_mid, 1))
    price_list.pop()   # 需要去掉最后一个值，因为最后一个值即等于price2
    return price_list


def solve_it(veg_data, veg_name):
    """
    作为一个处理数据的入口
    :param veg_data: 原先的数据
    :param veg_name: 蔬菜名
    :return: 经处理后，日期和价格对应并且完整的数据
    """
    # 首先处理同一天有两个价格的entire
    last_date = None
    drop_index = []

    for index, row in veg_data.iterrows():  # 保留第一个价格数据
        if last_date is None:
            last_date = row['date']
        else:
            now_date = row['date']
            if now_date == last_date:
                drop_index.append(index)
            last_date = now_date

    veg_data.drop(drop_index, inplace=True)
    last_date = None    # 代表上一个日期
    last_price = None   # 代表上一个价格
    all_insert_date = []   # 代表最后要插入的日期
    all_insert_price = []   # 代表最后要插入的价格，长度应该与all_insert_date一致
    for index, row in veg_data.iterrows():
        if last_date is None:
            last_date = row['date']
            last_price = row['price']
        else:
            now_date = row['date']
            now_price = row['price']
            days = get_two_date_days(last_date, now_date)
            if days != 1:   # 当间隔天数大于1的时候
                # print(last_date, now_date)
                all_insert_date.extend(get_between_data1_2(last_date, now_date))
                # print(last_price, now_price)
                all_insert_price.extend(get_between_price1_2(last_price, now_price, days))
            last_date = now_date
            last_price = now_price
    for i in range(len(all_insert_date)):
        # 插入到df的最后面
        veg_data = veg_data.append({'date': all_insert_date[i], 'name': veg_name, 'price': all_insert_price[i]},
                                   ignore_index=True)
    # 按日期升序排列
    veg_data.sort_values(axis=0, ascending=True, by='date', inplace=True)
    # 返回重新整理索引的值
    return veg_data.reset_index().drop('index', axis=1)


if __name__ == '__main__':
    pwd = os.getcwd()  # 当前的完整路径
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")  # 父目录
    root_path = father_path + '/veg_data_18_12/'
    new_path = father_path + '/veg_data_solve_18_12/'
    file_path_list = SolveData.get_file_name(root_path)
    print(file_path_list)
    for file_path in file_path_list:
        total_file_path = root_path + file_path
        data = pd.read_csv(total_file_path).iloc[1:, :]
        data.drop('place', axis=1, inplace=True)
        data.drop_duplicates(inplace=True)
        the_veg_name = file_path.split('.')[0]
        data = solve_it(data, the_veg_name)
        data.to_csv(new_path + file_path)

