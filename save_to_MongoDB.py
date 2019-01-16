# coding: utf-8
# @author: lin
# @date: 18-7-22


import pandas as pd
from lib.mongo_base_model_lib import db, client
import time

pd.set_option('max_colwidth', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.max_rows', 5000)

vegetable_all_data_set = db.vegetable_all_data_set


def save_data(data):
    """
    将爬虫得到的数据，存入数据库，
    :param data:DataFrame结构
    :return:None
    """
    vegetable_list = []
    start = time.time()
    print('开始写入')
    for index, row in data.iterrows():
        if not row['name'] in vegetable_list:
            vegetable_list.append(row['name'])
        vegetable_dict = {'name': row['name'], 'date': row['date'], 'place': row['place'], 'price': row['price']}
        vegetable_all_data_set.insert(vegetable_dict)  # 存入一个字典
    vegetable_all_data_set.insert({'name': 'vegetable_list', 'data': vegetable_list})
    end = time.time()
    print('耗时', end - start, 's')


def delete_all_data():
    """
    先找到有所有蔬菜名字的list，再遍历删除
    :return:
    """
    vegetable_list = get_vegetable_list()
    for vegetable in vegetable_list:
        vegetable_all_data_set.remove({'name': vegetable})
    vegetable_all_data_set.remove({'name': 'vegetable_list'})


def get_one_vegetable_data(vegetable_name):
    """
    得到一种蔬菜的价格数据
    :param vegetable_name:
    :return:
    """
    data_list = []
    for data in vegetable_all_data_set.find({'name': vegetable_name}):
        list = [data['name'], '20' + data['date'], data['place'], data['price']]
        data_list.append(list)
    data_frame = pd.DataFrame(data_list, columns=['name', 'date', 'place', 'price'])
    return data_frame


def get_vegetable_list():
    """
    得到蔬菜列表
    :return:
    """
    vegetable_list = vegetable_all_data_set.find_one({'name': 'vegetable_list'})['data']
    return vegetable_list




# delete_all_data()
# for i in vegetable_all_data_set.find():
#     print(i)
# print(get_one_vegetable_data("大蒜"))
#
# for i in get_one_vegetable_data('大蒜'):
#     print(i)

# data = pd.read_csv("C:/Users/lin/Desktop/info.csv")
# sorted_data = data.sort_values(by = ['name', 'date'],axis = 0,ascending = True)
# save_data(sorted_data)
# print(sorted_data)
# print(get_vegetable_list())
#sorted_data.to_csv("C:/Users/lin/Desktop/sorted_vegetable.csv")