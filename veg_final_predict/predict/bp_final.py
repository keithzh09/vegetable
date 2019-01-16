# -*- coding: utf-8 -*-
# @Time    : 19-1-14
# @Author  : lin

import pandas as pd
import numpy as np
import tensorflow as tf
import time
import os
from veg_final_predict.test_data_ok.solve_data_tools import SolveData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 1000)

input_size = 100  # 输入一百个变量
output_size = 1  # 输出一个变量
many_days = 10  # 滚动预测的天数


def get_train_test_data(price_list, batch_size, train_begin,
                        train_end, test_begin, test_end):
    """
    得到训练和测试的数据
    :param price_list: 价格数组
    :param batch_size: 一次传入训练的批次大小
    :param train_begin: 训练数据起始位置
    :param train_end: 训练数据结束位置
    :param test_begin: 测试数据起始位置
    :param test_end: 测试数据结束位置
    :return:
    """
    # 获取训练数据
    data_train = price_list[train_begin:train_end]
    train_x, train_y = [], []
    for i in range(len(data_train) - input_size):
        x = data_train[i: i + input_size]
        y = data_train[i + input_size]
        train_x.append(x)
        train_y.append([y])

    # 获取测试数据
    data_test = price_list[test_begin:test_end]
    test_x, test_y = [], []
    for i in range(len(data_test) - input_size):
        if len(test_x) < len(data_test) - input_size - many_days:
            x = data_test[i: i + input_size]
            test_x.append(x)
        y = data_test[i + input_size]
        test_y.append([y])
    n_batch = len(train_x) // batch_size  # 整除批次大小
    return n_batch, train_x, train_y, test_x, test_y


def bp(x, keep_prob):
    """
    定义神经网络的结构
    :param x:
    :param keep_prob: 每次参与的神经元百分比
    :return:
    """
    w = tf.Variable(tf.truncated_normal([input_size, 500], stddev=0.1))
    b = tf.Variable(tf.zeros([500]) + 0.1)
    re = tf.matmul(x, w) + b
    l1 = tf.nn.elu(re)  # 激活函数
    l1_drop = tf.nn.dropout(l1, keep_prob)  # keep_prob设为1则百分百的神经元工作,L1作为神经元的输出传入
    w2 = tf.Variable(tf.truncated_normal([500, 30], stddev=0.1))
    b2 = tf.Variable(tf.zeros([30]) + 0.1)
    re2 = tf.matmul(l1_drop, w2) + b2
    l2 = tf.nn.elu(re2)  # 激活函数
    l2_drop = tf.nn.dropout(l2, keep_prob)
    # w3 = tf.Variable(tf.truncated_normal([300, 30], stddev=0.1))
    # b3 = tf.Variable(tf.zeros([30]) + 0.1)
    # re3 = tf.matmul(l2_drop, w3) + b3
    # l3 = tf.nn.elu(re3)  # 激活函数
    # l3_drop = tf.nn.dropout(l3, keep_prob)
    w4 = tf.Variable(tf.random_normal([30, output_size], stddev=0.1))
    b4 = tf.Variable(tf.zeros([output_size]) + 0.1)
    prediction = tf.matmul(l2_drop, w4) + b4
    return prediction


# correct_prediction = (abs(y - prediction) / y) < 0.1
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train_bp(price_list, batch_size, acc_1, acc_5, acc_10,
             train_begin, train_end, test_begin, test_end):
    """
    开始训练网络
    :param price_list: 价格数组
    :param batch_size: 一次训练的批次大小
    :param acc_1: 误差小于等于1%的准确率
    :param acc_5: 误差小于等于5%的准确率
    :param acc_10: 误差小于等于10%的准确率
    :param train_begin: 训练数据起始位置
    :param train_end: 训练数据结束位置
    :param test_begin: 测试数据起始位置
    :param test_end: 测试数据结束位置
    :return:
    """
    x = tf.placeholder(tf.float32, [None, input_size])
    y = tf.placeholder(tf.float32, [None, output_size])
    keep_prob = tf.placeholder(tf.float32)
    lf = tf.Variable(0.01, dtype=tf.float32)  # 学习率定义
    prediction = bp(x, keep_prob)  # 建立网络
    n_batch, train_x, train_y, test_x, test_y = get_train_test_data(price_list, batch_size,
                                                                    train_begin, train_end,
                                                                    test_begin, test_end)
    # 交叉熵
    loss = tf.reduce_mean(tf.square(y - prediction))
    # 梯度下降法
    train_op = tf.train.AdamOptimizer(lf).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1001):  # 迭代1000个周期
            i = 0
            sess.run(tf.assign(lf, 0.01 * 0.95 ** epoch))  # 修改学习率，越来越小
            # 运行得到的loss值
            loss_ = None
            # 分批次出来，batch_xs和batch_ys为每次投入训练的数据
            for batch in range(n_batch):
                batch_xs = train_x[i: i + batch_size]
                batch_ys = train_y[i: i + batch_size]
                i = i + batch_size
                loss_, _ = sess.run([loss, train_op], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1})
            if epoch % 100 == 0:
                print(epoch, loss_)
                # test_predict = []
                bool_list_1 = []
                bool_list_5 = []
                bool_list_10 = []
                for step in range(len(test_x)):
                    x_in = test_x[step]
                    for j in range(many_days):
                        predict_y = sess.run(prediction, feed_dict={x: [x_in], keep_prob: 1})  # 要三维
                        predict_y = predict_y[0]
                        origin_y = test_y[step + j]
                        if j == many_days - 1:
                            # 获取其准确率
                            bool_list_1.append((abs(predict_y - origin_y) / origin_y < 0.01)[0])
                            bool_list_5.append((abs(predict_y - origin_y) / origin_y < 0.05)[0])
                            bool_list_10.append((abs(predict_y - origin_y) / origin_y < 0.1)[0])
                        x_in = np.append(x_in[1:], predict_y)  # 将计算值添加进去
                        # x = [[num] for num in x]
                # 误差小于1%的准确率
                print(len(bool_list_1))
                # cast函数将其转换为float形式
                num_list = (tf.cast(bool_list_1, tf.float32))
                # reduce_mean取平均值，此时True为1，False为0，平均值其实就是准确率
                accuracy = tf.reduce_mean(num_list)
                acc_one_1 = sess.run(accuracy)
                acc_1.append(acc_one_1)
                print(acc_one_1)
                num_list = (tf.cast(bool_list_5, tf.float32))
                accuracy = tf.reduce_mean(num_list)
                acc_one_5 = sess.run(accuracy)
                acc_5.append(acc_one_5)
                print(acc_one_5)
                num_list = (tf.cast(bool_list_10, tf.float32))
                accuracy = tf.reduce_mean(num_list)
                acc_one_10 = sess.run(accuracy)
                acc_10.append(acc_one_10)
                print(acc_one_10)


def training(root_path, file_name, predict_path):
    """
    入口
    :param root_path:
    :param file_name:
    :param predict_path:
    :return:
    """
    veg_data = pd.read_csv(open(root_path + file_name, 'r', encoding='UTF-8'))
    # print(veg_data.price.tolist())
    # get_train_data(veg_data.price.tolist(), 10, 100, 0, 800)
    # get_test_data(veg_data.price.tolist(), 100, 700, 1000)
    price_list = veg_data.price.tolist()
    # 详细解释看train_bp函数
    acc_1 = []
    acc_5 = []
    acc_10 = []
    batch_size = 50
    train_begin = 0
    train_end = 800
    test_begin = 700
    test_end = 1060
    ddd = {'acc_1': acc_1, 'acc_5': acc_5, 'acc_10': acc_10}
    train_bp(price_list, batch_size, acc_1, acc_5, acc_10, train_begin,
             train_end, test_begin, test_end)
    final_data = pd.DataFrame(ddd)
    final_data.to_csv(predict_path + file_name)


def run():
    pwd = os.getcwd()  # 当前的完整路径
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")  # 父目录
    root_path = father_path + '/veg_data_solve_18_12/'
    predict_path = father_path + '/veg_bp_predict_accuracy/'
    # 获取所有文件名
    file_name_list = SolveData.get_file_name(root_path)
    for i in range(len(file_name_list)):
        start_time = time.time()
        training(root_path, file_name_list[i], predict_path)
        end_time = time.time()
        print(file_name_list[i] + ' wastes time ', end_time - start_time, ' s')


if __name__ == '__main__':
    run()
