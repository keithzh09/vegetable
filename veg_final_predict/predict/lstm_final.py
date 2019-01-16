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

pd.set_option('max_colwidth', 5000)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 1000)

# 定义常量
rnn_unit = 50  # hidden layer units
input_size = 1  # 输入一个变量，即前一天的价格
output_size = 1  # 输出一个变量


def get_train_data(price_list, batch_size, time_step, train_begin, train_end):
    """
    得到训练集
    得到的train_x的格式为shape[len(data)-time_step,time_step,输入变量数]
    得到的train_y的格式为shape[len(data)-time_step,time_step,输出变量数]
    batch_size*time_step=数据量
    :param price_list:价格的列表
    :param batch_size:一次训练多少个批次
    :param time_step: 以多少天的数据作为输入
    :param train_begin: 开始位置
    :param train_end: 结束位置
    :return:
    """
    batch_index = []  # 得到每两个批次间的下标
    data_train = np.array(price_list[train_begin:train_end])
    train_x, train_y = [], []  # 训练集
    for i in range(len(data_train) - time_step):
        if i % batch_size == 0:  # 每隔一个批次添加一下下表
            batch_index.append(i)
        x = data_train[i:i + time_step, np.newaxis]  # np.newaxis为array增加一维
        y = data_train[i + 1:i + time_step + 1, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(data_train) - time_step))
    return batch_index, train_x, train_y


def get_test_data(price_list, time_step, test_begin, test_end, many_days=10):
    """
    得到测试集
    其中test_x的格式应与训练数据一致，y应为shape[test_x, 输出变量数]的长度
    :param price_list: 价格列表
    :param time_step: 同上
    :param test_begin: 同上
    :param test_end: 同上
    :param many_days: 滚动预测的天数
    :return:
    """
    data_test = np.array(price_list[test_begin: test_end])
    # mean = np.mean(data_test, axis=0)
    # std = np.std(data_test, axis=0)
    test_x, test_y = [], []
    for i in range(len(data_test) - time_step):
        # test_y应该至少多出many_days个长度，才可进行滚动预测
        if len(test_x) < (len(data_test) - many_days - time_step):
            x = data_test[i:i + time_step, np.newaxis]
            test_x.append(x.tolist())
        y = data_test[i + time_step]
        test_y.append(y)
    test_y = [[i] for i in test_y]
    return test_x, test_y


# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置

weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# ——————————————————定义神经网络变量——————————————————
def lstm(x):
    """
    LSTM模型的建立
    :param x: shape[batch_size, time_step, 输入变量个数]
    :return:
    """
    # tf.reset_default_graph()
    batch_size = tf.shape(x)[0]
    time_step = tf.shape(x)[1]
    w_in = weights['in']
    b_in = biases['in']
    # -1表示第一层靠第二层来决定, 可以说其实第一层就会编程batch_size * time_step
    input_value = tf.reshape(x, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input_value, w_in) + b_in
    # lstm的输入格式即为[batch_size, time_step, 输入变量数目]
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output_rnn, final_states = tf.nn.dynamic_rnn(cell,
                                                 input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)
    # -1表示根据实际情况分配,比如出来的数据为100个,rnn_unit为1,则-1的位置会变为100
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    predict_value = tf.matmul(output, w_out) + b_out
    return predict_value, final_states


def train_lstm(name, acc_1, acc_5, acc_10, price_list, batch_size,
               time_step, train_begin, train_end, test_begin,
               test_end, many_days):
    """
    进行模型的训练以及获取准确率
    :param name: 为了将命名域区分开来,方可实现多个模型的定义,保证每种蔬菜开始预测用的都是初始化的模型
    :param acc_1: 误差小于1%的准确率
    :param acc_5: 误差小于5%的准确率
    :param acc_10: 误差小于10%的准确率
    :param price_list: 价格数组
    :param batch_size: 同时传入的数据批次为多少
    :param time_step: 以几日的数据进行预测
    :param train_begin: 获取训练数据开始的下标
    :param train_end: 获取训练数据结束的下标
    :param test_begin: 获取测试数据开始的下标
    :param test_end: 获取测试数据结束的下标
    :param many_days: 滚动预测的天数
    :return:
    """
    input_x = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    output_y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(price_list, batch_size, time_step, train_begin, train_end)
    test_x, test_y = get_test_data(price_list, time_step, test_begin, test_end, many_days)
    # 命名域区分开来,方可实现多个模型的定义,否则会不知道是哪个
    with tf.variable_scope(name):
        predict_value, _ = lstm(input_x)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(predict_value, [-1]) - tf.reshape(output_y, [-1])))
    lr = tf.Variable(0.01, dtype=tf.float32)  # 学习率定义
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    with tf.Session() as sess:
        # 初始化
        sess.run(tf.global_variables_initializer())
        # 训练1000次，每次以batch_size为一个批次
        for i in range(1001):
            # 对应训练过程中出现的loss值
            loss_ = None
            sess.run(tf.assign(lr, 0.01 * 0.95 ** i))     # 逐渐下降
            for step in range(len(batch_index) - 1):
                # 分别对应传入sess.run()的两个值
                _, loss_ = sess.run([train_op, loss], feed_dict={
                    input_x: train_x[batch_index[step]:batch_index[step + 1]],
                    output_y: train_y[batch_index[step]:batch_index[step + 1]]})
            # 每隔100次则测试一次准确率
            if i % 100 == 0:
                print(i, loss_)
                test_predict = []
                bool_list_1 = []
                bool_list_5 = []
                bool_list_10 = []
                for step in range(len(test_x)):
                    x = test_x[step]
                    # prob代表投入数据预测得到的结果
                    prob = None
                    # 接下来是滚动预测了
                    for j in range(many_days):
                        # 输入要三维
                        prob = sess.run(predict_value, feed_dict={input_x: [x]})
                        # 先把输出可能多维的情况，转换成一维
                        predict = prob.reshape((-1))
                        # 最后一个便是预测对应y的那一天
                        predict_y = predict[-1]
                        origin_y = test_y[step + j]
                        # 当滚动预测到了最后一天了
                        if j == many_days - 1:
                            # 之所以后面需要加个[0]就是因为predict_y和origin_y都是数组，虽然只有一个数
                            bool_list_1.append((abs(predict_y - origin_y) / origin_y < 0.01)[0])
                            bool_list_5.append((abs(predict_y - origin_y) / origin_y < 0.05)[0])
                            bool_list_10.append((abs(predict_y - origin_y) / origin_y < 0.1)[0])
                        # 重置输入x，将预测得到的值加进去，并去掉原来的第一个数
                        x = np.append(x[1:], [predict_y])  # 将计算值添加进去
                        x = [[num] for num in x]
                    predict = prob.reshape((-1))
                    test_predict.extend(predict)
                # 得到误差小于1%的准确率了，tf.cast将值转换为float格式
                num_list = (tf.cast(bool_list_1, tf.float32))
                # reduce_mean得到平均值，此时True已经为1，False已经为0，故平均值就是准确率
                accuracy = tf.reduce_mean(num_list)
                acc_one_1 = sess.run(accuracy)
                acc_1.append(acc_one_1)
                print(acc_one_1)
                # 得到5%的
                num_list = (tf.cast(bool_list_5, tf.float32))
                accuracy = tf.reduce_mean(num_list)
                acc_one_5 = sess.run(accuracy)
                acc_5.append(acc_one_5)
                print(acc_one_5)
                # 得到10%的
                num_list = (tf.cast(bool_list_10, tf.float32))
                accuracy = tf.reduce_mean(num_list)
                acc_one_10 = sess.run(accuracy)
                acc_10.append(acc_one_10)
                print(acc_one_10)
    sess.close()


def training(num, root_path, file_name, predict_path):
    """
    训练及预测准确率入口
    :param num: 为了神经网络中初始化模型定义的一个每种蔬菜不会重复的值
    :param root_path: 源数据存放的目录
    :param file_name: 文件路径，指单个蔬菜的路径
    :param predict_path: 预测得到的准确率存放的位置
    :return:
    """
    veg_data = pd.read_csv(open(root_path + file_name, 'r', encoding='UTF-8'))
    # print(veg_data.price.tolist())
    # get_train_data(veg_data.price.tolist(), 10, 100, 0, 800)
    # get_test_data(veg_data.price.tolist(), 100, 700, 1000)
    price_list = veg_data.price.tolist()
    # 错误率小于1%、5%、10%的准确率
    acc_1 = []
    acc_5 = []
    acc_10 = []
    # 以下各值的意义，在train_lstm函数下有简介
    batch_size = 50
    time_step = 100
    train_begin = 0
    train_end = 800
    test_begin = 700
    test_end = 1060
    many_days = 10
    train_lstm('veg' + str(num), acc_1, acc_5, acc_10, price_list, batch_size, time_step,
               train_begin, train_end, test_begin, test_end, many_days)
    final_data = pd.DataFrame({'acc_1': acc_1, 'acc_5': acc_5, 'acc_10': acc_10})
    final_data.to_csv(predict_path + file_name)


def run():
    pwd = os.getcwd()  # 当前的完整路径
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")  # 父目录
    root_path = father_path + '/veg_data_solve_18_12/'
    predict_path = father_path + '/veg_lstm_predict_accuracy/'
    # 获取所有文件名
    file_name_list = SolveData.get_file_name(root_path)
    for i in range(len(file_name_list)):
        start_time = time.time()
        training(i, root_path, file_name_list[i], predict_path)
        end_time = time.time()
        print(file_name_list[i] + ' wastes time ', end_time-start_time, 's')


if __name__ == '__main__':
    run()
