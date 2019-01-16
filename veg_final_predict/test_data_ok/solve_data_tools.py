# -*- coding: utf-8 -*-
# @Time    : 19-1-14
# @Author  : lin


"""
一些需要对数据进行处理的函数
"""

from datetime import datetime
import os


class SolveData:
    def __init__(self):
        pass

    @staticmethod
    def tran_datetime_index_to_str(datetime_index):
        # 将索引为date_time的数据的索引转化为str
        if len(datetime_index) > 0:
            str_date = [i.to_pydatetime().strftime('%Y-%m-%d %H:%M:%S') for i in datetime_index]
            str_date = [i[0: 10] for i in str_date]   # 保留年月日即可
        else:
            str_date = datetime_index
        return str_date

    @staticmethod
    def strtime_to_timestamp(strtime):
        """
        字符串时间转时间戳
        :param strtime:
        :return: timestamp
        """
        default_format = '%Y-%m-%d %H:%M:%S'
        if len(strtime) == 19:  # '2018-07-20 12:00:00'
            time_format = default_format
        elif len(strtime) == 10:
            time_format = '%Y-%m-%d'
        elif len(strtime) == 7:
            time_format = '%Y-%m'
        elif len(strtime) == 4:
            time_format = '%Y'
        else:
            time_format = default_format
        timestamp = datetime.strptime(strtime, time_format).timestamp()
        return timestamp

    @staticmethod
    def timestamp_to_strtime(timestamp):
        """
        时间戳转字符串时间， 默认精确到秒
        :param timestamp:
        :return:
        """
        if len(str(timestamp)) == 13:
            timestamp *= 0.001
        strtime = datetime.strftime(datetime.fromtimestamp(timestamp), '%Y-%m-%d %H:%M:%S')
        return strtime

    @staticmethod
    def day_increase(strtime, increase_days):
        """
        日期加减
        :param strtime:
        :param increase_days:
        :return:
        """
        timestamp = SolveData.strtime_to_timestamp(strtime)
        timestamp += increase_days * 86400  # 1 days = 86400s
        time = SolveData.timestamp_to_strtime(timestamp)
        return time

    @staticmethod
    def day_decrease(strtime, decrease_days):
        """
        日期加减
        :param strtime:
        :param decrease_days:
        :return:
        """
        timestamp = SolveData.strtime_to_timestamp(strtime)
        timestamp -= decrease_days * 86400  # 1 days = 86400s
        time = SolveData.timestamp_to_strtime(timestamp)
        return time

    @staticmethod
    def get_file_name(file_dir):
        file_name_list = []
        for root, dirs, files_name in os.walk(file_dir):
            # root为当前目录路径，dirs为当前路径下所有子目录，files_name为当前路径下所有非目录子文件
            if root == file_dir:
                file_name_list = files_name
        return file_name_list
