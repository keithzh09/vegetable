# coding: utf-8
# @author: hongxin
# @date: 18-5-16


"""
mongoDB模型基类定义
"""


from config.mongo_config import db_config_data
from pymongo import MongoClient
import urllib.parse

username = db_config_data['db_user']
password = db_config_data['db_password']
host = db_config_data['db_host']
port = db_config_data['db_port']
db_name = db_config_data['db_name']
client = MongoClient(host, port)
db = client[db_name]
db.authenticate(username, password)






