# -*- coding: utf-8 -*-

# Define your item pipelines here

# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

# import pymysql
# from vegetable.MysqlHandle import DBHelper

# class VegetablePipeline(object):
    
#     def __init__(self):
#         self.db = DBHelper()
    
#     def process_item(self, item, spider):
#         if spider.name == "vegetable":
#             JiangnanInsert = 'insert into Jiangnan(name,place,price,data) VALUES (%s,%s,%s,%s)'
#             params = (item['name'],item['place'],item['price'],item['data'])
#             self.db.insert(JiangnanInsert,params)
            
#         return item

import csv

#将爬取的信息存入csv文件中
class VegetablePipeline(object):
    def process_item(self, item, spider):
        with open('vegetable.csv','a+') as f:
            writer = csv.writer(f)
            writer.writerow((item['name'],item['place'],item['price'],item['data']))
        return item