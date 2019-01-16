# -*- coding: utf-8 -*-

# Define here the models for your spider middleware
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/spider-middleware.html

import random
from fake_useragent import UserAgent #这是一个随机UserAgent的包，里面有很多UserAgent

#随机更换请求头的类
class RandomUserAgentMiddleware(object):
    def __init__(self, crawler):
        super(RandomUserAgentMiddleware, self).__init__() #继承RandomUserAgentMiddleware中间件
   
        self.ua = UserAgent()
        self.ua_type = crawler.settings.get('RANDOM_UA_TYPE', 'random') #从setting文件中读取RANDOM_UA_TYPE值
   
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)
   
    def process_request(self, request, spider):   #对请求进行处理
        def get_ua():
            '''Gets random UA based on the type setting (random, firefox…)'''
            return getattr(self.ua, self.ua_type) 
   
        user_agent_random=get_ua()
        request.headers.setdefault('User-Agent', user_agent_random) #这样就是实现了User-Agent的随即变换 

        
#随机更换ip
# class RandomProxyMiddleware(object):  
#     """docstring for RandomProxyMiddleware"""  
#     def process_request(self,request, spider):  
#         '''对request对象加上proxy'''  
#         proxy = self.get_random_proxy()  
#         print("this is request ip:" + proxy)  
#         request.meta['proxy'] = proxy   

#     def process_response(self, request, response, spider):  
#         '''对返回的response处理'''  
#         # 如果返回的response状态不是200，重新生成当前request对象  
#         if response.status != 200:  
#             proxy = self.get_random_proxy()  
#             print("this is response ip:"+proxy)  
#             # 对当前reque加上代理  
#             request.meta['proxy'] = proxy   
#             return request  
#         return response
        
        
#     def get_random_proxy(self):  
#         '''随机从文件中读取proxy'''  
#         while 1:  
#             with open('/Users/huangguobin/py/proxies.txt', 'r') as f:  
#                 proxies = f.readlines()  
#             if proxies:  
#                 break  
#             else:  
#                 time.sleep(1)  
#         proxy = random.choice(proxies).strip()  
#         return proxy
