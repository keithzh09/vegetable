from scrapy.spider import Spider
from scrapy.http import Request
from scrapy.selector import Selector
from vegetable.items import VegetableItem
import time

class VegetableSpider(Spider):
    name = 'vegetable'        #爬虫的名字   
    
    allowed_domains = ["jnmarket.net"]     #允许访问的域名  
    
    offset = 1
    
    url ="http://www.jnmarket.net/import/list-1_"
    
    start_urls = ["http://www.jnmarket.net/import/list-1.html"]      #开始爬取的网址
    
    
    def parse(self,response):
        #使用scrapy的selector，用xpath提取所需信息
        for sel in Selector(response).xpath("//table/tbody/tr"):
            item = VegetableItem()
            item['name'] = sel.xpath("./td[1]/text()").extract()[0]  #注意前面的.
            item['place'] = sel.xpath("./td[2]/text()").extract()[0]
            item['price'] = sel.xpath("./td[3]/text()").extract()[0]
            item['data'] = sel.xpath("./td[5]/text()").extract()[0]
            yield item
        
        #提取网页中所需的网址
        urls = Selector(response).xpath('//a[@class="pageindex"]/@href').extract()
        
        #使用循环对提取的网址发送请求
        for url in urls:
            url = "http://www.jnmarket.net" + url
            print (url)
            yield Request(url, callback=self.parse)  #回调函数为解析函数
#         if self.offset < 15:
#             self.offset += 1
        
#         yield Request(self.url + str(self.offset) + ".html", callback = self.parse)

