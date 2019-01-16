import requests
import re
import csv
from lxml import etree 


url  = "http://lishi.tianqi.com/guangzhou/index.html"
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'
}
response = requests.get(url,headers = headers)
html = response.text


selector = etree.HTML(html)
urls_all = selector.xpath('//div[@id="tool_site"]/div[2]/ul[*]/li[*]/a/@href')
urls_need = []
for url in urls_all:
    if re.search('2015|2016|2017|2018',url) and re.search('201607|201606',url) == None:
        urls_need.append(url)


data = []
maxC = []
minC = []
weather = []
windDerection = []
windPower = []
for url in urls_need:
    response = requests.get(url, headers = headers)
    html = response.text
    selector = etree.HTML(html)
    data += selector.xpath('//div[@class="tqtongji2"]/ul[*]/li[1]/a/text()')
    maxC += selector.xpath('//div[@class="tqtongji2"]/ul[*][not(@class)]/li[2]/text()')
    minC += selector.xpath('//div[@class="tqtongji2"]/ul[*][not(@class)]/li[3]/text()')
    weather += selector.xpath('//div[@class="tqtongji2"]/ul[*][not(@class)]/li[4]/text()')
    windDerection += selector.xpath('//div[@class="tqtongji2"]/ul[*][not(@class)]/li[5]/text()')
    windPower += selector.xpath('//div[@class="tqtongji2"]/ul[*][not(@class)]/li[6]/text()')


print(len(data),len(maxC),len(minC),len(weather),len(windDerection),len(windPower))
length = len(data)
with open('/Users/huangguobin/Scrapy/vegetable/weather.csv','w') as f:
    writer =  csv.writer(f)
    for i in range(0,length):
       writer.writerow((data[i],maxC[i],minC[i],weather[i],windDerection[i],windPower[i]))
print("over")