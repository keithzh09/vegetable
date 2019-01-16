# -*- coding: utf-8 -*-

# Scrapy settings for vegetable project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://doc.scrapy.org/en/latest/topics/settings.html
#     https://doc.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://doc.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'vegetable'

#设置爬虫的基本信息
SPIDER_MODULES = ['vegetable.spiders']
NEWSPIDER_MODULE = 'vegetable.spiders'
ITEM_PIPELINES = {
   'vegetable.pipelines.VegetablePipeline': 300,
}

JOBDIR='memorary'    #设置保存信息的文件夹，可实现爬虫的暂停和重启 

#设置数据库的用户信息，用于读取以建立和数据库的连接
# MYSQL_HOST = 'localhost'
# MYSQL_DBNAME = 'vegetable'
# MYSQL_USER = 'root'
# MYSQL_PASSWD = 'aa134333'
# MYSQL_PORT = 3306

# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'vegetable (+http://www.yourdomain.com)'

# Obey robots.txt rules
ROBOTSTXT_OBEY = False    #设置为不访问robot.txt页面

# Configure maximum concurrent requests performed by Scrapy (default: 16)
# CONCURRENT_REQUESTS = 100

# Configure a delay for requests for the same website (default: 0)
# See https://doc.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
DOWNLOAD_DELAY = 3    #下载延迟为3秒
# The download delay setting will honor only one of:
# CONCURRENT_REQUESTS_PER_DOMAIN = 100
# CONCURRENT_REQUESTS_PER_IP = 100

# Disable cookies (enabled by default)
COOKIES_ENABLED = False     #不存储cookie

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
#DEFAULT_REQUEST_HEADERS = {
#   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#   'Accept-Language': 'en',
#}

# Enable or disable spider middlewares
# See https://doc.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    'vegetable.middlewares.VegetableSpiderMiddleware': 543,
#}

# Enable or disable downloader middlewares
# See https://doc.scrapy.org/en/latest/topics/downloader-middleware.html
#DOWNLOADER_MIDDLEWARES = {
#    'vegetable.middlewares.VegetableDownloaderMiddleware': 543,
#}

#设置爬虫的中间件
DOWNLOADER_MIDDLEWARES = {
#      'vegetable.middlewares.RandomProxyMiddleware':542,      #变换ip的中间件
     'scrapy.downloadermiddlewares.downloadtimeout.DownloadTimeoutMiddleware':350,       #处理下载延迟的中间件
     'vegetable.middlewares.RandomUserAgentMiddleware': 400,       #变换请求头的中间件
     'scrapy.contrib.downloadermiddleware.retry.RetryMiddleware': 500,       #对异常的页面进行重新访问的中间件
     'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware':None, #这里要设置原来的scrapy的useragent为None，否者会被覆盖掉
  }
RANDOM_UA_TYPE='random'
DOWNLOAD_TIMEOUT = 200
# Enable or disable extensions
# See https://doc.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
#}

# Configure item pipelines
# See https://doc.scrapy.org/en/latest/topics/item-pipeline.html
#ITEM_PIPELINES = {
#    'vegetable.pipelines.VegetablePipeline': 300,
#}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://doc.scrapy.org/en/latest/topics/autothrottle.html
#AUTOTHROTTLE_ENABLED = True
# The initial download delay
#AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
#AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://doc.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = 'httpcache'
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'
