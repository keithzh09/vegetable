import pymysql
from scrapy.utils.project import get_project_settings  #导入seetings配置

class DBHelper():
    '''这个类也是读取settings中的配置，自行修改代码进行操作'''

    def __init__(self):
        self.settings = get_project_settings()  #获取settings配置，设置需要的信息
        self.use_unicode = True

   #连接mysql
    def connectMysql(self):
        conn = pymysql.connect(host=self.host,
                             port=self.port,
                             user=self.user,
                             passwd=self.passwd,
                             charset=self.charset)
        return conn
    #连接数据库
    def connectDatabase(self):
        conn = pymysql.connect(host=self.settings.get('MYSQL_HOST'),
                             port=self.settings.get('MYSQL_PORT'),
                             user=self.settings.get('MYSQL_USER'),
                             passwd=self.settings.get('MYSQL_PASSWD'),
                             db=self.settings.get('MYSQL_DBNAME'),
                             charset='utf8'
                              )
        return conn

    #创建数据库
    def createDatabase(self):
        conn = self.connectMysql()
        
        sql = "create database if not exists "+self.db
        cur = conn.cursor()
        cur.execute(sql)
        cur.close()
        conn.close()

    #创建数据表
    def createTable(self,sql):
        conn=self.connectDatabase()
        
        cur = conn.cursor()
        cur.execute(sql)
        cur.close()
        conn.close()

    #插入数据
    def insert(self,sql,params):
        conn=self.connectDatabase()

        cur=conn.cursor();
        cur.execute(sql,params)
        
    # 提交，不然无法保存新建或者修改的数据
        conn.commit()
        
        cur.close()
        conn.close()

    #更新数据
    def update(self,sql,params):
        conn=self.connectDatabase()

        cur=conn.cursor()
        cur.execute(sql,params)
        conn.commit()
        cur.close()
        conn.close()

    #删除数据
    def delete(self,sql,params):
        conn=self.connectDatabase()

        cur=conn.cursor()
        cur.execute(sql,params)
        conn.commit()
        cur.close()
        conn.close()
        