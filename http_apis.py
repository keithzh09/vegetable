# coding: utf-8
# @author: lin
# @date: 18-7-22


from k_line import k_line_app
from flask import Flask

app = Flask(__name__)

# 蓝图注册
app.register_blueprint(k_line_app, url_prefix="/customer/")

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=12006, debug=True)
