# coding: utf-8
# @author: lin
# @date: 18-7-22

from lib.mongo_base_model_lib import db, client
from flask import Blueprint, request
from lib.msg_template_lib import flask_succ_template_fun, flask_error_template_fun

vegetable_all_data_set = db.vegetable_all_data_set

k_line_app = Blueprint("k_line_app", __name__)


@k_line_app.route('k_line_data', methods=['POST'])
def get_k_line_data():
    #暂时说明以json格式请求，现在还略粗糙，之后改进
    try:
        vegetable_name = request.json['vegetable_name']
        start_time = request.json['start_time']  # eg.'15-02-15'
        end_time = request.json['end_time']
    except Exception as error:
        return flask_error_template_fun({'msg': 'imformation error'})
    data_list = []
    for data in vegetable_all_data_set.find({'name': vegetable_name}):
        if start_time <= data['date'] <= end_time:
            a_list = [data['name'], data['date'], data['place'], data['price']]
            data_list.append(a_list)
    count = len(data_list)
    return flask_succ_template_fun({'count': count, 'data': data_list})


if __name__ == '__main__':
    pass
