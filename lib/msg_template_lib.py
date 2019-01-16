# coding: utf-8
# @author: lin
# @date: 18-7-16

import json
from traceback import format_tb


def log_request_template_fun(url, request_data, response_data):
    """
    记录访问接口的信息及放回数据
    :param url: string
    :param request_data:访问信息
    :param response_data: 返回信息
    :return:
    """
    if isinstance(response_data, Exception):
        msg = url + '--' + str(request_data) + ' -- ' + str(format_tb(response_data.__traceback__)) + ' -- ' + str(type(response_data)) + ' -- ' + str(response_data)
    else:
        msg = url + ' -- ' + str(request_data) + ' -- ' + str(response_data)
    return msg


def log_error_template_fun(name, error):
    """
    记录错误日志信息模板
    :return:
    """
    if isinstance(error, Exception):
        msg = name + '--' + str(format_tb(error.__traceback__)) + '--' + str(type(error)) + '--' + str(error)
    else:
        msg = name + '    ' + str(error)
    return msg


def flask_succ_template_fun(data):
    """
    Flask 请求成功时返回的信息模板
    :param data: data的格式必须为dict
    :return:
    """
    msg = {'code': 200, 'msg': 'successfully'}
    if isinstance(data, dict):
        msg.update(data)
        return json.dumps(msg, ensure_ascii=False)
    else:
        raise TypeError("The type must be dict")


def flask_error_template_fun(data):
    """
    Flask 请求失败时返回的信息模板
    :param data: data的格式必须为dict
    :return:
    """
    msg = {}
    if isinstance(data, dict):
        msg.update(data)
        return json.dumps(msg)
    else:
        raise TypeError("The type must be dict")