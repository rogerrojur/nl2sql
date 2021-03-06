#!/usr/bin/env python3
# docker run --name corenlp -d -p 9000:9000 vzhong/corenlp-server
# Wonseok Hwang. Jan 6 2019, Comment added
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import records
import ujson as json
from tqdm import tqdm
import copy
import re

from ordered_set import OrderedSet
from collections import OrderedDict

import itertools

import difflib

import pynlpir as pr
pr.open()
import pkuseg
seg = pkuseg.pkuseg()

import jieba

import http.client
import hashlib
import urllib
import random
import json

client = None

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']
syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE', 'SECTION', 'OP', 'COND', 'QUESTION', 'AGG', 'AGGOPS', 'CONDOPS']



def find_sub_list(sl, l):
    # from stack overflow.
    # TODO ：这个地方应该可以改进
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))

    return results

def check_wv_tok_in_nlu_tok(wv_tok1, nlu_t1):
    """
    Jan.2019: Wonseok
    Generate SQuAD style start and end index of wv in nlu. Index is for of after WordPiece tokenization.

    Assumption: where_str always presents in the nlu.

    return:
    st_idx of where-value string token in nlu under CoreNLP tokenization scheme.
    """
    g_wvi1_corenlp = []
    # nlu_t1_low = [tok.lower() for tok in nlu_t1]
    nlu_t1_low = [tok for tok in nlu_t1]
    for i_wn, wv_tok11 in enumerate(wv_tok1):
        # wv_tok11_low = [tok.lower() for tok in wv_tok11]
        wv_tok11_low = [tok for tok in wv_tok11]
        results = find_sub_list(wv_tok11_low, nlu_t1_low)
        st_idx, ed_idx = results[0]

        g_wvi1_corenlp.append( [st_idx, ed_idx] )

    return g_wvi1_corenlp


def get_end_index(str1, str2):
    # 使用str2对str1的子串进行匹配，最终返回最优的子串的结束位置
    max_ratio = 0
    end_index = 0
    for i in range(1, len(str1)):
        ratio = difflib.SequenceMatcher(None, str1[:i], str2).quick_ratio()
        if max_ratio < ratio:
            max_ratio = ratio
            end_index = i-1 # 包含最后一个字符

    if max_ratio >= 0.6:
        return end_index    # 如果找到则返回结束的index，表示[0, end_index]是和str2最相似的字符串
    else:
        return -1   # 如果不相似，则返回-1，表示没有找到对应的字符串


def get_similarity(str1, str2):
    return difflib.SequenceMatcher(None, str1, str2).quick_ratio()


def find_str_in_list(s, l):
    # from stack overflow.
    # s 包括 l 中的一个连续的子序列
    results = []
    stage = '-1'
    s_len, l_len = len(s), len(l)
    # for ix, token in enumerate(l):
    #     if token.startswith(s):
    #         results.append((ix, ix))

    # if len(results) != 0:
    #     return results

    # 第一遍：假设存在完全匹配
    if not results:
        stage = '0'
        # print('1-th iter')
        for ix, token in enumerate(l):
            if s[0] == token[0]:
                tmp_str, tmp_idx = '', ix
                while len(tmp_str) < s_len and tmp_idx < l_len:
                    tmp_str += l[tmp_idx]
                    tmp_idx += 1
                if tmp_str == s:
                    results.append((ix, tmp_idx-1))

    # 如果不存在完全匹配，则进行模糊匹配, 限制条件：必须开头是相同的
    if not results:
        stage = '1'
        for ix, token in enumerate(l):
            if s[0] == token[0]:
                max_ratio = 0
                end_index = -1
                for end_ix in range(ix, l_len):
                    tmp_str = ''.join(l[ix:(end_ix+1)])
                    ratio = get_similarity(tmp_str, s)
                    if ratio > max_ratio:
                        max_ratio = ratio
                        end_index = end_ix
                if max_ratio >= 0.6:
                    results.append((ix, end_index))

    # 2017年 2017
    if not results:
        stage = '2'
        # print('2-th iter')
        for ix, token in enumerate(l):
            if s[0] == token[0]:
                if token.startswith(s):
                    results.append((ix, ix))

    # 如果不存在完全匹配, 如 携 程， 携程网; 途牛， 途牛网;
    if not results:
        # print('3-th iter')
        stage = '3'
        for ix, token in enumerate(l):
            if s[0] == token[0]:
                for i in range(ix, l_len):
                    tmp_str = ''.join(l[ix:i+1])    # 把第i个token包含进去
                    if not s.startswith(tmp_str) and i == ix:
                        break
                    if not s.startswith(tmp_str) and i > ix:
                        tmp_str = ''.join(l[ix:i])  # 把第i个token剔除，只包含到i-1个token
                        if len(tmp_str) >= s_len / 2:
                            results.append((ix, i-1))
                            break
            if results:
                break

    # if not results and not is_all_number_word(s):
    if not results:
        # print('4-th iter')
        stage = '4'
        # 放大招：合并字符串，记录每个字符所在的token
        ss = ''
        c_to_t_dic = {}   # char-index : token index
        # 将ss中的每个字符对应到token的索引index
        for t_ix, token in enumerate(l):
            for i in range(len(token)):
                c_to_t_dic[len(ss) + i] = t_ix
            ss += token
        # 如果找的是字符串，不是数字
        # if not is_all_number_word(s):
        # 寻找的字符串最短为2，设定不能寻找单字, 如果原名字长为8，则寻找的最短长度为3
        search_length = 3
        if len(s) <= 2:
            search_length = len(s)
        elif len(s) <= 7:
            search_length = 2
        for sub_len in range(len(s), search_length-1, -1):
            if results:
                break
            for si in range(0, len(s) - sub_len + 1):
                sub_str = s[si:si+sub_len]
                if not results:
                    tmp_start = ss.find(sub_str)
                    if tmp_start != -1:
                        results.append((c_to_t_dic[tmp_start], c_to_t_dic[tmp_start+sub_len-1]))
                        stage='4'
                else:
                    break

    if not results:
        stage = '0'

    return results, stage


def check_wv_in_nlu_tok(wv_str_list, nlu_t1):
    """
    不对where value进行annote！
    原因：'铁龙物流'在question中的token可能是 铁龙 / 物流， 而单独annote的时候可能是 铁 / 龙 / 物流
    """
    g_wvi1_corenlp = []
    for i_wn, wv_str in enumerate(wv_str_list):
        # stage: 找到子串的阶段，方便调试，字符串表示
        results, stage = find_str_in_list(wv_str, nlu_t1)
        st_idx, ed_idx = results[0] # 选择第1个元素，忽略后面的

        g_wvi1_corenlp.append( [st_idx, ed_idx] )

    return g_wvi1_corenlp, stage


def is_all_number_word(s):
    if not s:
        return False
    for c in s:
        if c not in '两零一二三四五六七八九十百千万亿0123456789':
            return False
    return True


dic1 ={u'零':0, u'一':1, u'二':2, u'三':3, u'四':4, u'五':5, u'六':6, u'七':7, u'八':8, u'九':9, u'十':10, u'百':100, u'千':1000, u'万':10000,
       u'0':0, u'1':1, u'2':2, u'3':3, u'4':4, u'5':5, u'6':6, u'7':7, u'8':8, u'9':9,
                u'壹':1, u'贰':2, u'叁':3, u'肆':4, u'伍':5, u'陆':6, u'柒':7, u'捌':8, u'玖':9, u'拾':10, u'佰':100, u'仟':1000, u'萬':10000,
       u'亿':100000000}

def getResultForDigit(a):
    # '八一'
    if len(a) == 2 and a[0] in '三四五六七八九' and a[1] in '零一二三四五六七八九':
        return a
    if len(a) > 0 and a[0] == '两':
        a = '二' + a[1:]
    if len(a) >= 3 and a[-1] in '一二三四五六七八九':
        if a[-2] == '万': a += '千'
        elif a[-2] == '千': a += '百'
        elif a[-2] == '百': a += '十'
        else: pass
    # if a[0] in '亿万千百':
    #     return a
    count = 0 
    result = 0
    tmp = 0
    Billion = 0  
    try:
        while count < len(a):
            tmpChr = a[count]
            #print tmpChr
            tmpNum = dic1.get(tmpChr, None)
            #如果等于1亿
            if tmpNum == 100000000:
                result = result + tmp
                result = result * tmpNum
                #获得亿以上的数量，将其保存在中间变量Billion中并清空result
                Billion = Billion * 100000000 + result 
                result = 0
                tmp = 0
            #如果等于1万
            elif tmpNum == 10000:
                result = result + tmp
                result = result * tmpNum
                tmp = 0
            #如果等于十或者百，千
            elif tmpNum >= 10:
                if tmp == 0:
                    tmp = 1
                result = result + tmpNum * tmp
                tmp = 0
            #如果是个位数
            elif tmpNum is not None:
                tmp = tmp * 10 + tmpNum
            count += 1

        result = result + tmp
        result = result + Billion
    except:
        return a
    return str(result)

def is_decimal_number_word(token):
    point_idx = token.find('点')
    if point_idx < 1 or point_idx == len(token) - 1:
        return False
    tt = token.split('点')
    if len(tt) != 2:
        return False

    # 一点六四
    return is_all_number_word(tt[0]) and is_all_number_word(tt[1])


zh_digit_dic = {'零':'0', '一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八':'8', '九':'9', 
            '1':'1', '0':'0','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','幺':'1'}


helper = lambda x : ''.join([zh_digit_dic[i] for i in x])


def getResultForDecimal(token):
    res = ''
    try:
        tt = token.split('点')
        res = getResultForDigit(tt[0]) + '.'
        for c in tt[1]:
            if c in zh_digit_dic:
                res += zh_digit_dic[c]
            else:
                res += c
    except:
        return token
    return res


def get_numberical_value(text):
    if len(text) == 0:
        return None

    result = None
    is_numberical = True
    for c in text:
        if c not in '0123456789.-':
            is_numberical = False
            break
    if is_numberical:
        return text

    is_str = True
    for c in text:
        # 负十
        if c not in '零一二三四五六七八九十百千万亿点两负':
            is_str = False

    # 包含 点
    if is_str and text.find('点') != -1:
        result = getResultForDecimal(text) if text[0] != '负' else '-'+getResultForDecimal(text[1:])
        for c in result:
            if c not in '0123456789.-':
                result = None
                break
        return result

    # 不包含 点 
    if is_str and text.find('点') == -1:
        result = getResultForDigit(text) if text[0] != '负' else '-'+getResultForDigit(text[1:])
        for c in result:
            if c not in '0123456789-':
                result = None
                break
        return result
    
    return None


def pre_with_change_process(token_list):
    results = []
    ix = -1
    dic = zh_digit_dic
    while ix < len(token_list) - 1:
        ix += 1
        token = token_list[ix]

        # 去除空格
        if token == ' ':
            continue

        ############ 时间处理 时间处理 时间处理 #############
        if len(token) == 1 and token == '年' and ix > 0:
            last_token = token_list[ix-1]
            # 16/年；
            if ix > 0 and len(last_token) == 2 and str.isdigit(last_token):
                results[-1] = '20' + results[-1]  if int(results[-1]) <= 20 else results[-1]  # 16变成2016
                results.append('年')
                continue
            # 一六/年；
            if ix > 0 and len(last_token) == 2 and last_token[0] in dic and last_token[1] in dic:
                s_value = dic[last_token[0]] + dic[last_token[1]]
                pre_tmp_str = '20' if int(s_value) <= 20 else '19'
                tmp_str = pre_tmp_str + s_value
                results[-1] = tmp_str
                results.append('年')
                continue
        # 一六年；一二年
        if len(token) == 3 and token[-1] == '年' and token[0] in dic and token[1] in dic:
            s_value = dic[token[0]] + dic[token[1]]
            pre_tmp_str = '20' if int(s_value) <= 20 else '19'
            tmp_str = pre_tmp_str + s_value
            results.append(tmp_str)
            results.append('年')
            continue

        ############ 数字处理 数字处理 数字处理 #############
        # 百分之/39.56；百分之/五; 百分之/十二; 百分之/负十
        if len(token) == 3 and token == '百分之' and ix+1 < len(token_list):
            val = get_numberical_value(token_list[ix+1])
            # 如果value是一个有效的数字
            if val != None:
                results.append(val)
                results.append('%')
                ix += 1
                continue

        # 处理带有 亿 的数字
        # 对于 1/亿 这种不进行处理
        if len(token) > 1 and token[-1] == '亿':
            val = get_numberical_value(token[:-1])
            if val != None:
                results.append(val)
                results.append('亿')
                continue
        # 处理 一百/亿 这种数据
        if len(token) == 1 and token == '亿' and ix > 0:
            val = get_numberical_value(token_list[ix-1])
            if val != None:
                results[-1] = val
                results.append('亿')
                continue

        # 处理带有 万 的数字 
        # 对于 一百万，两千万, 百万 这种 万 不进行处理， 五/千万
        if len(token) > 1 and token[-1] == '万':
            val = get_numberical_value(token[:-1])
            # 三千万；一千万; 前面大于100的都这样处理，前面小于100的，把万进行转化，如把 两万 一万 转化为 2，0000， 1，0000
            if val != None:
                # 五/千万;
                if ix > 0 and get_numberical_value(token_list[ix-1]) != None and token in ['千万','百万','十万']:
                    results[-1] = get_numberical_value(token_list[ix-1])
                    tmp_val = getResultForDigit(token)
                    results.append(tmp_val[1:-4])   # 去掉首个1
                    results.append(tmp_val[-4:])
                    continue
                if eval(val) >= 100:
                    results.append(val)
                    results.append('万')
                else:   # 对于 一万/两万 这种进行转化10000, 20000
                    results.append(val)
                    results.append('0000')
                continue
        # 处理 4/万;十/万； 这种数据; 
        if len(token) == 1 and token == '万' and ix > 0:
            val = get_numberical_value(token_list[ix-1])
            if val != None and eval(val) < 100:
                results[-1] = val
                results.append('0000')
                continue

        # 处理带有 千 的数字
        # 三千；
        if len(token) > 1 and token[-1] == '千':
            val = get_numberical_value(token[:-1])
            if val != None:
                results.append(val)
                results.append('000')
                continue
        # 三/千;
        if len(token) == 1 and token == '千' and ix > 0:
            val = get_numberical_value(token_list[ix-1])
            if val != None:
                results[-1] = val
                results.append('000')
                continue

        # 处理 70000000, 7000000这种数字
        if len(token) > 4 and token[-4:] == '0000':
            if len(token) > 8 and token[-8:-4] == '0000':
                results.append(token[:-8])
                results.append('0000')
            elif len(token) > 7 and token[-7:-4] == '000':
                results.append(token[:-7])
                results.append('000')
            elif len(token) > 6 and token[-6:-4] == '00':
                results.append(token[:-6])
                results.append('00')
            else:
                results.append(token[:-4])
            results.append('0000')
            continue


        # 处理数字, 对于长度大于2的如 零点二 转化成数字
        if len(token) > 2:
            val = get_numberical_value(token)
            if val != None:
                results.append(val)
                continue


        ############ 钱的问题 钱的问题 钱的问题 #############
        # 一/元；五十/块；一点六/元; 20/万/元,处理到万的时候变成0000，需要做判断
        # 十二/块/五/毛；十/一/块/六
        if len(token) == 1 and token in '元块' and ix > 0:
            val = get_numberical_value(token_list[ix-1])
            if val != None:
                if eval(val) == 0 and len(token_list[ix-1]) > 0:
                    pass
                elif eval(val) == 0 and token_list[ix-1] == '亿':
                    pass
                else:
                    results[-1] = val
                    results.append(token)
                    continue


        # 处理钱的问题
        # 十二/块/五/毛；一块/六/毛；
        if len(token) > 1 and token[-1] in '元块':
            val = get_numberical_value(token[:-1])
            tmp_val = None
            if ix < len(token_list) - 1:
                tmp_val = get_numberical_value(token_list[ix+1])
            if val != None and tmp_val != None:
                results.append(val + '.' + tmp_val)
                results.append(token[-1])
                ix += 1
                if ix < len(token_list) - 1 and token_list[ix+1] in '角毛':
                    ix += 1
                continue
            if val != None:
                results.append(val)
                results.append(token[-1])
                continue

        # 五/角/钱；五/毛/钱；
        if len(token) == 1 and token in '角毛' and ix > 0:
            val = get_numberical_value(token_list[ix-1])
            if val != None:
                results[-1] = '0.' + val
                results.append('元')
                continue


        ## 其他
        if len(token) == 1 and token in '个倍人' and ix > 0:
            val = get_numberical_value(token_list[ix-1])
            if val != None:
                results[-1] = val
                results.append('个')
                continue

        if len(token) > 1 and token[0] == '第':
            results.append('第')
            results.append(token[1:])
            continue

        # 如果不符合上述规则，则直接添加到results列表
        results.append(token)

    # 去除空的token
    copy = []
    for r in results:
        r = r.strip()
        if r != '' and r != ' ':
            copy.append(r)

    return copy


def pre_no_change_process(token_list):
    # 不改变句子本身，只对一些特殊的token进行进一步的细粒度划分
    results = []
    for token in token_list:
        # 针对原数据集里面的数字和汉字混合进行处理，如6月；2012年
        findit = False
        if token[0] in '1234567890.-':
            for i in range(len(token)):
                if token[i] not in '1234567890.':
                    results.append(token[:i])
                    results.append(token[i:])
                    findit = True
                    break
            if not findit:
                results.append(token)
            continue

        # 如果出现河南省等，则将名字和后面的等级分开
        if token[-1] in '省市区县' and len(token) > 1:
            results.append(token[:-1])
            results.append(token[-1])
            continue

        if token == '部门':
            results.append('部')
            results.append('门')
            continue

        if token[-1] in '个倍' and len(token) > 1:
            results.append(token[:-1])
            results.append(token[-1])
            continue

        results.append(token)

    # 去除空的
    copy = []
    for r in results:
        r = r.strip()
        if r != '' and r != ' ':
            copy.append(r)

    return copy


def remove_null(token_list):
    # 去除空的
    copy = []
    for r in token_list:
        r = r.strip()
        if r != '' and r != ' ':
            copy.append(r)

    return copy



def seg_summary(org_str, nlpir_list, pkuseg_list, jieba_list=None):
    """jieba_list是搜索引擎方式，可能和原始字符串不对应，我们需要亿org_str为基础，综合后两个列表"""
    spilt_indices = set()
    curr_len = 0
    spilt_indices.add(0)
    for token in nlpir_list:
        curr_len += len(token)
        spilt_indices.add(curr_len)

    curr_len = 0
    for token in pkuseg_list:
        curr_len += len(token)
        spilt_indices.add(curr_len)

    curr_len = 0
    if jieba_list != None:
        for token in jieba_list:
            ix = org_str.find(token) + len(token)
            if ix != -1:
                spilt_indices.add(ix)

    spilt_indices = sorted(list(spilt_indices))

    results = []
    for i in range(len(spilt_indices)-1):
        results.append(org_str[spilt_indices[i]:spilt_indices[i+1]])

    return results


def post_with_change_process(token_list):
    ix = -1
    results = []
    while ix < len(token_list) - 1:
        ix += 1
        token = token_list[ix]

        # 将中文数字token转化为数字
        if token[0] in '0123456789.-零一二三四五六七八九十百千万亿点两负千万百亿':
            val = get_numberical_value(token)
            try:
                if val != None:
                    tmp_val = eval(val)
                    if tmp_val > 10 and tmp_val != 100 and tmp_val != 1000 and tmp_val != 10000:
                        # 对于 百 千 等常见字不处理
                        results.append(val)
                        continue
            except:
                pass

        results.append(token)

    copy = remove_null(results)

    return copy


def find_str_full_match(s, l):
    # from stack overflow.
    # s 包括 l 中的一个连续的子序列
    results = []
    s_len, l_len = len(s), len(l)

    if not results:
        # print('1-th iter')
        for ix, token in enumerate(l):
            if len(s) == 0 or len(token) == 0:
                print(l)
                continue
            if s[0] == token[0]:
                tmp_str, tmp_idx = '', ix
                while len(tmp_str) < s_len and tmp_idx < l_len:
                    tmp_str += l[tmp_idx]
                    tmp_idx += 1
                if tmp_str == s:
                    # 找到完全匹配
                    return True
    return False


def wvi_conflict(replace_list, wvi_list, ix):
    if not replace_list:
        return False
    conflict = False
    for e in replace_list:
        # 如果没有交集
        wvi = wvi_list[ix]
        tmp = wvi_list[e]
        if wvi[0] > tmp[1] or wvi[1] < tmp[0]:
            pass
        else:
            conflict = True
            break
    return conflict


def get_unmatch_set(token_list, wv_str_list, wvi_list):
    replace_list = []
    for ix, wv_str in enumerate(wv_str_list):
        # 对于不完全匹配的都加进去
        if not find_str_full_match(wv_str, token_list):
            # 如果替换列表为空，或者加入替换列表的数据和已经存在的数据没有冲突
            if not wvi_conflict(replace_list, wvi_list, ix):
                replace_list.append(ix)
    # 返回替换列表，列表中全部是wvi_list中的index
    return replace_list


def replace_unmatch_set(token_list, wv_list, wvi_list, replace_list):
    flag_list = [-1 for _ in range(len(token_list))]
    for e in replace_list:
        wvi = wvi_list[e]
        for p in range(wvi[0], wvi[1]+1):
            flag_list[p] = e
    # flag_list: [-1,-1,-1,2,2,-1,-1,1,1,-1,-1], 非-1表示需要替换wvi_list中的值

    results_list = []
    for i, flag in enumerate(flag_list):
        if flag == -1:
            results_list.append(token_list[i])
        else:
            # 在需要进行替换的位置进行替换
            if (flag >= 0 and i == 0) or (flag >= 0 and i > 0 and flag_list[i-1] == -1):
                results_list.append(wv_list[flag])

    return results_list


words_dic = {'诶，':'','诶':'','那个':'','那个，':'', '呀':'','啊':'','呃':'', '鹅厂':'腾讯', '企鹅公司':'腾讯公司',
            '马桶台':'湖南芒果TV', '荔枝台':'江苏卫视', '北上广':'北京和上海和广州','北上':'北京和上海','北京，深圳':'北京和深圳',
            '厦大':'厦门大学', '中大':'中山大学', '广大':'广州大学', '东航':'东方航空', '国图':'国家图书馆',
            '内师大':'内蒙古师范大学','武大':'武汉大学','中科大':'中国科学技术大学','欢乐喜和剧人':'欢乐喜剧人',
            '两个人':'两人','啥':'什么','市价盈利比率':'市盈率',
            '华师':'华南师范大学',
            '本科或者本科以上':'本科及本科以上','并且':'而且','为负':'小于0','为正':'大于0','陆地交通运输':'陆运','亚太地区':'亚太',
            '负数':'小于0','两万一':'21000','辣椒台':'江西卫视','一二线城市':'一线城市和二线城市','二三线城市':'二线城市和三线城市',
            '世贸':'世茂','山职':'山东职业学院','安徽职院':'安徽职业技术学院','冯玉祥自传':'冯玉祥自述',
            '科研岗位1，2':'科研岗位01和科研岗位02','科研岗位1':'科研岗位01','水关':'水官','上海交大':'上海交通大学',
            '毫克':'mg','写的':'著的','3A':'AAA','红台节目':'江苏卫视','超过':'大于','青铜器辨伪三百例上下':'青铜器辨伪三百例上和青铜器辨伪三百例下',
            '阅读思维人生':'“阅读•思维•人生”','台湾':'中国台湾','建行':'建设银行','蜂制品':'蜂产品',
            '上周或者上上周环比':'上周环比或者环比上上周','上周或上上周环比':'上周环比或者环比上上周',
            '去年':'2018年','幺':'一','二零':'',
            '收盘的价格':'收盘价', '收盘价格':'收盘价','EP2011':'PE2011',
            '不好意思，':'', '请问一下':'', '请问':'', '打扰一下，':'',
            '你好，':'', '你好':'', '你知道':'',  
            '的一个':'', '它的一个':'', '他的一个':'',
            '能不能':'',
            }


def is_valid_char(uchar):
    # 中文，英文，数字，字母
    # 去除特殊字符，还有空格，换行符，制表符
    if u'\u4e00' <= uchar <= u'\u9fa5' or u'\u0030' <= uchar <= uchar<=u'\u0039' \
        or u'\u0041' <= uchar <= u'\u005a' or u'\u0061' <= uchar <= u'\u007a':
        return True

def is_punctuation_en(uchar):
    return uchar in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

def is_punctuation_ch(uchar):
    return uchar in '！？｡。＂·＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〜〝〞–—‘\'‛“”„‟…‧・﹏.Ⅱα•Ⅳ¥'


def replace_words(s):
    for key in words_dic:
        if s.find(key) != -1:
            s = s.replace(key, words_dic[key])
    return s


def remove_special_char(s):
    new_s = ''
    for c in s:
        if is_punctuation_en(c) or is_punctuation_ch(c) or is_valid_char(c):
            new_s += c
    return new_s


def _change_pattern_time(s):
    # [0-9]{2}年到[0-9]{2}年
    mp = re.compile('[0-9]{2}年到[0-9]{2}年')
    mp_digit = re.compile('[0-9]{2}')

    target = ''
    res = ''
    if mp.search(s):
        res = mp.search(s).group()
        sy, ey = mp_digit.findall(res)
        sy, ey = int('20'+sy), int('20'+ey)
        # 2018 2019 2020
        if ey - sy > 3:
            return s
        for i in range(sy, ey+1):
            tmp = str(i)[2:] + '年'
            target += tmp
            if i < ey:
                target += '和'
        s = s.replace(res, target)

    # [0-9]{1,2}月[0-9]{1,2}号[到至][0-9]{1,2}号
    mp = re.compile('[0-9]{1,2}月[0-9]{1,2}[号日][到至][0-9]{1,2}[号日]')
    mp_digit = re.compile('[0-9]+')
    target, res = '', ''
    if mp.search(s):
        res = mp.search(s).group()
        m, d1, d2 = mp_digit.findall(res)
        target = m + '月' + d1 + '日' + '-' + d2 + '日'
        s = s.replace(res, target)

    # [0-9]+年[0-9]+[到-][0-9]+月

    return s


def _change_pattern_compare(s):
    # 在[0-9]+.{0,4}以上
    mp1 = re.compile('在[0-9]+.{0,4}以上')

    target, res = '', ''
    if mp1.search(s):
        res = mp1.search(s).group()
        target = '大于' + res[1:-2]
        
    s = s.replace(res, target)

    mp2 = re.compile('在[0-9]+.{0,4}以下')
    target, res = '', ''
    if mp2.search(s):
        res = mp2.search(s).group()
        target = '小于' + res[1:-2]
        
    s = s.replace(res, target)

    return s


def _replace_pattern(s, p1, target):
    # 将p1模式的字符串替换成s字符串
    mp = re.compile(p1)
    if mp.search(s):
        s = s.replace(mp.search(s).group(), target)
    return s


def _change_pattern_other(s):
    s = _replace_pattern(s, '股票(交易)?的?价格', '股价')
    s = _replace_pattern(s, '最?(近|新)一个?(周|星期)', '近7日')
    s = _replace_pattern(s, '日的?成交量?', '日成交')
    s = _replace_pattern(s, '平均每天成交量?', '日均成交')
    s = _replace_pattern(s, '每(日|天)的?成交', '日成交')
    s = _replace_pattern(s, '上一(周|星期)|上个?(礼拜|星期)', '上周')
    s = _replace_pattern(s, '这一?(周|星期)|这个?(礼拜|星期)也?', '本周')
    s = _replace_pattern(s, '你好啊?，?|诶+，', '')
    s = _replace_pattern(s, '或者?是?', '或')
    s = _replace_pattern(s, '这一?本书?的?', '')
    s = _replace_pattern(s, '有没有', '')
    s = _replace_pattern(s, '一平米?', '每平')
    s = _replace_pattern(s, '每?一股', '每股')
    s = _replace_pattern(s, '这一?期', '本期')
    s = _replace_pattern(s, '上一?期', '上期')
    s = _replace_pattern(s, '合格不|是否合格', '结果')
    s = _replace_pattern(s, '(青岛)?雀巢', '青岛雀巢')
    s = _replace_pattern(s, '久光百货(有限公司)?', '久光百货有限公司')
    s = _replace_pattern(s, '慧奔鞋业(有限公司)?', '慧奔鞋业有限公司')
    s = _replace_pattern(s, '什么时候', '什么时间')
    s = _replace_pattern(s, '[哪那]几个|哪一些', '哪些')
    s = _replace_pattern(s, '几个', '多少')
    s = _replace_pattern(s, '哪家|哪一个|哪一位', '哪个')
    s = _replace_pattern(s, '有几', '有多少')
    s = _replace_pattern(s, '不[大高]于|[低少]于|不足|不超|没破|没大于', '小于')
    s = _replace_pattern(s, '高于|不[低少小]于|超过|多于', '大于')
    s = _replace_pattern(s, '一共', '总共')
    s = _replace_pattern(s, '的?下午一点钟', '下午13:00')
    s = _replace_pattern(s, '我国', '中国')
    s = _replace_pattern(s, '的数量', '数量')
    s = _replace_pattern(s, '是谁著的', '作者是谁')
    s = _replace_pattern(s, '西安市?神电(电器)?', '西安神电电器')
    s = _replace_pattern(s, '并且|此外|同时', '而且')
    s = _replace_pattern(s, '月的(时候)?', '月')
    s = _replace_pattern(s, '年的(时候)?', '年')
    s = _replace_pattern(s, '平均价格', '均价')
    s = _replace_pattern(s, '16岁?.35岁', '16至35岁、普通招生计划应届毕业生不限')
    s = _replace_pattern(s, '2012年1-5月', '2012:1-5')
    s = _replace_pattern(s, '2月23号下午13:00', '2月23日下午13:00')

    mp = re.compile('(麻烦)?请?你?(可以|能|能不能)?(就是)?(帮|给|告诉)?我?已?(想|想要|还想|可不可以|能不能)?(了解|查一查|查查|查询|查|知道|看看|看|列列|咨询|问问|问|说说|说|数数|数|列一下|算算|算|统计)(一下|到)?你?(就是)?')
    if mp.search(s):
        tmp_str = mp.search(s).group()
        if len(tmp_str) >= 2:
            s = s.replace(mp.search(s).group(), '')

    mp = re.compile('突?破[一二三四五六七八九十1-9]')
    if mp.search(s):
        tmp_str = mp.search(s).group()
        s = s.replace(mp.search(s).group()[:-1], '大于')

    s = _replace_pattern(s, '麻烦?请?你?(能|可以|方便)?跟?(告诉我|帮我|给我|告知)(算|找找|查查)?(一下)?', '')
    s = _replace_pattern(s, '^(的是|，|查|麻烦跟?|就是，?|一下|们，?|我就|我只)', '')
    s = _replace_pattern(s, '一下|谢谢，?。?', '')
    # s = _replace_pattern(s, '高于|')

    return s


def _chnage_pattern_share(s):
    # [0-9]+年(还有|和|，)[0-9]+年(还有|和|，)[0-9]+年(每股盈余|每股收益|EPS)
    mp = re.compile('[0-9]+年(还有|和|，)?[0-9]+年(还有|和|，)?[0-9]+年的?(每股盈余|每股收益|EPS)')
    mp_digit = re.compile('[0-9]+')
    target, res = '', ''
    if mp.search(s):
        res = mp.search(s).group()
        y1, y2, y3 = mp_digit.findall(res)
        target = y1 + '年EPS和' + y2 + '年EPS和' + y3 + '年EPS'
        s = s.replace(res, target)

    # [0-9]+年(还有|和|，)[0-9]+年(还有|和|，)[0-9]+年(市盈率|本益比|PE)
    mp = re.compile('[0-9]+年(还有|和|，)?[0-9]+年(还有|和|，)?[0-9]+年的?(市盈率|本益比|PE)')
    mp_digit = re.compile('[0-9]+')
    target, res = '', ''
    if mp.search(s):
        res = mp.search(s).group()
        y1, y2, y3 = mp_digit.findall(res)
        target = y1 + '年PE和' + y2 + '年PE和' + y3 + '年PE'
        s = s.replace(res, target)

    # [0-9]+年(还有|和|，)[0-9]+年(每股盈余|每股收益|EPS)
    mp = re.compile('[0-9]+年(还有|和|，)?[0-9]+年的?(每股盈余|每股收益|EPS)')
    mp_digit = re.compile('[0-9]+')
    target, res = '', ''
    if mp.search(s):
        res = mp.search(s).group()
        y1, y2 = mp_digit.findall(res)
        target = y1 + '年EPS和' + y2 + '年EPS'
        s = s.replace(res, target)

    # [0-9]+年(还有|和|，)[0-9]+年(市盈率|本益比|PE)
    mp = re.compile('[0-9]+年(还有|和|，)?[0-9]+年的?(市盈率|本益比|PE)')
    mp_digit = re.compile('[0-9]+')
    target, res = '', ''
    if mp.search(s):
        res = mp.search(s).group()
        y1, y2 = mp_digit.findall(res)
        target = y1 + '年PE和' + y2 + '年PE'
        s = s.replace(res, target)

    # [一二三四五六七八九零]{2,}年(还有|和|，)[一二三四五六七八九零]{2,}年(每股盈余|每股收益|EPS)
    mp = re.compile('[一二三四五六七八九零]{2,}年(还有|和|，)[一二三四五六七八九零]{2,}年(每股盈余|每股收益|EPS)')
    mp_digit = re.compile('[一二三四五六七八九零]+')
    target, res = '', ''
    if mp.search(s):
        res = mp.search(s).group()
        y1, y2 = mp_digit.findall(res)
        target = y1 + '年EPS和' + y2 + '年EPS'
        s = s.replace(res, target)

    # [一二三四五六七八九零]{2,}年(还有|和|，)[一二三四五六七八九零]{2,}年(市盈率|本益比|PE)
    mp = re.compile('[0-9]+年(还有|和|，)?[0-9]+年的?(市盈率|本益比|PE)')
    mp_digit = re.compile('[一二三四五六七八九零]+')
    target, res = '', ''
    if mp.search(s):
        res = mp.search(s).group()
        y1, y2 = mp_digit.findall(res)
        target = y1 + '年PE和' + y2 + '年PE'
        s = s.replace(res, target)

    return s


def change_special_pattern(s):
    """
    改变一些固有的语句格式，如 不是...吗, 本益比，平均市盈率
    """
    s = _change_pattern_time(s)
    s = _change_pattern_compare(s)
    s = _change_pattern_other(s)
    s = _chnage_pattern_share(s)
    
    return s


def find_str_wvi_full_match(s, l):
    # from stack overflow.
    # s 包括 l 中的一个连续的子序列
    results = []
    s_len, l_len = len(s), len(l)

    if not results:
        for ix, token in enumerate(l):
            if len(token) == 0:
                continue
            if s[0] == token[0]:
                tmp_str, tmp_idx = '', ix
                while len(tmp_str) < s_len and tmp_idx < l_len:
                    tmp_str += l[tmp_idx]
                    tmp_idx += 1
                if tmp_str == s:
                    # 找到完全匹配
                    return [ix, tmp_idx-1]
    return None


def tokens_full_match(token_list, words, order):
    """
    函数：如果words中有word出现在token_list中，则对word对应的token进行聚合
    参数：
        words：候选词列表
        token_list: 被替换的词列表
    注：该过程不会对token_list本身作出改变，只进行聚合，不会进行变形等操作
    """
    new_list = token_list
    for word in words:
        if len(word) == 0:
            continue
        wvi = find_str_wvi_full_match(word, new_list)
        if wvi == None:
            continue
        new_list = new_list[:wvi[0]] + [''.join(new_list[ wvi[0] : wvi[1]+1 ])] + new_list[wvi[1]+1:]
    return new_list


def full_match_agg(token_list, table, table_words, conds_value, split, order=0):
    # 如果不是test数据集，则使用wv进行匹配，否则使用table中的内容进行匹配，val待定
    new_list = token_list
    if split == 'train':
        new_list = tokens_full_match(new_list, conds_value, order)
    else:
        new_list = tokens_full_match(new_list, table_words, order)
    return new_list


def left_tokens_match(token_list, words, ix, word, candidate_list=None):
    # 从ix位置开始对后续的tokens列表进行匹配
    # word是当前测试匹配的候选词
    # word[0] == token[0]才会进入该函数，候选词列表是每个长度大于2的token对应的候选词列表
    # 函数返回从ix开始，对word最大的匹配度max_ratio和结束位置end_index
    token = token_list[ix]
    token_list_len = len(token_list)
    new_list = token_list
    max_ratio, end_index = 0, -1
    for end_ix in range(ix, token_list_len):
        # 如果碰到`和`字，或者碰到已经token过的词，则停止匹配
        # if end_ix > ix and (new_list[end_ix] == '和' or new_list[end_ix] in words):
        if end_ix > ix and (new_list[end_ix] == '和' or new_list[end_ix] == '，' or new_list[end_ix] == '还有'):
            break
        # 如果碰到两个字及以上的词，并且其对应的候选词列表长度为1，并且和正在匹配的word不同，则终止匹配
        if end_ix > ix and candidate_list != None and len(candidate_list[end_ix]) == 1:
            candidate_word = candidate_list[end_ix][0]
            if word != candidate_word:
                break
        tmp_str = ''.join(new_list[ix:(end_ix+1)])
        ratio = get_similarity(tmp_str, word)
        # 取最大的相似度
        if ratio > max_ratio:
            max_ratio, end_index = ratio, end_ix
    return max_ratio, end_index


def get_best_match(token_list, words, ix, tmp_list, candidate_list):
    """当从第ix个位置开始进行匹配时，选取words中最优的匹配词，并获取匹配度和结束的位置"""
    # tmp_list: 表示需要从ix位置开始匹配的候选词集
    # candidate_list: 表示每个长度大于2的token对应的候选词集
    total_max_ratio, total_max_word, total_max_end_ix = 0, None, -1
    dup_ratio, dup_end_ix = 0, -1
    for word in tmp_list:
        max_ratio, end_index = left_tokens_match(token_list, words, ix, word, candidate_list)
        # 航空 同时匹配到 南方航空，深圳航空，则忽略这种匹配
        if max_ratio == total_max_ratio and end_index == total_max_end_ix and max_ratio < 0.7:
            dup_ratio, dup_end_ix = max_ratio, end_index
        if max_ratio == total_max_ratio and end_index == total_max_end_ix and word.endswith(token_list[ix]) and max_ratio < 0.8:
            dup_ratio, dup_end_ix = max_ratio, end_index
        # 如 蜂/制品 -- 蜂蜜 蜂产品 都是0.66666，则对较长的进行匹配 
        if max_ratio > total_max_ratio or (max_ratio == total_max_ratio and end_index > total_max_end_ix):
            total_max_ratio, total_max_word, total_max_end_ix = max_ratio, word, end_index
        
    if total_max_ratio == dup_ratio and total_max_end_ix == dup_end_ix:
        return None, 0, -1
    return total_max_word, total_max_ratio, total_max_end_ix


def _contain_chinese(s):
    if not s: return False
    for uchar in s:
        if u'\u4e00' <= uchar <= u'\u9fa5':
            return True
    return False

def _contain_non_digit(s):
    if not s: return False
    for uchar in s:
        if uchar not in '0123456789.':
            return True
    return False


def _qualify(words, token, next_token, order):
    """
    对token进行检查，是否符合进行模糊匹配的条件：年份，包含中文，包含非数字
    """
    if token == '': return False    # 如果已经被处理
    if token[0] in '0123456789':
        # 如果这个全部是数字并且很可能是年份，则我们假设可以进行匹配
        if str.isdigit(token) and 1970 <= int(token) <= 2050: pass
        elif _contain_chinese(token): pass
        elif _contain_non_digit(token): pass
        elif next_token != None and next_token in '月号日': pass   # xx 月 xx 号
        # elif order==0 and token[0] == '0': pass     # 如果是第一轮处理，遇到首个0，则进行匹配
        else: return False    # 不对数字进行处理
    # 先不用这个条件
    # if token in words: return False     # 如果是words中的元素，则不需要再部分匹配
    return True


def search_list_filter(new_list, ix, candidate_set):
    """
    函数: 如果ix位置的token在候选列表，延迟匹配(不立刻进行匹配)，而是往后看，找到一个最优的匹配
    """
    # 如果有多个，则进行最优匹配
    # 如token='英语', next_token='补习' ss = {'英语','英语补习','英语补习班'}, 则匹配以最长匹配为准，匹配为 英语补习
    # 如token='英语', next_token='课' ss = {'英语','英语补习','英语补习班'}, 则匹配为 英语
    # 如果token在ss中，不能跳过去，而是判断token+next_token是否也是candidate_set中某一个word的前缀
    # 这个其实可以变成一个while循环，一直判断下去
    if new_list[ix] in candidate_set and ix < len(new_list)-1:
        tmp_token = new_list[ix] + new_list[ix+1]
        for word in candidate_set:
            if word.find(tmp_token) != -1 and get_similarity(tmp_token, word) > 0.7:
                candidate_set.remove(new_list[ix])
                break

    # 如果第一轮没有去掉，则进行持续往后的匹配
    if new_list[ix] in candidate_set:
        tmp_token = new_list[ix]
        end_ix = ix
        for tx in range(ix+1, len(new_list)):
            tmp_set = set()
            tmp_token = tmp_token + new_list[tx]
            for word in candidate_set:
                if word.find(tmp_token) != -1:
                    tmp_set.add(word)
            if len(tmp_set) == 0:
                end_ix = tx - 1
                break
        if end_ix > ix:
            candidate_set.remove(new_list[ix])

    return candidate_set


def tokens_part_match_1th(token_list, words, other_words, order):
    """
    函数：第一轮处理，以token和word第一个字符相等为触发条件往后进行匹配，选择相似度最大的
    """
    new_list = token_list
    token_list_len = len(new_list)

    candidate_list = get_candidate_list(token_list, words)

    # 以token第一个字符和候选词的第一个字符相等为触发条件，进行后续token的匹配
    for ix, token in enumerate(new_list):
        # 如果不符合条件，则继续下一个
        next_token = new_list[ix+1] if ix < token_list_len-1 else None
        # 年份，中文，非数字
        if not _qualify(words, token, next_token, order): continue

        tmp_set = set()
        if token[0] in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIGKLMNOPQRSTUVWXYZ':
            # 如果token第一个字符是数字，那么则对token进行完全匹配，即word需要包含token
            tmp_set = set([word for word in words if word.startswith(token)])
        else:
            # 以第一个字符相等为触发条件, 只针对汉字
            tmp_set = set([word for word in words if word[0] == token[0]])

        tmp_set = search_list_filter(new_list, ix, tmp_set)
        tmp_list = list(tmp_set)
        max_word, max_ratio, max_end_ix = get_best_match(new_list, words, ix, tmp_list, candidate_list)
        # if token == '河东':
        #     print(token)
        #     print(tmp_set)
        #     print(new_list)
        #     print(max_word, max_ratio, max_end_ix)
        # 阈值设定可能需要调整,范围大概是0.61~0.66
        if max_ratio >= 0.63 and len(new_list[ix]) > 1:
            new_list[ix] = max_word
            # 北京(市) 北京xxx
            if ix > 0 and max_word.startswith(new_list[ix-1][:2]):
                new_list[ix-1] = ''
            for t in range(ix+1, max_end_ix+1):
                new_list[t] = ''  # 将这个设为''，因为已经被匹配了

    # 对于一个token(长度大于1)，如果这个token只在唯一的一个word中出现，则对该word进行替换？？？
    # new_list = first_iteration(new_list, words)
    new_list = remove_null(new_list)
    return new_list


def tokens_part_match_2th(token_list, words, other_words, order):
    """
    函数：第二轮处理，以word包含token为触发条件，往后进行匹配
    参数：
        other_words: 用于train数据集，帮助训练，在train中是table_words,在val和test中为None
    """
    new_list = token_list
    # 获取候选词列表
    candidate_list = get_candidate_list(token_list, words)
    token_list_len = len(token_list)
    for ix, token in enumerate(new_list):
        tmp_token = token   # 由于后续可能需要对token进行处理，比如在token为单字的情况下，将token和后续的token进行连接操作进行判断
        # 如果不符合条件，则继续下一个
        next_token = new_list[ix+1] if ix < token_list_len-1 else None
        if not _qualify(words, tmp_token, next_token, order): continue

        # 如果token的长度为1，则加上后面的token一起匹配，如 红/楼/梦
        if len(token) == 1:
            if ix < len(new_list) - 1:
                tmp_token = tmp_token + new_list[ix+1]
        tmp_set = set()
        # 这里为什么要使用table_words进行处理呢??? 不能直接使用conds words吗，先不用这句话看看效果
        tmp_words = words if other_words == None else other_words
        # 如果遇到单字，则和后面的token串接进行判断
        tmp_set = set([word for word in tmp_words if word.find(tmp_token) != -1])
        # 直觉：如果token在words中只出现一次，则进行替换
        if len(tmp_set) == 1:
            single_word = tmp_set.pop()
            if single_word not in words: continue
            end_ix = ix
            # ["江苏","省","农行","虹桥","支行","和","学府","路","支行"]
            tmp_str = ''
            for t in range(ix+1, len(new_list)):
                tmp_str = ''.join(new_list[ix:t+1])
                if tmp_str not in single_word:
                    end_ix = t-1    # 回退一个
                    tmp_str = ''.join(new_list[ix:end_ix+1])
                    break
            if get_similarity(tmp_str, single_word) < 0.7 and single_word.endswith(tmp_token):
                continue
            # 对于常见字符单独处理
            if tmp_str in ['本书','房地产','出版社','医师'] and single_word.startswith(tmp_str):
                continue
            if tmp_str in ['城市','房地产','出版社','医师'] and single_word.endswith(tmp_str):
                continue
            # 当我们找到一个唯一的匹配时，还要加一个限制条件，即tmp_str和single_word差异太大时忽略改变
            # 徐伟水泥制品厂 徐伟 0.5
            # 可能需要针对train进行单独处理，毕竟train的val很少
            similarity = get_similarity(tmp_str, single_word)
            if (len(words) > 3 and ((single_word.startswith(tmp_str) and similarity > 0.4) or similarity > 0.5)) \
                or (len(words) <= 3 and similarity > 0.4):
                new_list[ix] = single_word
                # 北京(市) 北京xxx
                if ix > 0 and single_word.startswith(new_list[ix-1][:2]):
                    new_list[ix-1] = ''
                for t in range(ix+1, end_ix+1):
                    new_list[t] = ''
        # 如果当前词对应words中的不止一个词，即token出现在多个词中,能进行替换的词不止一个
        if len(tmp_set) > 1:
            # 搜索列表筛选, 将ss中不太可能是结果的给过滤掉
            tmp_set = search_list_filter(new_list, ix, tmp_set)
            # 对剩下的进行最优匹配
            tmp_list = list(tmp_set)
            max_word, max_ratio, max_end_ix = get_best_match(token_list, words, ix, tmp_list, candidate_list)
            # 阈值设定可能需要调整,范围大概是0.61~0.66
            # 如：管理岗位，岗位
            if max_ratio < 0.7 and max_word != None and max_word.endswith(new_list[ix]):
                continue
            if max_ratio >= 0.65:
                new_list[ix] = max_word
                # 北京(市) 北京xxx
                if ix > 0 and max_word.startswith(new_list[ix-1][:2]):
                    new_list[ix-1] = ''
                for t in range(ix+1, max_end_ix+1):
                    new_list[t] = ''  # 将这个设为''，因为已经被匹配了

    new_list = remove_null(new_list)
    return new_list


def get_candidate_list(token_list, words):
    # 对每个长度大于1的token，获取候选词(token在候选词中出现)列表
    new_list = token_list
    candidate_list = [[] for _ in range(len(new_list))]
    for ix, token in enumerate(new_list):
        if len(token) >= 2:
            ss = set()
            for word in words:
                if word.find(token) != -1:
                    ss.add(word)
            candidate_list[ix] = list(ss)   # 候选词列表
    return candidate_list


def tokens_part_match(token_list, words, table=None, other_words=None, order=0):
    """
    函数：对于没有进行完全匹配的token进行部分匹配
    """
    new_list = token_list

    words = remove_null(words)

    # 第一轮匹配，以第一个字符相等为触发条件进行匹配
    new_list = tokens_part_match_2th(new_list, words, other_words, order)
    # 第二轮匹配，以token出现在word中为触发条件进行匹配
    new_list = tokens_part_match_1th(new_list, words, other_words, order)

    return new_list
    

def part_match_agg(token_list, table, table_words, conds_value, split, order):
    # 如果不是test数据集，则使用wv进行匹配，否则使用table中的内容进行匹配，val待定
    new_list = token_list
    if split == 'train':
        new_list = tokens_part_match(new_list, conds_value, table=None, other_words=None, order=order)
    else:
        new_list = tokens_part_match(new_list, table_words, table=table, other_words=None, order=order)
    return new_list


def _process_time(token_list, table_words, conds_value):
    new_list = []
    words = table_words if table_words != None else conds_value
    ix = -1
    while ix < len(token_list) - 1:
        ix += 1
        token = token_list[ix]
        # 对 12/年 这种模式进行补全
        if token == '年' and ix > 0:
            last_token = token_list[ix-1]
            # 16/年；
            if len(last_token) == 2 and str.isdigit(last_token):
                new_list[-1] = '20' + last_token    # 假设是20xx年，待定
                new_list.append('年')
                continue
            # 一九/年
            if len(last_token) == 2 and last_token[0] in tmp_dic and last_token[1] in tmp_dic:
                s_value = tmp_dic[last_token[0]] + tmp_dic[last_token[1]]
                pre_tmp_str = '20' if int(s_value) <= 70 else '19'
                tmp_str = pre_tmp_str + s_value
                new_list[-1] = tmp_str
                new_list.append('年')
                continue
        if token not in words:
            # 一六年；一二年
            tmp_dic = zh_digit_dic
            if len(token) == 3 and token[-1] == '年' and token[0] in tmp_dic and token[1] in tmp_dic:
                s_value = tmp_dic[token[0]] + tmp_dic[token[1]]
                pre_tmp_str = '20' if int(s_value) <= 70 else '19'
                tmp_str = pre_tmp_str + s_value
                new_list.append(tmp_str)
                new_list.append('年')
                continue
            # 八月
            if len(token) > 1 and token[-1] == '月':
                val = get_numberical_value(token[:-1])
                if val != None:
                    new_list.append(val)
                    new_list.append('月')
                    continue
        new_list.append(token)
    return new_list


def _digit_split(val):
    res_list = []
    if val.endswith('00000000'):
        res_list.append(val[:-8])
        res_list.append('0000')
        res_list.append('0000')
    elif val.endswith('0000'):
        res_list.append(val[:-4])
        res_list.append('0000')
    else:
        res_list.append(val)
    return res_list


def _process_digit(token_list, table_words, conds_value):
    new_list = []
    words = table_words if table_words != None else conds_value
    ix = -1
    while ix < len(token_list) - 1:
        ix += 1
        token = token_list[ix]
        if token not in words:
            # if token == '亿':
            #     new_list.append('0000')
            #     new_list.append('0000')
            #     continue
            if token == '万':
                new_list.append('0000')
                continue
            if token == '百万':
                new_list.append('00')
                new_list.append('万')
                continue
            if token == '千万':
                new_list.append('000')
                new_list.append('万')
                continue
            # 处理带有 亿 的数字
            # 对于 1/亿 这种不进行处理
            if len(token) > 1 and token[-1] == '亿':
                val = get_numberical_value(token[:-1])
                if val != None:
                    new_list.append(val)
                    new_list.append('亿')
                    continue
            if token[0] in '0123456789.-零一二三四五六七八九十百千万两负千万百':
                val = get_numberical_value(token_list[ix])
                if val != None:
                    new_list.extend(_digit_split(val))
                    continue
            # 百分之/39.56；百分之/五; 百分之/十二; 百分之/负十
            if len(token) == 3 and token == '百分之' and ix+1 < len(token_list):
                val = get_numberical_value(token_list[ix+1])
                # 如果value是一个有效的数字
                if val != None:
                    new_list.append(val)
                    new_list.append('%')
                    ix += 1
                    continue

        new_list.append(token)
    return new_list


def _process_money(token_list, table_words, conds_value):
    new_list = []
    words = table_words if table_words != None else conds_value
    ix = -1
    while ix < len(token_list) - 1:
        ix += 1
        token = token_list[ix]
        if token not in words:
            if token[-1] == '元' and len(token) > 1:
                val = get_numberical_value(token[:-1])
                if val != None:
                    new_list.append(val)
                    new_list.append('元')
                    continue

        new_list.append(token)
    return new_list


def special_patten_agg(token_list, table_words, conds_value):
    new_list = token_list
    new_list = _process_time(new_list, table_words, conds_value)

    new_list = _process_digit(new_list, table_words, conds_value)

    new_list = _process_money(new_list, table_words, conds_value)

    return new_list


def _read_common_words(fname):
    """Read dict file dic.txt"""
    common_words = set()
    with open(fname, encoding='utf8') as fs:
        for line in fs:
            if line.startswith('#') or line.strip() == '':
                continue
            if line.find(':') == -1:
                words = line.strip().split()
                for word in words:
                    common_words.add(word.strip())
            else:
                words = line.strip().split(':')
                common_words.add(words[0])
                common_words.add(words[1])

    common_words = sorted(list(common_words), key=lambda x : -len(x))
    common_words = [word for word in common_words if len(word) > 1]

    return common_words


words_remove = set(['就是'])
def _remove_common_words(new_list):
    """去除一些常用词，不会影响语意的"""
    for i in range(len(new_list)):
        if new_list[i] in words_remove:
            new_list[i] = ''
    return new_list


# dic中的词
common_words = _read_common_words('./dic.txt')
def common_words_agg(token_list):
    # 'train' 或者 'test' 都可以的
    # 需要进行两遍，不然有的词不能聚合
    new_list = full_match_agg(token_list, None, common_words, common_words, 'train', order=0)
    new_list = full_match_agg(token_list, None, common_words, common_words, 'train', order=0)

    new_list = _remove_common_words(new_list)

    new_list = remove_null(new_list)
    return new_list


def post_process(token_list):
    # 将1990-2500的数字尾数0去掉，比如2016.0 -> 2016
    for ix, token in enumerate(token_list):
        if len(token) == 6 and token.endswith('.0'):
            token_list[ix] = token[:4]
        if token == '月' or token == '号' or token == '日':
            if ix > 0:
                if token_list[ix-1].endswith('.0'):
                    token_list[ix-1] = token_list[ix-1][:-2]
    token_list = remove_null(token_list)
    return token_list


def table_words_filter(table_words):
    tmp_list = list(table_words)
    for ix, elem in enumerate(tmp_list):
        # 去除2012.0等年份表示
        if elem.endswith('.0') and str.isdigit(elem[:-2]):
            if 2000 <= int(elem[:-2]) <= 2050:
                tmp_list[ix] = elem[:-2]
    return set(tmp_list)


def get_row_in_table(cond, rows):
    """
    函数: 对于每个wv，获取位置列表(row number, column number)
    参数:
        cond: [col, op, wv]
    """
    res = []
    for ir, row in enumerate(rows):
        row_val = rows[ir][cond[0]]
        elem = str(row_val)
        try:
            if eval(elem) == eval(cond[2]):
                res.append(ir)
        except:
            if elem == cond[2]:
                res.append(ir)

    return res


def check_wv_in_table(table, conds, split):
    """wv_pos中的每个值表示`table_content中的一个值的index`"""
    if split == 'test':
        return None

    wv_poses = []
    for cond in conds:
        table_content = table['table_content']
        wv_pos = -1
        for ix, v in enumerate(table_content):
            if v == str(cond[2]):
                wv_pos = ix
        wv_poses.append(wv_pos)
    return wv_poses


useless_words = {'来着', '了', '呢', '儿', '会', '吗', '来着', '已','呗'}
def remove_useless_words(token_list):
    """删除一些无用的词，去掉不影响语意的"""
    new_list = []
    for token in token_list:
        if token not in useless_words:
            new_list.append(token)
    return new_list


def get_table_words(table):
    """提取table中的词, 同时进行排序和过滤"""
    table_words = set([str(w) for row in table['rows'] for w in row])
    # 对table中的词进行过滤
    table_words = table_words_filter(table_words)   
    table_words = sorted([w for w in table_words if len(w) <= 50], key=lambda x : -len(x))   # 从长到短排序
    return table_words


def get_header_words(table):
    """提取table中的词, 同时进行排序和过滤"""
    table_headers = set([str(re.split('（|\(', header)[0]) for header in table['header']])
    # TODO: 可以考虑对header进行分词，然后加入到候选词列表
    table_headers = sorted([w for w in table_headers if len(w) <= 50], key=lambda x : -len(x))   # 从长到短排序
    return table_headers


def is_arabic_number(s):
    """阿拉伯数字"""
    for c in s:
        if c not in '0123456789.-':
            return False
    return True


def _insert_headers_nan_train(ann, table):
    """
    将header信息插入到question_tok，两种选择：插入到wv前；插入到句子前；先试试第1种
    """
    # 找到每个wvi对应的header，放到列表
    if ann['wvi_corenlp'] == None:
        return ann
    # 确定在table_content中的index，没有为-1
    wvis = ann['wvi_corenlp']
    ixs = []
    for wvi in wvis:
        tmp_val = ''.join(ann['question_tok'][wvi[0]:(wvi[1]+1)])
        find_it = False
        if is_arabic_number(tmp_val):
            ixs.append(-1)
            continue
        for ix, content in enumerate(table['table_content']):
            if content == tmp_val:
                find_it = True
                ixs.append(ix)
                break
        if not find_it:
            ixs.append(-1)

    # 确定header列表，没有为None
    header_list = []
    for ix in ixs:
        find_it = False
        headers = table['content_header'][ix]
        # 只考虑长度为1的header，怎么简单怎么来
        if len(headers) == 1 and ix != -1:
            find_it = True
            header_list.append(headers[0])
        if not find_it:
            header_list.append(None)

    # 将header插入，然后对wvi进行更新
    insert_num = [0] * len(ann['wvi_corenlp'])
    for i, wvi in enumerate(ann['wvi_corenlp']):
        if header_list[i]:
            ann['question_tok'].insert(wvi[0], header_list[i])
            # Update wvi
            for j, twv in enumerate(ann['wvi_corenlp']):
                if twv[0] >= wvi[0]:
                    twv[0] += 1
                    twv[1] += 1

    return ann


def _insert_headers_nan_test(ann, table):
    """对val和test进行操作，将可能是table_content中的值的header信息插入到content之前"""
    for ix, content in enumerate(table['table_content']):
        wvi = find_str_wvi_full_match(content, ann['question_tok'])
        # 如果找到content并且这个content对应的header只有一个, 并且不是数字
        if wvi:
            tmp_str = ''.join(ann['question_tok'][wvi[0]:(wvi[1]+1)])
            if len(table['content_header'][ix]) == 1 and not is_arabic_number(tmp_str):
                # 这个过程可能会陷入死循环
                # 即：如果插入的值出现在还没有遍历到的content时，可能性很小，先不管 warn
                ann['question_tok'].insert(wvi[0], table['content_header'][ix][0])
    return ann


def _get_date(token_list, ix):
    # 遍历顺序 ix向前，ix向后，找到对应的日期,是4位的或者None
    tmp_token_list = token_list[:ix][::-1] + token_list[ix+1:]
    date = None
    for token in tmp_token_list:
        if str.isdigit(token):
            # 返回年份,ix之前的遍历
            if len(token) == 4 and 1970 <= int(token) <= 2050:
                return token
            # if len(token) == 2 and 0 <= int(token) <= 50:
            #     date_list.add('20' + token)
    # 没找到返回None
    return None


def _get_header(headers, term_type, date):
    """根据日期和术语类型如PE, EPS等寻找对应的全称"""
    results = []
    mp = None
    mp_digit = re.compile('(20)?[0-9]{2}')
    if term_type == 'PE':
        # PE(TTM) PE2017E PE18E P/E18E 12EPE 2011市盈率 2011E市盈率 PE(X)2011A
        mp = re.compile('P/?E.*(20)?[0-9]{2}|PE.*|(20)?[0-9]{2}.?PE|(20)?[0-9]{2}.?市盈率')
    elif term_type == 'EPS':
        # 格式同上
        mp = re.compile('EPS.*(20)?[0-9]{2}|EPS.*|(20)?[0-9]{2}.?EPS|(20)?[0-9]{2}.?每股(收益|盈余)')
    elif term_type == 'PS':
        mp = re.compile('市销率|^PS$')
    elif term_type == 'PB':
        mp = re.compile('P/?B.*(20)?[0-9]{2}|PE.*|(20)?[0-9]{2}.?PB|(理论)?P/?B|市净率')
    elif term_type == 'ROE':
        mp = re.compile('ROE.*(20)?[0-9]{2}|ROE.*|ROE|净资产收益率')
    elif term_type == 'RNAV':
        mp = re.compile('RNAV|重估净资产')
    # 第一遍确定个数，如果只有一个，直接返回就行, 以PE为例
    match_results = []
    for header in headers:
        # 如果匹配成功, 匹配不成功返回None
        if mp.match(header):
            # match_str = curr_result.group()     # 匹配的字符串
            match_results.append(header)
    if len(match_results) == 0:
        return []
    if len(match_results) == 1:
        return match_results
    # 如果匹配多个，但是日期为空(日期是选择的关键)
    # print(date)
    # print(match_results)
    if not date:
        return []

    # 如果不止一个匹配结果
    for header in match_results:
        ms = mp.match(header).group()   # 匹配的字符串
        digit = mp_digit.search(ms)     # 获取字符串中的数字
        if digit:
            digit = digit.group()
            if len(digit) == 2:
                digit = '20' + digit    # 将数字转化为日期
            if digit == date:
                return [header]
    return match_results


def _check_token_in_terms(token_list, ix, headers, term_type, term_list):
    """
    根据关键词如 本益比/PE/市盈率等 在header中寻找对应的值
    Parameters:
        term_type: PE, EPS, ROE, PS等
        term_list: ['本益比', '市盈率', 'PE']等每一个term_type对应的常用词语
    """
    token = token_list[ix]
    if token in term_list:
        # 从question_tok中寻找日期，先找ix之前的，再找ix之后的
        date = _get_date(token_list, ix) # None or 2018/2019...
        # 根据日期和术语类型如PE, EPS等寻找对应的全称(正确表达)
        header_list = _get_header(headers, term_type, date)
        # 如果只有一个符合条件，则进行替换
        if len(header_list) == 1:
            return header_list[0]
    return token


def _process_share_terms(ann, headers, split):
    """对证券领域股票专有名词的处理"""
    # PS: 市净率
    # ROE: 净资产收益率
    # 解决PE
    token_list = ann['question_tok']
    # print(token_list)
    # 必须确保每个关键词都是一个token
    for ix, token in enumerate(token_list):
        # 如果找到则进行替换，否则不进行替换
        token_list[ix] = _check_token_in_terms(token_list, ix, headers, 'PE', ['本益比', '市盈率', 'PE'])
        token_list[ix] = _check_token_in_terms(token_list, ix, headers, 'EPS', ['每股盈余', '每股收益', 'EPS'])
        token_list[ix] = _check_token_in_terms(token_list, ix, headers, 'PB', ['市净率', 'PB'])
        token_list[ix] = _check_token_in_terms(token_list, ix, headers, 'PS', ['市销率', 'PS'])
        token_list[ix] = _check_token_in_terms(token_list, ix, headers, 'ROE', ['净资产收益率', 'ROE'])
        token_list[ix] = _check_token_in_terms(token_list, ix, headers, 'RNAV', ['重估净资产', 'RNAV'])

    return ann


def _is_share_field(headers):
    # 如果headers中出现这些则表示该table可能是属于证券领域
    for header in headers:
        if header.find('PE') != -1 or header.find('EPS') != -1 or header.find('PB') != -1 or \
            header.find('本益比') != -1 or header.find('市盈率') != -1 or header.find('收益') != -1:
            return True
    return False


def insert_headers_nan(ann, table, split):
    """
    将header信息插入到question_tok，两种选择：插入到wv前；插入到句子前；先试试第1种
    """
    # val和test不使用wvi进行操作, 即按照test的方式来, train使用wvi来进行操作
    return _insert_headers_nan_train(ann, table) if split == 'train' else _insert_headers_nan_test(ann, table)


def insert_headers_digit(ann, table, split):
    """
    对关键的token进行替换，替换成表头
    """
    # 根据header判断是否属于证券领域，对证券领域股票专有名词的处理
    if _is_share_field(table['header']):
        ann = _process_share_terms(ann, table['header'], split)

    return ann


def annotate_example_nlpir(example, table, split):
    """
    Jan. 2019: Wonseok
    Annotate only the information that will be used in our model.
    """
    ann = {'table_id': example['table_id']}

    example['question'] = example['question'].strip()   # 去除首尾空格
    example['question'] = replace_words(example['question'])    # 替换
    example['question'] = remove_special_char(example['question'])  # 去除question中的特殊字符
    example['question'] = change_special_pattern(example['question'])   # 改变一些固有模式，如 不是...吗 在...以下

    # 去除wv中的特殊字符，可能val也需要类似的处理
    conds_value = set()
    wv_ann1 = []
    if split != 'test':   # val按照tables进行聚合，但是按照和train同样的方法求wvi，所以val数据集会有wvi属性
        ann['sql'] = example['sql']
        ann['query'] = sql = copy.deepcopy(example['sql'])
        for ix, conds11 in enumerate(ann['sql']['conds']):
            tmp_val = ann['sql']['conds'][ix][2]
            tmp_val = remove_special_char(tmp_val)
            ann['sql']['conds'][ix][2] = tmp_val
            wv_ann1.append(tmp_val)
            conds_value.add(tmp_val)
        conds_value = sorted(list(conds_value), key=lambda x : -len(x))   # 按照字符串长度，从长到短排序

    # 分别使用中科大的和北大的分词系统进行分词
    _nlu_ann_pr = pr.segment(example['question'],  pos_tagging=False)
    _nlu_ann_pk = seg.cut(example['question'])
    _nlu_ann_jb = None
    # big_tokens = list(jieba.cut(example['question']))

    # 综合 北大 分词和 pynlpir 分词的结果，二者取短, 分出来的词会比较细粒度
    _nlu_ann = seg_summary(example['question'], _nlu_ann_pr, _nlu_ann_pk, _nlu_ann_jb)
    # _nlu_ann = _nlu_ann_pr
    ann['question'] = example['question']

    # 不加预处理
    # 对原始数据进行操作，不改变原question内容，''.join()后的内容不会发生变化, 进一步细粒度划分
    processed_nlu_token_list = pre_no_change_process(_nlu_ann)

    # 去除token_list中的停用词, >>> 来着 了 呢 儿 会 吗 来着 已
    processed_nlu_token_list = remove_useless_words(processed_nlu_token_list)

    # 获取table中的words
    table_words = get_table_words(table)
    # 获取table header中的words
    header_words = get_header_words(table)

    # 如果可以进行完全匹配(子列表是wv或者table中的一个元素，则聚合成一个整体，后续不再对该token进行处理，包括其中的数字) 
    # 完全匹配可以对数字处理
    processed_nlu_token_list = full_match_agg(processed_nlu_token_list, table, table_words, conds_value, split, order=0)
    # 对header进行全匹配
    processed_nlu_token_list = full_match_agg(processed_nlu_token_list, None, header_words, header_words, split, order=0)
    # 如果不能进行完全匹配，则对 **没有进行完全匹配的token** 进行模糊匹配，只对中文进行处理\
    processed_nlu_token_list = part_match_agg(processed_nlu_token_list, table, table_words, conds_value, split, order=0)

    # 将常用词进行聚合，如一下 哪个
    processed_nlu_token_list = common_words_agg(processed_nlu_token_list)
    ##TODO：是不是要进行特殊的change处理，再来一轮full match和part match？？
    # 主要处理 时间(年份，月份)，数字(亿，万)
    processed_nlu_token_list = special_patten_agg(processed_nlu_token_list, table_words, conds_value)

    processed_nlu_token_list = full_match_agg(processed_nlu_token_list, table, table_words, conds_value, split, order=1)
    processed_nlu_token_list = full_match_agg(processed_nlu_token_list, None, header_words, header_words, split, order=1)
    processed_nlu_token_list = part_match_agg(processed_nlu_token_list, table, table_words, conds_value, split, order=1)

    # 后处理,如年份转化，2016.0 -> 2016
    processed_nlu_token_list = post_process(processed_nlu_token_list)

    # 正常情况下的question_tok
    ann['question_tok'] = processed_nlu_token_list

    # train数据集应该按照哪种模式来插入？和val相同？
    ann = insert_headers_digit(ann, table, split)

    # 对非数字wv插入header
    if split != 'train':
        # 如果找到table中的value，则将该value对应的header插入到value之前, 根据wv插入header
        # 对非数字进行操作
        pass
        # ann = insert_headers_nan(ann, table, split='test')

    # 测试集中没有sql属性，在这个地方进行判断
    if 'sql' not in example:
        return ann, table_words
        
    # Check whether wv_ann exsits inside question_tok
    try:
        # state 变量方便调试
        wvi1_corenlp, state = check_wv_in_nlu_tok(wv_ann1, ann['question_tok'])
        # 不匹配的wvi不冲突
        # 这个变量表示，如果需要插入wv
        insert_wv = True
        if insert_wv and split == 'train':
            unmatch_set = get_unmatch_set(processed_nlu_token_list, wv_ann1, wvi1_corenlp)
            if unmatch_set:
                # 如果存在不匹配的列表
                # print('unmatch_set find')
                question_tok_new = replace_unmatch_set(processed_nlu_token_list, wv_ann1, wvi1_corenlp, unmatch_set)

                # 重新分词进行token, 确定wvi的值
                question_new = ''.join(question_tok_new)

                # 分别使用中科大的和北大的分词系统进行分词
                _nlu_ann_pr_new = pr.segment(question_new,  pos_tagging=False)
                _nlu_ann_pk_new = seg.cut(question_new)
                _nlu_ann_new = seg_summary(question_new, _nlu_ann_pr_new, _nlu_ann_pk_new, None)

                _nlu_ann_new = pre_no_change_process(_nlu_ann_new)
                _nlu_ann_new = pre_with_change_process(_nlu_ann_new)
                _nlu_ann_new = post_with_change_process(_nlu_ann_new)

                # state 变量方便调试
                wvi1_corenlp, state = check_wv_in_nlu_tok(wv_ann1, _nlu_ann_new)

                ann['question_tok'] = _nlu_ann_new

        ann['wvi_corenlp'] = wvi1_corenlp
        ann['stage'] = state
    except:
        ann['wvi_corenlp'] = None
        ann['tok_error'] = 'SQuAD style st, ed are not found under CoreNLP.'

    # 增加一个属性，对每个wv，将其在table中的位置(行1, 行2)放到一个列表
    wv_pos = check_wv_in_table(table, ann['sql']['conds'], split)
    ann['wv_pos'] = wv_pos

    # 对非数字wv插入header
    if split == 'train':
        pass
        # ann = insert_headers_nan(ann, table, split)

    return ann, table_words


def count_lines(fname):
    with open(fname) as f:
        return sum(1 for line in f)


def check_in_words(words, token):
    """Check if token is contained in one of the words"""
    for word in words:
        if word.find(token) != -1:
            return True
    return False


def _part_ignore(token_list, wvis, table, prob):
    """Ignore part of the tokens in the question_tok"""
    flags = [False] * len(token_list)   # False: can be ignored
    exclude_words = table['header']     # Any token contained in these words shouldn't be ignored
    for wvi in wvis:
        for i in range(wvi[0], wvi[1]+1):
            flags[i] = True     # True: can not be ignored

    # Ignore part of the words.
    for i in range(len(token_list)):
        if not flags[i] and random.random() < prob and not check_in_words(exclude_words, token_list[i]):
            token_list[i] = ''

    # Update wvi
    for i in range(len(wvis)):
        wvi = wvis[i]
        null_num = sum([1 for token in token_list[:wvi[0]] if token == ''])
        wvi[0], wvi[1] = wvi[0] - null_num, wvi[1] - null_num

    # Remove '' in token_list
    new_token_list = []
    for token in token_list:
        if token != '':
            new_token_list.append(token)

    return new_token_list, wvis



def ignore_words(ann, table_words, prob=0.15, repeat=2):
    """
    Ignore 15% of the words which are not in the table
    """
    results = [ann]
    if not ann['wvi_corenlp']:  # if wvi is none, then do nothing
        return results
    for i in range(repeat):
        new_ann = copy.deepcopy(ann)
        new_ann['question_tok'], new_ann['wvi_corenlp'] = _part_ignore(new_ann['question_tok'], new_ann['wvi_corenlp'], table, prob)
        results.append(new_ann)
    return results


def _synonyms_replace(token_list, table_words, synonyms_dic):
    for i, token in enumerate(token_list):
        if token in synonyms_dic and token not in table_words:
            token_list[i] = random.choice(synonyms_dic[token])  # random select one
    return token_list


def synonyms_replace(ann, table_words, synonyms_dic, repeat=3):
    """
    Replace the synonyms in the dataset

    Parameters:
        synonyms_dic: w1:[w1, w2, ..., wn]
    """
    results = []
    for i in range(repeat):
        new_ann = copy.deepcopy(ann)
        new_ann['question_tok'] = _synonyms_replace(new_ann['question_tok'], table_words, synonyms_dic)
        results.append(new_ann)
    return results


def permute_agg_sel(ann):
    agg_and_sel_list = []
    agg_list = [list(t) for t in itertools.permutations(ann['sql']['agg'])]
    sel_list = [list(t) for t in itertools.permutations(ann['sql']['sel'])]
    for agg, sel in zip(agg_list, sel_list):
        new_dic = {'agg':agg, 'sel':sel}
        agg_and_sel_list.append(new_dic)
    return agg_and_sel_list


def permute_conds_wvi(ann):
    conds_and_wvi_list = []
    conds_list = [list(t) for t in itertools.permutations(ann['sql']['conds'])]
    wvi_list = [list(t) for t in itertools.permutations(ann['wvi_corenlp'])]
    for conds, wvi in zip(conds_list, wvi_list):
        new_dic = {'conds':conds, 'wvi_corenlp':wvi}
        conds_and_wvi_list.append(new_dic)
    return conds_and_wvi_list


def data_broaden(ann):
    # ann是一个json语句，该函数对ann进行数据增广，即交换一些数据的位置从而对数据进行扩展
    if ann['wvi_corenlp'] == None:
        return [ann]
    agg_and_sel_list = permute_agg_sel(ann)     # [{agg:[0,1], sel:[1,2]},{agg:[1,0], sel:[2,1]}]
    conds_and_wvi_list = permute_conds_wvi(ann)   # [{conds:[[4,1,"2"],[8,1,"10"]], wvi_corenlp:[[23,23],[13,13]]},{conds:[[8,1,"10"],[4,1,"2"]], wvi_corenlp:[[13,13]],[23,23]}]
    new_ann_list = []
    for dic1 in agg_and_sel_list:
        for dic2 in conds_and_wvi_list:
            new_ann = copy.deepcopy(ann)
            new_ann['sql']['agg'] = dic1['agg']
            new_ann['sql']['sel'] = dic1['sel']
            new_ann['sql']['conds'] = dic2['conds']
            new_ann['query']['agg'] = dic1['agg']
            new_ann['query']['sel'] = dic1['sel']
            new_ann['query']['conds'] = dic2['conds']
            new_ann['wvi_corenlp'] = dic2['wvi_corenlp']
            new_ann_list.append(new_ann)
    return new_ann_list


def get_mvl(example):
    max_len = 0
    if example['wvi_corenlp'] == None:
        return -1
    for wvi in example['wvi_corenlp']:
        max_len = max(max_len, wvi[1] - wvi[0] + 1)
    return max_len


appid = '20190727000321747' #你的appid
secretKey = 'Xx4oRkPel9qANIpqh1kT' #你的密钥
httpClient = None   # Initialize to None
def translate(q, httpClient, fromLang, toLang):
    """
    translate question q from one language to another language
    Parameters:
        q: question
        httpClient: client
        fromLang: source language
        toLang: dest language
    Return:
        return the translated result of q in `toLang` language
    """
    if not q: 
        return q
    salt = random.randint(32768, 65536)
    sign = appid+q+str(salt)+secretKey
    sign = hashlib.md5(sign.encode(encoding='UTF-8')).hexdigest()

    myurl = '/api/trans/vip/translate'
    myurl = myurl+'?appid='+appid+'&q='+urllib.parse.quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign

    res = {}
    try:
        httpClient.request('GET', myurl)
     
        #response是HTTPResponse对象
        response = httpClient.getresponse().read()

        res = json.loads(response.decode('utf-8'))
    except Exception as e:
        print(e)

    if 'trans_result' in res:
        return res['trans_result'][0]['dst']
    return None


def trans(path='./wikisql/data/tianchi/'):
    """
    Translate each record in test dataset.
    """
    httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
    fail_count = 0
    cnt = 0
    with open(os.path.join(path, 'test.json'), encoding='utf8') as fs, open(os.path.join(path, 'test_trans.json'), 'wt', encoding='utf8') as fo:
        for line in fs:
            example = json.loads(line)
            example['trans'] = example['question'].strip()   # 去除首尾空格
            example['trans'] = replace_words(example['trans'])    # 替换
            example['trans'] = remove_special_char(example['trans'])  # 去除question中的特殊字符
            example['trans'] = translate(translate(example['trans'], httpClient, 'zh', 'jp'), httpClient, 'jp', 'zh')
            if not example['trans']:
                fail_count += 1
            if cnt > 30:
                break
            cnt += 1
            fo.write(json.dumps(example, ensure_ascii=False) + '\n')
        print('%d of %d fail.' % (fail_count, cnt))

    httpClient.close()


def _read_dic(fname):
    """Read dict file dic.txt"""
    synonyms_dic = {}
    with open(fname, encoding='utf8') as fs:
        for line in fs:
            if line.startswith('#') or line.strip() == '':
                continue
            if line.find(':') == -1:
                words = line.strip().split()
                for word in words:
                    synonyms_dic[word] = words
            else:
                words = line.strip().split(':')
                synonyms_dic[words[0]] = words[1]

    return synonyms_dic


def _is_reasonable(s, ignore_digit=True):
    """去掉table_content中不符合条件的词，包括数字和单字"""
    if len(s) == 0:
        return False
    if len(s) > 50:
        return False
    if len(s) == 1 and s not in '0123456789':
        return False
    if ignore_digit:
        for c in s:
            if c not in '0123456789-.':
                return True
        return False
    else:
        return True


def _generate_wv_pos_each(table):
    """
    为table生成三个新的属性，依次为table_content, content_header, content_index
    依次取第一列，第二列，...， 不重复生成一个单列表
    Parameters:
        content_header: 表示每个content对应的header
        contnet_index: 表示每个content对应在table_content中的列数
    """
    table_content = OrderedSet()
    content_header = []
    content_index = []
    row_num, col_num = len(table['rows']), len(table['rows'][0])
    headers = [str(h) for h in table['header']]

    # 先判断最后得到的表是否为空, 如果为空则把数字全部加入
    ignore_digit = True
    table_content_null = True
    for c in range(col_num):
        for r in range(row_num):
            sv = remove_special_char(str(table['rows'][r][c]))
            if _is_reasonable(sv, ignore_digit=True):
                table_content_null = True
                break
    if table_content_null:
        ignore_digit = False


    # 生成table的table_content属性，从上到下，从左到右
    for c in range(col_num):
        for r in range(row_num):
            sv = remove_special_char(str(table['rows'][r][c]))
            if _is_reasonable(sv, ignore_digit):
                # 去除特殊字符
                table_content.add(sv)

    # 字典content_ix_dic：每个value对应的index
    content_ix_dic = {}
    table_content = list(table_content)
    for ix, content in enumerate(table_content):
        content_ix_dic[content] = ix
        content_header.append(OrderedSet())
        content_index.append(OrderedSet())

    for c in range(col_num):
        for r in range(row_num):
            sv = remove_special_char(str(table['rows'][r][c]))
            if _is_reasonable(sv, ignore_digit):
                ix = content_ix_dic[sv]     # 获取这个content在`table_content`中的位置
                content_header[ix].add(headers[c])
                content_index[ix].add(c)
    for ix in range(len(table_content)):
        content_header[ix] = list(content_header[ix])
        content_index[ix] = list(content_index[ix])

    # 生成三个属性
    table['table_content'] = table_content
    table['content_header'] = content_header
    table['content_index'] = content_index

    return table


# 定义接口函数
synonyms_dic = _read_dic('./dic.txt')
def token_train_val(base_path='./wikisql/data/tianchi/'):
    """
    生成train和val的token文件
    Parameters:
        base_path: 基本路径
    """
    # the tokened file of test is used to debug.
    for split in ['val', 'test']:
        fsplit = os.path.join(base_path, split, split) + '.json'
        ftable = os.path.join(base_path, split, split) + '.tables.json'
        ftable_new = os.path.join(base_path, split, split) + '_new.tables.json'
        fout = os.path.join(base_path, split, split) + '_tok.json'

        print('tokening {}'.format(fsplit))
        with open(fsplit, encoding='utf8') as fs, open(ftable, encoding='utf8') as ft, open(fout, 'wt', encoding='utf8') as fo, open(ftable_new, 'wt', encoding='utf8') as fto:
            print('loading tables')

            # ws: Construct table dict with table_id as a key.
            tables = {}
            for line in tqdm(ft, total=count_lines(ftable)):
                d = json.loads(line)
                d = _generate_wv_pos_each(d)    # 对table生成三个新的属性
                tables[d['id']] = d
                fto.write(json.dumps(d, ensure_ascii=False) + '\n')
            print('loading examples')
            n_written = 0
            mvl_bigger_2 = mvl_none = 0
            cnt = 0
            for line in tqdm(fs, total=count_lines(fsplit)):
                d = json.loads(line)
                # a = annotate_example(d, tables[d['table_id']])
                a, table_words = annotate_example_nlpir(d, tables[d['table_id']], split)

                # repeat 表示进行随机忽略的个数
                a_list = [a]
                if split == 'train':
                    # data boarden by replacing with synonyms.
                    syn_results = synonyms_replace(a, table_words, synonyms_dic, repeat=0)
                    a_list.extend(syn_results)
                    if False:
                        # data boarden by ignoring part of the words.
                        a_list = ignore_words(a, table_words, prob=0.15, repeat=0)
                # 如果数据为train，则对json语句进行数据增广，即数据进行扩展
                # if split == 'train':
                #     a_list = data_broaden(a)
                cnt += 1
                for t in a_list:
                    # 使用ensure_ascii=False避免写到文件的中文数据是ASCII编码表示
                    mvl = 1
                    if split != 'test':
                        mvl = get_mvl(t)
                    if mvl > 2 and (split == 'train' or split == 'val'):
                        mvl_bigger_2 += 1
                        continue
                    if mvl == -1 and (split == 'train' or split == 'val'):
                        mvl_none += 1
                        continue
                    fo.write(json.dumps(t, ensure_ascii=False) + '\n')
                    n_written += 1
            print('wrote {} examples'.format(n_written))
            print('drop %d(%.3f) examples where mvl > 2' % (mvl_bigger_2, mvl_bigger_2/cnt))
            print('drop %d(%.3f) examples where mvl is None' % (mvl_none, mvl_none/cnt))


def remove_crap(ann):
    """去除废话产生新的增广数据"""
    return ann


def record_broaden(ann, table_words, synonyms_dic, repeat=0):
    results = [ann]
    # 近义词替换来产生
    syn_results = synonyms_replace(ann, table_words, synonyms_dic, repeat-1)
    results.extend(syn_results)

    # 去除废话来产生
    # crap_result = remove_crap(ann)
    # results.append(crap_result)
    return results


# 定义接口函数,在线时使用，不会生成中间文件
def token_each(record, table, split):
    """
    Token each record in the `split` dataset.

    Parameters:
        record: a record in the `split` dataset
        tables: the tables for the test dataset

    Return:
        the tokened record for the input record
    """
    table = _generate_wv_pos_each(table)    # 为table生成三个新的属性
    ann, table_words = annotate_example_nlpir(record, table, split)

    results = record_broaden(ann, table_words, synonyms_dic, repeat=0)
    
    if split != 'test':
        mvl = get_mvl(ann)
        # 如果token error或者mvl大于2表示不符合条件，返回None值，只针对train和val
        if mvl > 2 or mvl == -1:
            return []
    return results
