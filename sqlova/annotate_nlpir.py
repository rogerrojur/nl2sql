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

import difflib

import pynlpir as pr
pr.open()

import jieba
# 北大的分词系统
import pkuseg
seg = pkuseg.pkuseg()

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


words_dic = {'诶，':'','诶':'','那个':'','那个，':'', '呀':'','啊':'','呃':'', '鹅厂':'腾讯', 
            '马桶台':'湖南芒果TV', '荔枝台':'江苏卫视', '北上广':'北京和上海和广州','北上':'北京和上海',
            '厦大':'厦门大学', '中大':'中山大学', '广大':'广州大学', '东航':'东方航空', '国图':'国家图书馆',
            '内师大':'内蒙古师范大学','武大':'武汉大学','中科大':'中国科学技术大学','欢乐喜和剧人':'欢乐喜剧人',
            '本科或者本科以上':'本科及本科以上','并且':'而且','为负':'小于0','为正':'大于0','陆地交通运输':'陆运','亚太地区':'亚太',
            '负数':'小于0','两万一':'21000','辣椒台':'江西卫视'}


def is_valid_char(uchar):
    # 中文，英文，数字，字母
    # 去除特殊字符，还有空格，换行符，制表符
    if u'\u4e00' <= uchar <= u'\u9fa5' or u'\u0030' <= uchar <= uchar<=u'\u0039' \
        or u'\u0041' <= uchar <= u'\u005a' or u'\u0061' <= uchar <= u'\u007a':
        return True


def replace_words(s):
    for key in words_dic:
        if s.find(key) != -1:
            s = s.replace(key, words_dic[key])
    return s


def remove_special_char(s):
    new_s = ''
    for c in s:
        if c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' or \
            c in '！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〜〝〞–—‘\'‛“”„‟…‧﹏.' or \
            is_valid_char(c):
            new_s += c
    return new_s


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


def tokens_full_match(token_list, words):
    # words：候选词列表
    # token_list: 被替换的词列表
    new_list = token_list
    for word in words:
        if len(word) == 0:
            continue
        wvi = find_str_wvi_full_match(word, new_list)
        if wvi == None:
            continue
        new_list = new_list[:wvi[0]] + [''.join(new_list[ wvi[0] : wvi[1]+1 ])] + new_list[wvi[1]+1:]
    return new_list


def full_match_agg(token_list, table_words, conds_value, split):
    # 如果不是test数据集，则使用wv进行匹配，否则使用table中的内容进行匹配，val待定
    new_list = token_list
    if split == 'train':
        new_list = tokens_full_match(new_list, conds_value)
    else:
        new_list = tokens_full_match(new_list, table_words)
    return new_list


def tokens_part_match(token_list, words):
    # 对于没有进行完全匹配的token进行部分匹配
    new_list = token_list
    

    # 对于一个token(长度大于1)，如果这个token只在唯一的一个word中出现，则对该word进行替换？？？
    # new_list = first_iteration(new_list, words)
    return new_list


def part_match_agg(token_list, table_words, conds_value, split):
    # 如果不是test数据集，则使用wv进行匹配，否则使用table中的内容进行匹配，val待定
    new_list = token_list
    if split != 'test':
        new_list = tokens_part_match(new_list, conds_value)
    else:
        new_list = tokens_part_match(new_list, table_words)
    return new_list


def annotate_example_nlpir(example, table, split):
    """
    Jan. 2019: Wonseok
    Annotate only the information that will be used in our model.
    """
    ann = {'table_id': example['table_id']}

    example['question'] = example['question'].strip()   # 去除首尾空格
    example['question'] = replace_words(example['question'])    # 替换
    example['question'] = remove_special_char(example['question'])  # 去除question中的特殊字符

    # 去除wv中的特殊字符，可能val也需要类似的处理
    conds_value = []
    wv_ann1 = []
    if split != 'test':   # val按照tables进行聚合，但是按照和train同样的方法求wvi，所以val数据集会有wvi属性
        ann['sql'] = example['sql']
        ann['query'] = sql = copy.deepcopy(example['sql'])
        for ix, conds11 in enumerate(ann['sql']['conds']):
            tmp_val = ann['sql']['conds'][ix][2]
            tmp_val = remove_special_char(tmp_val)
            ann['sql']['conds'][ix][2] = tmp_val
            wv_ann1.append(tmp_val)
            conds_value.append(tmp_val)
        conds_value = sorted(conds_value, key=lambda x : -len(x))   # 按照字符串长度，从长到短排序

    # 分别使用中科大的和北大的分词系统进行分词
    _nlu_ann_pr = pr.segment(example['question'],  pos_tagging=False)
    _nlu_ann_pk = seg.cut(example['question'])
    # _nlu_ann_jb = list(jieba.cut_for_search(example['question']))
    _nlu_ann_jb = None

    # 综合 北大 分词和 pynlpir 分词的结果，二者取短, 分出来的词会比较细粒度
    _nlu_ann = seg_summary(example['question'], _nlu_ann_pr, _nlu_ann_pk, _nlu_ann_jb)
    # _nlu_ann = _nlu_ann_pr
    ann['question'] = example['question']

    # 不加预处理
    # 对原始数据进行操作，不改变原question内容，''.join()后的内容不会发生变化, 进一步细粒度划分
    processed_nlu_token_list = pre_no_change_process(_nlu_ann)

    # 获取table中的words
    table_words = set([str(w) for row in table['rows'] for w in row])
    table_words = sorted([w for w in table_words if len(w) < 30], key=lambda x : -len(x))   # 从长到短排序
    # 如果可以进行完全匹配(子列表是wv或者table中的一个元素，则聚合成一个整体，后续不再对该token进行处理，包括其中的数字) 
    # 完全匹配可以对数字处理
    processed_nlu_token_list = full_match_agg(processed_nlu_token_list, table_words, conds_value, split)
    # 如果不能进行完全匹配，则对 **没有进行完全匹配的token** 进行模糊匹配，只对中文进行处理
    processed_nlu_token_list = part_match_agg(processed_nlu_token_list, table_words, conds_value, split)

    ##TODO：是不是要进行特殊的change处理，再来一轮full match和part match？？

    # 改变question进行token后的内容，以提升完全匹配的准确率
    # processed_nlu_token_list = pre_with_change_process(processed_nlu_token_list)

    # 对上一步进行操作的列表进行再操作，因为可能会出现冲突问题，如钱的问题，不是按照先后顺序处理的，如十二/块/五
    # processed_nlu_token_list = post_with_change_process(processed_nlu_token_list)

    # 正常情况下的question_tok
    ann['question_tok'] = processed_nlu_token_list
    # ann['table'] = {
    #     'header': [annotate(h) for h in table['header']],
    # }
    # 测试集中没有sql属性，在这个地方进行判断
    if 'sql' not in example:
        return ann
        
    # Check whether wv_ann exsits inside question_tok

    try:
        # state 变量方便调试
        wvi1_corenlp, state = check_wv_in_nlu_tok(wv_ann1, ann['question_tok'])
        # 不匹配的wvi不冲突
        # 这个变量表示，如果需要插入wv
        insert_wv = False
        if insert_wv:
            unmatch_set = get_unmatch_set(processed_nlu_token_list, wv_ann1, wvi1_corenlp)
            if unmatch_set and split != 'test':
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

    return ann


def count_lines(fname):
    with open(fname) as f:
        return sum(1 for line in f)


def detokenize(tokens):
    ret = ''
    for g, a in zip(tokens['gloss'], tokens['after']):
        ret += g + a
    return ret.strip()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--din', default='./wikisql/data/tianchi/', help='data directory')
    parser.add_argument('--dout', default='./wikisql/data/tianchi/', help='output directory')
    parser.add_argument('--split', default='train,val,test', help='comma=separated list of splits to process')
    args = parser.parse_args()

    answer_toy = not True
    toy_size = 10

    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    # 加载缩写词对应的词典,对token进行替换
    # annotate_dic = {}
    # with open('annotate_dic.txt', encoding='utf8') as fin:
    #     for line in fin:
    #         # 字典扩容，合并
    #         annotate_dic.update(json.loads(line))

    # 替换列表
    # replace_dic = {}
    # with open('replace_dic.txt', encoding='utf8') as fin:
    #     for line in fin:
    #         # 字典扩容，合并
    #         replace_dic.update(json.loads(line))


    # for split in ['train', 'val', 'test']:
    for split in args.split.split(','):
        fsplit = os.path.join(args.din, split) + '.json'
        ftable = os.path.join(args.din, split) + '.tables.json'
        fout = os.path.join(args.dout, split) + '_tok.json'

        print('annotating {}'.format(fsplit))
        with open(fsplit, encoding='utf8') as fs, open(ftable, encoding='utf8') as ft, open(fout, 'wt', encoding='utf8') as fo:
            print('loading tables')

            # ws: Construct table dict with table_id as a key.
            tables = {}
            for line in tqdm(ft, total=count_lines(ftable)):
                d = json.loads(line)
                tables[d['id']] = d
            print('loading examples')
            n_written = 0
            cnt = -1
            for line in tqdm(fs, total=count_lines(fsplit)):
                cnt += 1
                d = json.loads(line)
                # a = annotate_example(d, tables[d['table_id']])
                a = annotate_example_nlpir(d, tables[d['table_id']], split)
                # print(a)
                # if cnt > 10:
                #     break
                # 使用ensure_ascii=False避免写到文件的中文数据是ASCII编码表示
                fo.write(json.dumps(a, ensure_ascii=False) + '\n')
                n_written += 1

                if answer_toy:
                    if cnt > toy_size:
                        break
            print('wrote {} examples'.format(n_written))
