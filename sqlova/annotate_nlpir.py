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
    nlu_t1_low = [tok.lower() for tok in nlu_t1]
    for i_wn, wv_tok11 in enumerate(wv_tok1):
        wv_tok11_low = [tok.lower() for tok in wv_tok11]
        results = find_sub_list(wv_tok11_low, nlu_t1_low)
        st_idx, ed_idx = results[0]

        g_wvi1_corenlp.append( [st_idx, ed_idx] )

    return g_wvi1_corenlp


def find_str_in_list(s, l):
    # from stack overflow.
    # s 包括 l 中的一个连续的子序列
    results = []
    stage = '0'
    s_len, l_len = len(s), len(l)
    # for ix, token in enumerate(l):
    #     if token.startswith(s):
    #         results.append((ix, ix))

    # if len(results) != 0:
    #     return results

    # 第一遍：假设存在完全匹配
    if not results:
        stage = '1'
        # print('1-th iter')
        for ix, token in enumerate(l):
            if s[0] == token[0]:
                tmp_str, tmp_idx = '', ix
                while len(tmp_str) < s_len and tmp_idx < l_len:
                    tmp_str += l[tmp_idx]
                    tmp_idx += 1
                if tmp_str == s:
                    results.append((ix, tmp_idx-1))

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
    nlu_t1_low = [tok.lower() for tok in nlu_t1]
    for i_wn, wv_str in enumerate(wv_str_list):
        wv_low = wv_str.lower()
        # stage: 找到子串的阶段，方便调试，字符串表示
        results, stage = find_str_in_list(wv_low, nlu_t1_low)
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
                results[-1] = '20' + results[-1]    # 16变成2016
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
                    results[:-1] = get_numberical_value(token_list[ix-1])
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
            results.append('个')
            continue

        results.append(token)

    # 去除空的
    copy = []
    for r in results:
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

    # 去除空的
    copy = []
    for r in results:
        r = r.strip()
        if r != '' and r != ' ':
            copy.append(r)

    return copy




def annotate_example_nlpir(example, table):
    """
    Jan. 2019: Wonseok
    Annotate only the information that will be used in our model.
    """
    ann = {'table_id': example['table_id']}

    # 语气词
    # special_words = ['吧','罢','呗','啵','啦','来','了','嘞','哩','咧','咯','啰','喽','吗','嘛','么','哪','呢','呐']
    # for word in special_words:
    #     example['question'] = example['question'].replace(word, '')

    example['question'] = example['question'].strip()   # 去除首尾空格

    _nlu_ann_pr = pr.segment(example['question'],  pos_tagging=False)
    _nlu_ann_pk = seg.cut(example['question'])
    # _nlu_ann_jb = list(jieba.cut_for_search(example['question']))
    _nlu_ann_jb = None

    # 综合 北大 分词和 pynlpir 分词的结果，二者取短
    _nlu_ann = seg_summary(example['question'], _nlu_ann_pr, _nlu_ann_pk, _nlu_ann_jb)
    # _nlu_ann = _nlu_ann_pr
    ann['question'] = example['question']

    # 不加预处理
    # 对原始数据进行操作，不改变原question内容，''.join()后的内容不会发生变化
    processed_nlu_token_list = pre_no_change_process(_nlu_ann)

    # 用于测试
    # ann['orig_tok'] = processed_nlu_token_list

    # 改变question进行token后的内容，以提升完全匹配的准确率
    processed_nlu_token_list = pre_with_change_process(processed_nlu_token_list)

    # 对上一步进行操作的列表进行再操作，因为可能会出现冲突问题，如钱的问题，不是按照先后顺序处理的，如十二/块/五
    processed_nlu_token_list = post_with_change_process(processed_nlu_token_list)

    ann['question_tok'] = processed_nlu_token_list
    # ann['table'] = {
    #     'header': [annotate(h) for h in table['header']],
    # }
    # 测试集中没有sql属性，在这个地方进行判断
    if 'sql' not in example:
        return ann

    ann['sql'] = example['sql']
    ann['query'] = sql = copy.deepcopy(example['sql'])

    # 对sql中conds的属性进行排序，重复的在前
    # conds_sort(ann['sql']['conds'])

    conds1 = ann['sql']['conds']    # "conds": [[0, 2, "大黄蜂"], [0, 2, "密室逃生"]]

    wv_ann1 = []
    for conds11 in conds1:
        # _wv_ann1 = annotate(str(conds11[2]))
        # wv_ann11 = pre_translate(_wv_ann1['gloss'], annotate_dic)
        wv_ann11_str = str(conds11[2])
        # wv_ann1.append(str(conds11[2]))
        wv_ann1.append(wv_ann11_str)

        # Check whether wv_ann exsits inside question_tok

    try:
        # state 变量方便调试
        wvi1_corenlp, state = check_wv_in_nlu_tok(wv_ann1, ann['question_tok'])
        # wvi1_corenlp = check_wv_tok_in_nlu_tok(wv_ann1, ann['question_tok'])
        ann['wvi_corenlp'] = wvi1_corenlp
        ann['stage'] = state
    except:
        ann['wvi_corenlp'] = None
        ann['tok_error'] = 'SQuAD style st, ed are not found under CoreNLP.'

    return ann


def is_valid_example(e):
    if not all([h['words'] for h in e['table']['header']]):
        return False
    headers = [detokenize(h).lower() for h in e['table']['header']]
    if len(headers) != len(set(headers)):
        return False
    input_vocab = set(e['seq_input']['words'])
    for w in e['seq_output']['words']:
        if w not in input_vocab:
            print('query word "{}" is not in input vocabulary.\n{}'.format(w, e['seq_input']['words']))
            return False
    input_vocab = set(e['question']['words'])
    for col, op, cond in e['query']['conds']:
        for w in cond['words']:
            if w not in input_vocab:
                print('cond word "{}" is not in input vocabulary.\n{}'.format(w, e['question']['words']))
                return False
    return True


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
    annotate_dic = {}
    with open('annotate_dic.txt', encoding='utf8') as fin:
        for line in fin:
            # 字典扩容，合并
            annotate_dic.update(json.loads(line))

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
        fout = os.path.join(args.dout, split) + '_tok_v.json'

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
                a = annotate_example_nlpir(d, tables[d['table_id']])
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
