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
    for i_wn, wv_str in enumerate(wv_str_list):
        # stage: 找到子串的阶段，方便调试，字符串表示
        results, stage = find_str_in_list(wv_str, nlu_t1)
        st_idx, ed_idx = results[0] # 选择第1个元素，忽略后面的

        g_wvi1_corenlp.append( [st_idx, ed_idx] )

    return g_wvi1_corenlp, stage


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


def find_str_full_match(s, l):
    # from stack overflow.
    # s 包括 l 中的一个连续的子序列
    results = []
    s_len, l_len = len(s), len(l)

    if not results:
        # print('1-th iter')
        for ix, token in enumerate(l):
            if s[0] == token[0]:
                tmp_str, tmp_idx = '', ix
                while len(tmp_str) < s_len and tmp_idx < l_len:
                    tmp_str += l[tmp_idx]
                    tmp_idx += 1
                if tmp_str == s:
                    # 找到完全匹配
                    return [ix, tmp_idx-1]
    return None


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


def get_similarity(str1, str2):
    return difflib.SequenceMatcher(None, str1, str2).quick_ratio()


def agg_func(token_list, words):
    new_list = token_list

    # 对token元素进行聚合, 如把腾讯/视频 变成 腾讯视频
    for word in words:
        wvi = find_str_full_match(word, new_list)
        if wvi == None:
            continue
        new_list = new_list[:wvi[0]] + [''.join(new_list[ wvi[0] : wvi[1]+1 ])] + new_list[wvi[1]+1:]

    # 将token中的词替换成table中的词
    for i, token in enumerate(new_list):
        if len(token) > 0 and token[0] in '0123456789':
            continue
        if len(token) < 2:
            continue
        ss = set()
        for word in words:
            # 直觉：如果token在words中只出现一次，则进行替换
            if word.find(token) != -1:
                ss.add(token)
        if len(ss) == 1:
            new_list[i] = ss.pop()

    return new_list


def fuzzy_match(token_list, words):
    token_list_len = len(token_list)
    for ix, token in enumerate(token_list):
        if len(token) > 0 and token[0] in '0123456789':
            continue
        if token in words:
            continue
        if token == '':
            continue
        for word in words:
            if word[0] == token[0]:
                max_ratio = 0
                end_index = -1
                for end_ix in range(ix, token_list_len):
                    tmp_str = ''.join(token_list[ix:(end_ix+1)])
                    ratio = get_similarity(tmp_str, word)
                    if ratio > max_ratio:
                        max_ratio = ratio
                        end_index = end_ix
                if max_ratio >= 0.6:
                    token_list[ix] = word
                    for t in range(ix+1, end_index+1):
                        token_list[t] = ''  # 将这个设为''，因为已经被匹配了
                        break

    # 去除空的token
    copy = []
    for r in token_list:
        r = r.strip()
        if r != '' and r != ' ':
            copy.append(r)

    return copy



def retok_by_table(example, table, split):
    question_tok = example['question_tok']
    table_words = set([str(w) for row in table['rows'] for w in row])
    table_words = sorted([w for w in table_words if len(w) < 30], key=lambda x : -len(x))
    
    # 将token list中出现在table_words中的词进行聚合, words中的单词已经按照长度进行排序
    retok_list = agg_func(question_tok, table_words)
    # 模糊匹配，在对table中的词进行聚合之后，再次对部分出现在table中的词进行聚合
    if split == 'test':
        retok_list = fuzzy_match(retok_list, table_words)

    return retok_list



def annotate_example_jieba(example, table, split):
    """
    Jan. 2019: Wonseok
    Annotate only the information that will be used in our model.
    """
    ann = example

    ann['question_tok'] = retok_by_table(example, table, split)

    if 'sql' not in ann:
        return ann

    conds1 = ann['sql']['conds']    # "conds": [[0, 2, "大黄蜂"], [0, 2, "密室逃生"]]
    wv_ann1 = []
    for conds11 in conds1:
        wv_ann11_str = str(conds11[2])
        # wv_ann1.append(str(conds11[2]))
        wv_ann1.append(wv_ann11_str)

    try:
        # state 变量方便调试
        wvi1_corenlp, state = check_wv_in_nlu_tok(wv_ann1, ann['question_tok'])
        ann['wvi_corenlp'] = wvi1_corenlp
        ann['stage'] = state
    except:
        ann['wvi_corenlp'] = None
        ann['tok_error'] = 'SQuAD style st, ed are not found under CoreNLP.'

    if ann['wvi_corenlp'] != None:
        for wvi in ann['wvi_corenlp']:
            if wvi[1] - wvi[0] + 1 >= 3:
                ann = None
                break

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
    # 对分词后的结果重新进行token，使得数据库中的词处于相同的位置
    for split in args.split.split(','):
        fsplit = os.path.join(args.din, split) + '_tok.json'
        ftable = os.path.join(args.din, split) + '.tables.json'
        fout = os.path.join(args.dout, split) + '_retok.json'

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
                a = annotate_example_jieba(d, tables[d['table_id']], split)
                # print(a)
                # if cnt > 10:
                #     break
                # 使用ensure_ascii=False避免写到文件的中文数据是ASCII编码表示
                # 如果生成的数据不符合条件，如wvi差距太大，则剔除
                if a == None:
                    continue
                fo.write(json.dumps(a, ensure_ascii=False) + '\n')
                n_written += 1

                if answer_toy:
                    if cnt > toy_size:
                        break
            print('wrote {} examples'.format(n_written))
