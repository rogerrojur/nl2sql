#!/usr/bin/env python3
# docker run --name corenlp -d -p 9000:9000 vzhong/corenlp-server
# Wonseok Hwang. Jan 6 2019, Comment added
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import records
import ujson as json
from stanza.nlp.corenlp import CoreNLPClient
from tqdm import tqdm
import copy
import re

bai = re.compile(r'(\d+)百')

qian = re.compile(r'(\d+)千')

wan = re.compile(r'(\d+)万')

baiwan = re.compile(r'(\d+)百万')

qianwan = re.compile(r'(\d+)千万')

yi = re.compile(r'(\d+)亿')

def process(s):
    output = bai.sub(lambda x: x.group(1) + '00', s)
    output = qian.sub(lambda x: x.group(1) + '000', output)
    output = wan.sub(lambda x: x.group(1) + '0000', output)
    output = baiwan.sub(lambda x: x.group(1) + '000000', output)
    output = qianwan.sub(lambda x: x.group(1) + '0000000', output)
    return yi.sub(lambda x: x.group(1) + '00000000', output)

client = None

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']
syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE', 'SECTION', 'OP', 'COND', 'QUESTION', 'AGG', 'AGGOPS', 'CONDOPS']



def annotate(sentence, lower=True):
    global client
    if client is None:
        client = CoreNLPClient(default_annotators='ssplit,tokenize'.split(','))
    words, gloss, after = [], [], []
    for s in client.annotate(sentence):
        for t in s:
            words.append(t.word)
            gloss.append(t.originalText)
            after.append(t.after)
    if lower:
        words = [w.lower() for w in words]
    return {
        'gloss': gloss,
        'words': words,
        'after': after,
        }


def annotate_example(example, table):
    ann = {'table_id': example['table_id']}
    ann['question'] = annotate(example['question'])
    ann['table'] = {
        'header': [annotate(h) for h in table['header']],
    }
    ann['query'] = sql = copy.deepcopy(example['sql'])
    for c in ann['query']['conds']:
        c[-1] = annotate(str(c[-1]))

    q1 = 'SYMSELECT SYMAGG {} SYMCOL {}'.format(agg_ops[sql['agg']], table['header'][sql['sel']])
    q2 = ['SYMCOL {} SYMOP {} SYMCOND {}'.format(table['header'][col], cond_ops[op], detokenize(cond)) for col, op, cond in sql['conds']]
    if q2:
        q2 = 'SYMWHERE ' + ' SYMAND '.join(q2) + ' SYMEND'
    else:
        q2 = 'SYMEND'
    inp = 'SYMSYMS {syms} SYMAGGOPS {aggops} SYMCONDOPS {condops} SYMTABLE {table} SYMQUESTION {question} SYMEND'.format(
        syms=' '.join(['SYM' + s for s in syms]),
        table=' '.join(['SYMCOL ' + s for s in table['header']]),
        question=example['question'],
        aggops=' '.join([s for s in agg_ops]),
        condops=' '.join([s for s in cond_ops]),
    )
    ann['seq_input'] = annotate(inp)
    out = '{q1} {q2}'.format(q1=q1, q2=q2) if q2 else q1
    ann['seq_output'] = annotate(out)
    ann['where_output'] = annotate(q2)
    assert 'symend' in ann['seq_output']['words']
    assert 'symend' in ann['where_output']['words']
    return ann

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


def pre_translate(token_list):
    results = []
    dic = zh_digit_dic
    # 由于token_list和where_value均按照这个标准token，可以统一到一个标准
    str_to_str_dic = {'北上':'北京上海','北上广':'北京上海广州','苏杭':'苏州杭州','买入':'增持', '两':'2','闽':'福建',
    '国航':'中国国航中国国际航空有限公司','星期':'周','津厦':'天津厦门','达标':'合格','师大':'师范大学','广东话':'粤语',
    '及格':'合格','工大':'工业大学','开卷':'闭卷','符合':'合格','小汽车':'小型轿车','教师':'老师','SAMSUNG':'三星','首都':'北京市',
    '苏泊尔':'SUPOR','豫':'河南','研究生':'硕士研究生','财经':'经济','BTV':'北京电视台','Duke':'杜克','University':'大学',
    'Press':'出版社','同意':'通过','AAAAA':'5A','AAAA':'4A','AAA':'3A','经贸':'经济与贸易','CITVC':'中国国际电视总公司',
    '央视':'中央电视台','周一至周五':'工作日','HongKongUniversityofScienceandTechnology':'HKUST','星期天':'周日','星期一':'周一',
    '星期二':'周二','星期三':'周三','星期四':'周四','星期五':'周五','星期六':'周六','建行':'建设银行','招行':'招商银行','工行':'工商银行',
    '符合规定':'合格','广警院':'广东警官学院','国体':'国家体育场','CNFIA':'中国食品工业协会','马钢':'马鞍山钢铁股份有限公司','武大':'武汉大学',
    '华科':'华中科技大学','医师':'医生','华师':'华中师范大学','首经贸':'首都经济贸易','社科':'社会科学','北大':'北京大学','浙大':'浙江大学',
    '上交':'上海交通大学','人大':'中国人民大学','南大':'南京大学','辽大':'辽宁大学','广大':'广州大学','厦大':'厦门大学','北师大':'北京师范大学',
    '中山大学':'中大','西财':'西南财大','东航':'东方航空','国泰':'国泰航空','湖南卫视':'湖南卫视芒果','国图':'国家图书馆','央视':'中央电视台',
    '三星':'三星电子电脑有限公司','硕博':'硕士博士','本硕博':'本科硕士博士','我国':'中国'}

    spectial_charlist1 = ['共','下','科','达','线','洲','星','度','川','能','变','化','起','宁','江']   # 和 一 搭配的字，三星，万科，四川
    # for ix, token in enumerate(token_list):
    ix = -1
    while ix < len(token_list) - 1:
        ix += 1
        token = token_list[ix]
        if token in str_to_str_dic:
            results.append(str_to_str_dic[token])
            continue
        if ix < len(token_list) - 1 and (token == '企鹅' and token_list[ix+1] == '公司') or (token == '鹅' and token_list[ix+1] == '厂'):
            results.append('腾讯')
            results.append('公司')
            ix += 1
            continue

        # "16","年"; "一六年"; "今年"
        if token.endswith('年'):
            if len(token) == 3 and token[0] in dic and token[1] in dic:
                pre_tmp_str = '20' if int(helper(token[:2])) <= 20 else '19'
                tmp_str = pre_tmp_str + str(dic[token[0]]) + str(dic[token[1]]) + token[2]
                results.append(tmp_str)
                continue
            if len(token) == 1:
                if ix > 0 and len(token_list[ix-1]) == 2 and str.isdigit(token_list[ix-1]):
                    # 16变成2016
                    results[-1] = '20' + results[-1] + '年'
                    continue
            if len(token) == 2 and token[0] in ['今','去','前']:
                if token[0] == '今':
                    results.append('2019年')
                if token[0] == '去':
                    results.append('2018年')
                if token[0] == '前':
                    results.append('2017年')
                continue
            # 两千年
            if is_all_number_word(token[:-1]):
                results.append(getResultForDigit(token[:-1]) + '年')
                continue

        #  '第二', '第几', '第2'
        if token.startswith('第'):
            tmp_str = '第'
            for i in range(1, len(token)):
                tmp_str += getResultForDigit(token[1:])
            results.append(tmp_str)
            continue

        # '一' '共';
        if token == '一' and ix < len(token_list) - 1 and token_list[ix+1] in spectial_charlist1:
            results.append(token)
            continue

        # 万 科
        if token in ['百', '千', '万', '亿']:
            results.append(token)
            continue

        # '百亿'; '一百亿'; '5万'; '二十'; '三千万'; '三十八万'; '百万';20万
        if is_all_number_word(token):
            results.append(getResultForDigit(token))
            continue

        # 一点六; 零点五; 十二点八
        if is_decimal_number_word(token):
            results.append(getResultForDecimal(token))
            continue

        # 十九点七二块
        if is_decimal_number_word(token[:-1]):
            results.append(getResultForDecimal(token[:-1]) + token[-1])
            continue

        if token[0] == '两' and is_all_number_word(token[1:]):
            results.append(getResultForDigit('二' + token[1:]))
            continue

        # '百分之八'
        if len(token) >= 4 and token[:3] == '百分之':
            if is_all_number_word(token[3:]):
                results.append(getResultForDigit(token[3:])+'%')
                continue
            if is_decimal_number_word(token[3:]):
                results.append(getResultForDecimal(token[3:])+'%')
                continue

        # 一共;百度；万科
        if len(token) > 1 and is_all_number_word(token[:-1]) and token[-1] not in spectial_charlist1:
            results.append(str(getResultForDigit(token[:-1])) + token[-1])
            continue

        if token[-2:] == '月份' and is_all_number_word(token[:-2]):
            results.append(getResultForDigit(token[:-2]) + '月')
            continue

        # 一点六；

        # '二十元';'二十块';"5","角";"四","角钱";"十二","块","五","毛"；"十二点五","元";
        if token[0] == '角' or token[0] == '毛':
            if results and is_all_number_word(results[-1]):
                results[-1] = '0.' + results[-1] + '元'
                continue

        # '五角钱';2角;
        if (len(token) == 2 and token[0] in dic and token[1] == '角') or (len(token) == 3 and token[0] in dic and token[1:] == '角钱'):
            results.append('0.'+getResultForDigit(token[0]) + '元')
            continue

        if (token == '块' or token == '元') and 0 < ix < len(token_list)-1:
            if is_all_number_word(token_list[ix-1]) and is_all_number_word(token_list[ix+1]):
                results[-1] = str(getResultForDigit(token_list[ix-1])) + '.' + str(getResultForDigit(token_list[ix+1])) + '元'
                ix += 1
                continue

        if (token == '块' or token == '元') and ix == len(token_list)-1:
            if is_all_number_word(token_list[ix-1]):
                results[-1] = str(getResultForDigit(token_list[ix-1])) + '元'
                continue

        results.append(token)

    return results


def annotate_example_ws(example, table):
    """
    Jan. 2019: Wonseok
    Annotate only the information that will be used in our model.
    """
    ann = {'table_id': example['table_id']}
    _nlu_ann = annotate(example['question'])
    ann['question'] = example['question']

    # 16, 年; 一六年; 这种形式转化为2016年
    # print(_nlu_ann['gloss'])
    processed_nlu_token_list = pre_translate(_nlu_ann['gloss'])

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
        # wv_ann11 = _wv_ann1['gloss']
        # wv_ann1.append( wv_ann11 )
        _wv_ann1 = annotate(str(conds11[2]))
        wv_ann11 = pre_translate(_wv_ann1['gloss'])
        wv_ann11_str = ''.join(wv_ann11)
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
    parser.add_argument('--din', default='./data/', help='data directory')
    parser.add_argument('--dout', default='./data/tok', help='output directory')
    parser.add_argument('--split', default='train,val,test', help='comma=separated list of splits to process')
    args = parser.parse_args()

    answer_toy = not True
    toy_size = 10

    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    # 加载缩写词对应的词典,对token进行替换
    # str_to_str_dic = {}
    # with open('./auxfiles/vocab.txt', encoding='utf8') as fin:
    #     for line in fin:
    #         arr = line.split(':')
    #         str_to_str_dic[arr[0]] = arr[1].strip()

    # for split in ['train', 'val', 'test']:
    for split in args.split.split(','):
        fsplit = os.path.join(args.din,split, split) + '.json'
        ftable = os.path.join(args.din,split, split) + '.tables.json'
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
                a = annotate_example_ws(d, tables[d['table_id']])
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
