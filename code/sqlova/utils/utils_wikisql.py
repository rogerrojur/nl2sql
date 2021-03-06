# Copyright 2019-present NAVER Corp.
# Apache License v2.0

# Wonseok Hwang

import os, json
import random as rd
from copy import deepcopy
import logging

from matplotlib.pylab import *

import torch
import torchvision.datasets as dsets
import torch.nn as nn
import torch.nn.functional as F


from .utils import generate_perm_inv
from .utils import json_default_type_checker

from .wikisql_formatter import get_squad_style_ans

from collections import defaultdict

import token_utils



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data -----------------------------------------------------------------------------------------------
def load_wikisql(path_wikisql, toy_model, toy_size, bert=False, no_w2i=False, no_hs_tok=False, aug=False):
    # Get data
    train_data, train_table = load_wikisql_data(path_wikisql, mode='train', toy_model=toy_model, toy_size=toy_size, no_hs_tok=no_hs_tok, aug=aug)
    dev_data, dev_table = load_wikisql_data(path_wikisql, mode='val', toy_model=toy_model, toy_size=toy_size, no_hs_tok=no_hs_tok)


    # Get word vector
    if no_w2i:
        w2i, wemb = None, None
    else:
        w2i, wemb = load_w2i_wemb(path_wikisql, bert)


    return train_data, train_table, dev_data, dev_table, w2i, wemb


def load_wikisql_data(path_wikisql, mode='train', toy_model=False, toy_size=10, no_hs_tok=False, aug=False):
    """ Load training sets
    """
    if aug:
        mode = f"{mode}"
        print('Augmented data is loaded!')

    path_sql = os.path.join(path_wikisql, mode, mode+'.json')
    if no_hs_tok:
        path_table = os.path.join(path_wikisql, mode, mode + '.tables.json')
    else:
        path_table = os.path.join(path_wikisql, mode+'_tok.tables.json')

    data = []
    table = {}
    with open(path_table, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if toy_model and idx > toy_size:
                break

            t1 = json.loads(line.strip())
            table[t1['id']] = t1

    with open(path_sql, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if toy_model and idx >= toy_size:
                break

            t1 = json.loads(line.strip())
            # 整合token函数，直接对train/val/test.json中的record进行token后加入data，再构造dataloader
            # None表示token产生的句子不符合条件，只针对train和val
            t1_l = token_utils.token_each(t1, table[t1['table_id']], mode)
            if t1_l:
                data.extend(t1_l)

    return data, table


def load_w2i_wemb(path_wikisql, bert=False):
    """ Load pre-made subset of TAPI.
    """
    if bert:
        with open(os.path.join(path_wikisql, 'w2i_bert.json'), 'r') as f_w2i:
            w2i = json.load(f_w2i)
        wemb = load(os.path.join(path_wikisql, 'wemb_bert.npy'), )#pylab's load = numpy.load
    else:
        with open(os.path.join(path_wikisql, 'w2i.json'), 'r') as f_w2i:
            w2i = json.load(f_w2i)

        wemb = load(os.path.join(path_wikisql, 'wemb.npy'), )
    return w2i, wemb

def get_loader_wikisql(data_train, data_dev, bS, shuffle_train=True, shuffle_dev=False):
    train_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=data_train,
        shuffle=shuffle_train,
        num_workers=0,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    dev_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=data_dev,
        shuffle=shuffle_dev,
        num_workers=0,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    return train_loader, dev_loader


def get_fields_1(t1, tables, no_hs_t=False, no_sql_t=False, generate_mode=False):
    #only query+label data has token, table data does not have token
    nlu1 = t1['question']
    nlu_t1 = t1['question_tok']
    tid1 = t1['table_id']
    if generate_mode:
        sql_i1 = []# generate_result need to convert it to []
        sql_q1 = []# generate_result need to convert it to []
    else:
        sql_i1 = t1['sql']# generate_result need to convert it to []
        sql_q1 = t1['query']# generate_result need to convert it to []
    #sql and query is the same, we may need to delete one
    if no_sql_t:
        sql_t1 = None
    else:
        sql_t1 = t1['query_tok']

    tb1 = tables[tid1]
    if no_hs_t:
        hs_t1 = []
    else:
        hs_t1 = tb1['header_tok']
    hs1 = tb1['header']
    #only nlu1, nlu_t1, tid1, sql_i1, sql_q1, tb1 and hs1 are valid

    return nlu1, nlu_t1, tid1, sql_i1, sql_q1, sql_t1, tb1, hs_t1, hs1

def get_fields(t1s, tables, no_hs_t=False, no_sql_t=False, generate_mode=False):#t1s is the query+label data from one batch, tables is the db.tables

    nlu, nlu_t, tid, sql_i, sql_q, sql_t, tb, hs_t, hs = [], [], [], [], [], [], [], [], []
    for t1 in t1s:#t1 is a query with y value
        nlu1, nlu_t1, tid1, sql_i1, sql_q1, sql_t1, tb1, hs_t1, hs1 = get_fields_1(t1, tables, no_hs_t, no_sql_t, generate_mode=generate_mode)
        nlu.append(nlu1)#ok
        nlu_t.append(nlu_t1)#ok
        tid.append(tid1)#ok
        sql_i.append(sql_i1)#ok
        sql_q.append(sql_q1)#may not ok
        sql_t.append(sql_t1)#not

        tb.append(tb1)#ok
        hs_t.append(hs_t1)#not
        hs.append(hs1)#ok

    return nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hs


# Embedding -------------------------------------------------------------------------

def word_to_idx1(words1, w2i, no_BE):
    w2i_l1 = []
    l1 = len(words1)  # +2 because of <BEG>, <END>


    for w in words1:
        idx = w2i.get(w, 0)
        w2i_l1.append(idx)

    if not no_BE:
        l1 += 2
        w2i_l1 = [1] + w2i_l1 + [2]

    return w2i_l1, l1


def words_to_idx(words, w2i, no_BE=False):
    """
    Input: [ ['I', 'am', 'hero'],
             ['You', 'are 'geneus'] ]
    output:

    w2i =  [ B x max_seq_len, 1]
    wemb = [B x max_seq_len, dim]

    - Zero-padded when word is not available (teated as <UNK>)
    """
    bS = len(words)
    l = torch.zeros(bS, dtype=torch.long).to(device) # length of the seq. of words.
    w2i_l_list = [] # shall be replaced to arr

    #     wemb_NLq_batch = []

    for i, words1 in enumerate(words):

        w2i_l1, l1 = word_to_idx1(words1, w2i, no_BE)
        w2i_l_list.append(w2i_l1)
        l[i] = l1

    # Prepare tensor of wemb
    # overwrite w2i_l
    w2i_l = torch.zeros([bS, int(max(l))], dtype=torch.long).to(device)
    for b in range(bS):
        w2i_l[b, :l[b]] = torch.LongTensor(w2i_l_list[b]).to(device)

    return w2i_l, l

def hs_to_idx(hs_t, w2i, no_BE=False):
    """ Zero-padded when word is not available (teated as <UNK>)
    Treat each "header tokens" as if they are NL-utterance tokens.
    """

    bS = len(hs_t)  # now, B = B_NLq
    hpu_t = [] # header pseudo-utterance
    l_hs = []
    for hs_t1 in hs_t:
        hpu_t  += hs_t1
        l_hs1 = len(hs_t1)
        l_hs.append(l_hs1)

    w2i_hpu, l_hpu = words_to_idx(hpu_t, w2i, no_BE=no_BE)
    return w2i_hpu, l_hpu, l_hs


# Encoding ---------------------------------------------------------------------

def encode(lstm, wemb_l, l, return_hidden=False, hc0=None, last_only=False):
    """ [batch_size, max token length, dim_emb]
    """
    bS, mL, eS = wemb_l.shape


    # sort before packking
    l = array(l)#l is the list of how many tokens in this question, so it is a list of int
    perm_idx = argsort(-l)#sort the indices from large to small
    perm_idx_inv = generate_perm_inv(perm_idx)#so now the largest element is in the position when the value in this list is 0

    # pack sequence
    #reconstruct the order of wemb_l and l from large to small length and then pack sequence

    packed_wemb_l = nn.utils.rnn.pack_padded_sequence(wemb_l[perm_idx, :, :],
                                                      l[perm_idx],
                                                      batch_first=True)
    # Time to encode
    if hc0 is not None:
        hc0 = (hc0[0][:, perm_idx], hc0[1][:, perm_idx])

    # ipdb.set_trace()
    packed_wemb_l = packed_wemb_l.float() # I don't know why..
    packed_wenc, hc_out = lstm(packed_wemb_l, hc0)#packed_wenc is (seq_length, batch, hiddenSize * nbDirection)
    hout, cout = hc_out

    # unpack
    wenc, _l = nn.utils.rnn.pad_packed_sequence(packed_wenc, batch_first=True)

    if last_only:
        # Take only final outputs for each columns.
        wenc = wenc[tuple(range(bS)), l[perm_idx] - 1]  # [batch_size, dim_emb]
        wenc.unsqueeze_(1)  # [batch_size, 1, dim_emb]

    wenc = wenc[perm_idx_inv]



    if return_hidden:
        # hout.shape = [batch, seq_len, num_of_layer * number_of_direction ] w/ batch_first.. w/o batch_first? I need to see.
        hout = hout[:, perm_idx_inv].to(device)
        cout = cout[:, perm_idx_inv].to(device)  # Is this correct operation?

        return wenc, hout, cout
    else:
        return wenc


def encode_hpu(lstm, wemb_hpu, l_hpu, l_hs):
    wenc_hpu, hout, cout = encode( lstm,
                                   wemb_hpu,
                                   l_hpu,
                                   return_hidden=True,
                                   hc0=None,
                                   last_only=True )

    wenc_hpu = wenc_hpu.squeeze(1)
    bS_hpu, mL_hpu, eS = wemb_hpu.shape
    hS = wenc_hpu.size(-1)

    wenc_hs = wenc_hpu.new_zeros(len(l_hs), max(l_hs), hS)
    wenc_hs = wenc_hs.to(device)

    # Re-pack according to batch.
    # ret = [B_NLq, max_len_headers_all, dim_lstm]
    # sum(hs) = len(l_hpu) so l_hpu 是展开来的col 长度列表 也即是 每一个col多少字， l_hs是这个玩意的大小是有多少个col
    st = 0
    #print('l_hpu: ', len(l_hpu), sum(l_hs), '; wenc_hs: ', wenc_hs.size(), '; wenc_hpu: ', wenc_hpu.size(), '; wemb_hpu: ', wemb_hpu.size())
    for i, l_hs1 in enumerate(l_hs):
        wenc_hs[i, :l_hs1] = wenc_hpu[st:(st + l_hs1)]
        st += l_hs1

    return wenc_hs

def encode_npu(lstm, wemb_npu, l_npu, l_token):
    return encode_hpu(lstm, wemb_npu, l_npu, l_token)



# Statistics -------------------------------------------------------------------------------------------------------------------



def get_wc1(conds):
    """
    [ [wc, wo, wv],
      [wc, wo, wv], ...
    ]
    """
    wc1 = []
    for cond in conds:
        wc1.append(int(cond[0]))
    return wc1


def get_wo1(conds):
    """
    [ [wc, wo, wv],
      [wc, wo, wv], ...
    ]
    """
    wo1 = []
    for cond in conds:
        wo1.append(int(cond[1]))
    return wo1


def get_wv1(conds):
    """
    [ [wc, wo, wv],
      [wc, wo, wv], ...
    ]
    """
    wv1 = []
    for cond in conds:
        wv1.append(cond[2])
    return wv1


def get_wrcn1(cols):
    d = defaultdict(int)
    if len(cols) >= 2:
        for col in cols:
            d[col] += 1
        for key in d:
            if d[key] >= 2:
                return [int(key), d[key]]
    return [-1, -1]


def re_order(conds, rp_key):
    srr = [i for i in range(len(conds)) if conds[i] == rp_key]
    err = [i for i in range(len(conds)) if i not in srr]
    return array(conds)[srr + err].tolist()


def get_g(sql_i):
    """ for backward compatibility, separated with get_g"""
    g_sn = []#sel nb list of int
    g_sc = []#sel col list of list
    g_sa = []#sel agg list of list
    g_wn = []
    g_wr = []#whe relation list of int
    g_dwn = []#distinct where col nb
    g_wc = []#whe col list of list
    g_wo = []#whe op list of list
    g_wv = []#whe val list of list
    g_r_c_n = []#whe repeated col index and its number list of list(type : [col_n, nb]) if [-1, -1] it means there is no repeated col
    wvi_change_index = []
    for psql_i1 in sql_i:
        if (len(psql_i1["sel"]) == len(psql_i1["agg"])):
            g_sn.append(len(psql_i1["sel"]))
            sels = psql_i1["sel"]
            for i in range(len(sels)):
                sels[i] = int(sels[i])
            sels_index = array(sels).argsort().tolist()#新尝试
            sels.sort()#新尝试
            g_sc.append(sels)
            aggs = psql_i1["agg"]
            for i in range(len(aggs)):
                aggs[i] = int(aggs[i])
            g_sa.append(array(aggs)[sels_index].tolist())

            conds = psql_i1['conds']
            conds_index = list(map(lambda x : x[0], array(conds).argsort(axis=0).tolist()))#新尝试
            wvi_change_index.append(conds_index)#新尝试
            conds.sort(key=lambda x : x[0])#新尝试
            if len(conds) != 0:
                for i in range(len(conds)):
                    for j in range(2):
                        conds[i][j] = int(conds[i][j])
            if all([0 <= e <= 5 for e in g_sa[-1]]):#if agg is valid 0~5
                g_wr.append(int(psql_i1["cond_conn_op"]))
                if 0 <= g_wr[-1] <= 2:
                    g_r_c_n.append(get_wrcn1(get_wc1(conds)))
                    g_wc.append( get_wc1(conds))
                    g_wn.append(len(g_wc[-1]))
                    g_dwn.append(g_wn[-1] if g_r_c_n[-1][0] == -1 else g_wn[-1] - g_r_c_n[-1][1] + 1)
                    g_wo.append( get_wo1(conds) )
                    g_wv.append( get_wv1(conds) )
        else:
            raise EnvironmentError
    #print(g_wc)
    return g_sn, g_sc, g_sa, g_wn, g_wr, g_dwn, g_wc, g_wo, g_wv, g_r_c_n, wvi_change_index

def get_g_wvi_corenlp(t, wvi_change_index):
    g_wvi_corenlp = []
    for t1, wvi_change_index1 in zip(t, wvi_change_index):
        '''
        print('-------------------error-----------------------')
        print('wvi_corenlp: ', t1['wvi_corenlp'])
        print('wvi_change_index: ', wvi_change_index1)
        print(t1)
        print('-------------------error-----------------------')
        '''
        g_wvi_corenlp.append(array(t1['wvi_corenlp'])[wvi_change_index1].tolist())
    return g_wvi_corenlp


def update_w2i_wemb(word, wv, idx_w2i, n_total, w2i, wemb):
    """ Follow same approach from SQLNet author's code.
        Used inside of generaet_w2i_wemb.
    """

    # global idx_w2i, w2i, wemb  # idx, word2vec, word to idx dictionary, list of embedding vec, n_total: total number of words
    if (word in wv) and (word not in w2i):
        idx_w2i += 1
        w2i[word] = idx_w2i
        wemb.append(wv[word])
    n_total += 1
    return idx_w2i, n_total

def make_w2i_wemb(args, path_save_w2i_wemb, wv, data_train, data_dev, data_test, table_train, table_dev, table_test):

    w2i = {'<UNK>': 0, '<BEG>': 1, '<END>': 2}  # to use it when embeds NL query.
    idx_w2i = 2
    n_total = 3

    wemb = [np.zeros(300, dtype=np.float32) for _ in range(3)]  # 128 is of TAPI vector.
    idx_w2i, n_total = generate_w2i_wemb(data_train, wv, idx_w2i, n_total, w2i, wemb)
    idx_w2i, n_total = generate_w2i_wemb_table(table_train, wv, idx_w2i, n_total, w2i, wemb)

    idx_w2i, n_total = generate_w2i_wemb(data_dev, wv, idx_w2i, n_total, w2i, wemb)
    idx_w2i, n_total = generate_w2i_wemb_table(table_dev, wv, idx_w2i, n_total, w2i, wemb)

    idx_w2i, n_total = generate_w2i_wemb(data_test, wv, idx_w2i, n_total, w2i, wemb)
    idx_w2i, n_total = generate_w2i_wemb_table(table_test, wv, idx_w2i, n_total, w2i, wemb)

    path_w2i = os.path.join(path_save_w2i_wemb, 'w2i.json')
    path_wemb = os.path.join(path_save_w2i_wemb, 'wemb.npy')

    wemb = np.stack(wemb, axis=0)

    with open(path_w2i, 'w') as f_w2i:
        json.dump(w2i, f_w2i)

    np.save(path_wemb, wemb)

    return w2i, wemb

def generate_w2i_wemb_table(tables, wv, idx_w2i, n_total, w2i, wemb):
    """ Generate subset of GloVe
        update_w2i_wemb. It uses wv, w2i, wemb, idx_w2i as global variables.

        To do
        1. What should we do with the numeric?
    """
    # word_set from NL query
    for table_id, table_contents in tables.items():

        # NLq = t1['question']
        # word_tokens = NLq.rstrip().replace('?', '').split(' ')
        headers = table_contents['header_tok'] # [ ['state/terriotry'], ['current', 'slogan'], [],
        for header_tokens in headers:
            for token in header_tokens:
                idx_w2i, n_total = update_w2i_wemb(token, wv, idx_w2i, n_total, w2i, wemb)
                # WikiSQL generaets unbelivable query... using state/territory in the NLq. Unnatural.. but as is
                # when there is slash, unlike original SQLNet which treats them as single token, we use
                # both tokens. e.g. 'state/terriotry' -> 'state'
                # token_spl = token.split('/')
                # for token_spl1 in token_spl:
                #         idx_w2i, n_total = update_w2i_wemb(token_spl1, wv, idx_w2i, n_total, w2i, wemb)

    return idx_w2i, n_total
def generate_w2i_wemb(train_data, wv, idx_w2i, n_total, w2i, wemb):
    """ Generate subset of GloVe
        update_w2i_wemb. It uses wv, w2i, wemb, idx_w2i as global variables.

        To do
        1. What should we do with the numeric?
    """
    # word_set from NL query
    for i, t1 in enumerate(train_data):
        # NLq = t1['question']
        # word_tokens = NLq.rstrip().replace('?', '').split(' ')
        word_tokens = t1['question_tok']
        # Currently, TAPI does not use "?". So, it is removed.
        for word in word_tokens:
            idx_w2i, n_total = update_w2i_wemb(word, wv, idx_w2i, n_total, w2i, wemb)
            n_total += 1

    return idx_w2i, n_total

def generate_w2i_wemb_e2k_headers(e2k_dicts, wv, idx_w2i, n_total, w2i, wemb):
    """ Generate subset of TAPI from english-to-korean dict of table headers etc..
        update_w2i_wemb. It uses wv, w2i, wemb, idx_w2i as global variables.

        To do
        1. What should we do with the numeric?
           Current version do not treat them specially. But this would be modified later so that we can use tags.
    """
    # word_set from NL query
    for table_name, e2k_dict in e2k_dicts.items():
        word_tokens_list = list(e2k_dict.values())
        # Currently, TAPI does not use "?". So, it is removed.
        for word_tokens in word_tokens_list:
            for word in word_tokens:
                idx_w2i, n_total = update_w2i_wemb(word, wv, idx_w2i, n_total, w2i, wemb)
                n_total += 1

    return idx_w2i, n_total


# BERT =================================================================================================================
def tokenize_nlu1(tokenizer, nlu1):
    nlu1_tok = tokenizer.tokenize(nlu1)
    return nlu1_tok


def tokenize_hds1(tokenizer, hds1):
    hds_all_tok = []
    for hds11 in hds1:
        sub_tok = tokenizer.tokenize(hds11)
        hds_all_tok.append(sub_tok)

def generate_inputs(tokenizer, nlu1_tok, hds1):
    tokens = []
    segment_ids = []

    tokens.append("[CLS]")
    i_st_nlu = len(tokens)  # to use it later

    segment_ids.append(0)
    for token in nlu1_tok:
        tokens.append(token)
        segment_ids.append(0)
    i_ed_nlu = len(tokens)
    tokens.append("[SEP]")
    segment_ids.append(0)

    i_hds = []
    # for doc
    for i, hds11 in enumerate(hds1):
        i_st_hd = len(tokens)
        sub_tok = tokenizer.tokenize(hds11)
        tokens += sub_tok
        i_ed_hd = len(tokens)
        i_hds.append((i_st_hd, i_ed_hd))
        segment_ids += [1] * len(sub_tok)
        if i < len(hds1)-1:
            tokens.append("[SEP]")
            segment_ids.append(0)
        elif i == len(hds1)-1:
            tokens.append("[SEP]")
            segment_ids.append(1)
        else:
            raise EnvironmentError

    i_nlu = (i_st_nlu, i_ed_nlu)

    return tokens, segment_ids, i_nlu, i_hds#smallest tokens; 0 is question or sep, 1 is header or end; (question start, question end -1); a list of (col start, col end - 1) 

def gen_l_hpu(i_hds):
    """
    # Treat columns as if it is a batch of natural language utterance with batch-size = # of columns * # of batch_size
    i_hds = [(17, 18), (19, 21), (22, 23), (24, 25), (26, 29), (30, 34)])
    """
    l_hpu = []
    for i_hds1 in i_hds:
        for i_hds11 in i_hds1:
            l_hpu.append(i_hds11[1] - i_hds11[0])

    return l_hpu

def get_bert_output_s2s(model_bert, tokenizer, nlu_t, hds, sql_vocab, max_seq_length):
    """
    s2s version. Treat SQL-tokens as pseudo-headers
    sql_vocab = ("sql select", "sql where", "sql and", "sql equal", "sql greater than", "sql less than")

    e.g.)
    Q: What is the name of the player with score greater than 15?
    H: Name of the player, score
    Input: [CLS], what, is, ...,
    [SEP], name, of, the, player, [SEP], score,
    [SEP] sql, select, [SEP], sql, where, [SEP], sql, and, [SEP], ...

    Here, input is tokenized further by WordPiece (WP) tokenizer and fed into BERT.

    INPUT
    :param model_bert:
    :param tokenizer: WordPiece toknizer
    :param nlu: Question
    :param nlu_t: CoreNLP tokenized nlu.
    :param hds: Headers
    :param hs_t: None or 1st-level tokenized headers
    :param max_seq_length: max input token length

    OUTPUT
    tokens: BERT input tokens
    nlu_tt: WP-tokenized input natural language questions
    orig_to_tok_index: map the index of 1st-level-token to the index of 2nd-level-token
    tok_to_orig_index: inverse map.

    """
    

    l_n = []
    l_hs = []  # The length of columns for each batch
    l_input = []
    input_ids = []
    tokens = []
    segment_ids = []
    input_mask = []

    i_nlu = []  # index to retreive the position of contextual vector later.
    i_hds = []
    i_sql_vocab = []

    nlu_tt = []

    t_to_tt_idx = []
    tt_to_t_idx = []
    for b, nlu_t1 in enumerate(nlu_t):
        

        hds1 = hds[b]
        l_hs.append(len(hds1))


        # 1. 2nd tokenization using WordPiece
        tt_to_t_idx1 = []  # number indicates where sub-token belongs to in 1st-level-tokens (here, CoreNLP).
        t_to_tt_idx1 = []  # orig_to_tok_idx[i] = start index of i-th-1st-level-token in all_tokens.
        nlu_tt1 = []  # all_doc_tokens[ orig_to_tok_idx[i] ] returns first sub-token segement of i-th-1st-level-token
        for (i, token) in enumerate(nlu_t1):
            t_to_tt_idx1.append(
                len(nlu_tt1))  # all_doc_tokens[ indicate the start position of original 'white-space' tokens.
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tt_to_t_idx1.append(i)
                nlu_tt1.append(sub_token)  # all_doc_tokens are further tokenized using WordPiece tokenizer
        nlu_tt.append(nlu_tt1)
        tt_to_t_idx.append(tt_to_t_idx1)
        t_to_tt_idx.append(t_to_tt_idx1)

        l_n.append(len(nlu_tt1))
        #         hds1_all_tok = tokenize_hds1(tokenizer, hds1)



        # [CLS] nlu [SEP] col1 [SEP] col2 [SEP] ...col-n [SEP]
        # 2. Generate BERT inputs & indices.
        # Combine hds1 and sql_vocab
        tokens1, segment_ids1, i_sql_vocab1, i_nlu1, i_hds1 = generate_inputs_s2s(tokenizer, nlu_tt1, hds1, sql_vocab)

        # i_hds1
        input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)

        # Input masks
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask1 = [1] * len(input_ids1)

        # 3. Zero-pad up to the sequence length.
        l_input.append( len(input_ids1) )
        while len(input_ids1) < max_seq_length:
            input_ids1.append(0)
            input_mask1.append(0)
            segment_ids1.append(0)

        assert len(input_ids1) == max_seq_length
        assert len(input_mask1) == max_seq_length
        assert len(segment_ids1) == max_seq_length

        input_ids.append(input_ids1)
        tokens.append(tokens1)
        segment_ids.append(segment_ids1)
        input_mask.append(input_mask1)

        i_nlu.append(i_nlu1)
        i_hds.append(i_hds1)
        i_sql_vocab.append(i_sql_vocab1)

    # Convert to tensor
    all_input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(device)

    # 4. Generate BERT output.
    all_encoder_layer, pooled_output = model_bert(all_input_ids, all_segment_ids, all_input_mask)

    # 5. generate l_hpu from i_hds
    l_hpu = gen_l_hpu(i_hds)

    return all_encoder_layer, pooled_output, tokens, i_nlu, i_hds, i_sql_vocab, \
           l_n, l_hpu, l_hs, l_input, \
           nlu_tt, t_to_tt_idx, tt_to_t_idx


def get_bert_output(model_bert, tokenizer, nlu_t, hds, max_seq_length):
    """
    Here, input is toknized further by WordPiece (WP) tokenizer and fed into BERT.

    INPUT
    :param model_bert:
    :param tokenizer: WordPiece toknizer
    :param nlu: Question
    :param nlu_t: CoreNLP tokenized nlu.
    :param hds: Headers
    :param hs_t: None or 1st-level tokenized headers
    :param max_seq_length: max input token length

    OUTPUT
    tokens: BERT input tokens
    nlu_tt: WP-tokenized input natural language questions
    orig_to_tok_index: map the index of 1st-level-token to the index of 2nd-level-token
    tok_to_orig_index: inverse map.

    """
    
    l_npu = [] #(total number of tokens, how many tokens of token in this token)
    l_token = [] #how many tokens in this sentense
    i_tks = []

    l_n = []
    l_hs = []  # The length of columns for each batch

    input_ids = []
    tokens = []
    segment_ids = []
    input_mask = []

    i_nlu = []  # index to retreive the position of contextual vector later.
    i_hds = []

    nlu_tt = []

    t_to_tt_idx = []
    tt_to_t_idx = []
    for b, nlu_t1 in enumerate(nlu_t):#for each datum
        l_token.append(len(nlu_t1))
        i_tks1 = []
        st = 1

        hds1 = hds[b]#header: column names
        l_hs.append(len(hds1))#how many columns in this table


        # 1. 2nd tokenization using WordPiece
        tt_to_t_idx1 = []  # number indicates where sub-token belongs to in 1st-level-tokens (here, CoreNLP).
        t_to_tt_idx1 = []  # orig_to_tok_idx[i] = start index of i-th-1st-level-token in all_tokens.
        nlu_tt1 = []  # all_doc_tokens[ orig_to_tok_idx[i] ] returns first sub-token segement of i-th-1st-level-token
        for (i, token) in enumerate(nlu_t1):
            t_to_tt_idx1.append(
                len(nlu_tt1))  # all_doc_tokens[ indicate the start position of original 'white-space' tokens.
            sub_tokens = tokenizer.tokenize(token)
            i_tks1.append((st, st + len(sub_tokens)))
            st += len(sub_tokens)
            l_npu.append(len(sub_tokens))
            for sub_token in sub_tokens:
                tt_to_t_idx1.append(i)
                nlu_tt1.append(sub_token)  # all_doc_tokens are further tokenized using WordPiece tokenizer
                #this is to separate an unknow word to many known words
        nlu_tt.append(nlu_tt1)#all smallest peceices tokens in each question, and it is list of list, first dimension is question, second is token in each question
        tt_to_t_idx.append(tt_to_t_idx1)#these smallest peceices tokens belong to original token's index, list of list
        t_to_tt_idx.append(t_to_tt_idx1)#original token indices, list of list
        
        i_tks.append(i_tks1)
        l_n.append(len(nlu_tt1))# how many valid token in this question
        #         hds1_all_tok = tokenize_hds1(tokenizer, hds1)



        # [CLS] nlu [SEP] col1 [SEP] col2 [SEP] ...col-n [SEP]
        # 2. Generate BERT inputs & indices.
        tokens1, segment_ids1, i_nlu1, i_hds1 = generate_inputs(tokenizer, nlu_tt1, hds1)#word peceices tokenizer, smallest pecies tokens in this question, this table header
        input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)#it will use vacab to convert token to index

        # Input masks
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask1 = [1] * len(input_ids1)

        # 3. Zero-pad up to the sequence length.
        while len(input_ids1) < max_seq_length:
            input_ids1.append(0)
            input_mask1.append(0)
            segment_ids1.append(0)

        assert len(input_ids1) == max_seq_length
        assert len(input_mask1) == max_seq_length
        assert len(segment_ids1) == max_seq_length

        input_ids.append(input_ids1)
        tokens.append(tokens1)#tokens and input_ids is one to one for each
        segment_ids.append(segment_ids1)
        input_mask.append(input_mask1)

        i_nlu.append(i_nlu1)
        i_hds.append(i_hds1)

    # Convert to tensor
    all_input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(device)

    # 4. Generate BERT output.
    all_encoder_layer, pooled_output = model_bert(all_input_ids, all_segment_ids, all_input_mask)

    # 5. generate l_hpu from i_hds
    l_hpu = gen_l_hpu(i_hds)

    return all_encoder_layer, pooled_output, tokens, i_nlu, i_hds, \
           l_n, l_hpu, l_hs, \
           nlu_tt, t_to_tt_idx, tt_to_t_idx, l_npu, l_token, i_tks



def get_wemb_n(i_nlu, l_n, hS, num_hidden_layers, all_encoder_layer, num_out_layers_n):
    """
    only get the later part of hidden layers called out layers
    Get the representation of each tokens.
    """
    bS = len(l_n)
    l_n_max = max(l_n)# the number of tokens in each datum
    wemb_n = torch.zeros([bS, l_n_max, hS * num_out_layers_n]).to(device)
    
    for b in range(bS):
        # [B, max_len, dim]
        # Fill zero for non-exist part.
        i_nlu1 = i_nlu[b]
        for i_noln in range(num_out_layers_n):
            i_layer = num_hidden_layers - 1 - i_noln
            st = i_noln * hS
            ed = (i_noln + 1) * hS
            wemb_n[b, 0:(i_nlu1[1] - i_nlu1[0]), st:ed] = all_encoder_layer[i_layer][b, i_nlu1[0]:i_nlu1[1], :]#inverse layer outputs
    return wemb_n
    #


def get_wemb_h(i_hds, l_hpu, l_hs, hS, num_hidden_layers, all_encoder_layer, num_out_layers_h):
    """
    As if
    each sub list is about one col in one table, so it means it contains all sub tokens in this single column
    [ [table-1-col-1-tok1, t1-c1-t2, ...],
       [t1-c2-t1, t1-c2-t2, ...].
       ...
       [t2-c1-t1, ...,]
    ]
    """
    l_hpu_max = max(l_hpu)#the max sub tokens number of a single col
    num_of_all_hds = sum(l_hs)# the sum of how many col in the table are used for one question
    wemb_h = torch.zeros([num_of_all_hds, l_hpu_max, hS * num_out_layers_h]).to(device)
    b_pu = -1
    for b, i_hds1 in enumerate(i_hds):# each single datum of one batch
        for b1, i_hds11 in enumerate(i_hds1):# each col from one single datum
            b_pu += 1
            for i_nolh in range(num_out_layers_h):
                i_layer = num_hidden_layers - 1 - i_nolh
                st = i_nolh * hS
                ed = (i_nolh + 1) * hS
                wemb_h[b_pu, 0:(i_hds11[1] - i_hds11[0]), st:ed] \
                    = all_encoder_layer[i_layer][b, i_hds11[0]:i_hds11[1],:]#last dimension is feature this is [distinct cols, its time step (it is not base on char, it is base on smallest token), feature (hiddensize * layernb)]


    return wemb_h



def get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length, num_out_layers_n=1, num_out_layers_h=1, num_out_layers_v=1):

    # get contextual output of all tokens from bert
    all_encoder_layer, pooled_output, tokens, i_nlu, i_hds,\
    l_n, l_hpu, l_hs, \
    nlu_tt, t_to_tt_idx, tt_to_t_idx, l_npu, l_token, i_tks = get_bert_output(model_bert, tokenizer, nlu_t, hds, max_seq_length)
    # all_encoder_layer: BERT outputs from all layers.
    # pooled_output: output of [CLS] vec.
    # tokens: BERT intput tokens
    # i_nlu: start and end indices of question in tokens
    # i_hds: start and end indices of headers for each col


    # get the wemb
    wemb_n = get_wemb_n(i_nlu, l_n, bert_config.hidden_size, bert_config.num_hidden_layers, all_encoder_layer,
                        num_out_layers_n)

    wemb_h = get_wemb_h(i_hds, l_hpu, l_hs, bert_config.hidden_size, bert_config.num_hidden_layers, all_encoder_layer,
                        num_out_layers_h)
    
    wemb_v = get_wemb_h(i_tks, l_npu, l_token, bert_config.hidden_size, bert_config.num_hidden_layers, all_encoder_layer,
                        num_out_layers_v)
    
    #print('wemb_n: ', wemb_n.size(), 'wemb_h: ', wemb_h.size(), 'wemb_v: ', wemb_v.size())

    return wemb_n, wemb_h, l_n, l_hpu, l_hs, \
           nlu_tt, t_to_tt_idx, tt_to_t_idx, wemb_v, l_npu, l_token


def gen_pnt_n(g_wvi, mL_w, mL_nt):
    """
    Generate one-hot idx indicating vectors with their lenghts.

    :param g_wvi: e.g. [[[0, 6, 7, 8, 15], [0, 1, 2, 3, 4, 15]], [[0, 1, 2, 3, 16], [0, 7, 8, 9, 16]]]
    where_val idx in nlu_t. 0 = <BEG>, -1 = <END>.
    :param mL_w: 4
    :param mL_nt: 200
    :return:
    """
    bS = len(g_wvi)
    for g_wvi1 in g_wvi:
        for g_wvi11 in g_wvi1:
            l11 = len(g_wvi11)

    mL_g_wvi = max([max([0] + [len(tok) for tok in gwsi]) for gwsi in g_wvi]) - 1
    # zero because of '' case.
    # -1 because we already have <BEG>
    if mL_g_wvi < 1:
        mL_g_wvi = 1
    # NLq_token_pos = torch.zeros(bS, 5 - 1, mL_g_wvi, self.max_NLq_token_num)

    # l_g_wvi = torch.zeros(bS, 5 - 1)
    pnt_n = torch.zeros(bS, mL_w, mL_g_wvi, mL_nt).to(device) # one hot
    l_g_wvi = torch.zeros(bS, mL_w).to(device)

    for b, g_wvi1 in enumerate(g_wvi):
        i_wn = 0  # To prevent error from zero number of condition.
        for i_wn, g_wvi11 in enumerate(g_wvi1):
            # g_wvi11: [0, where_conds pos in NLq, end]
            g_wvi11_n1 = g_wvi11[:-1]  # doesn't count <END> idx.
            l_g_wvi[b, i_wn] = len(g_wvi11_n1)
            for t, idx in enumerate(g_wvi11_n1):
                pnt_n[b, i_wn, t, idx] = 1

            # Pad
        if i_wn < (mL_w - 1):  # maximum number of conidtions is 4
            pnt_n[b, i_wn + 1:, 0, 1] = 1  # # cannot understand... [<BEG>, <END>]??
            l_g_wvi[b, i_wn + 1:] = 1  # it means there is only <BEG>.


    return pnt_n, l_g_wvi

def pred_sn(s_sn):
    """
    return : [ pr_sn1, pr_sn2, ...]
    """
    return [s_sn1.argmax().item() for s_sn1 in s_sn]

def guide_pred_sc(sn, s_sc, sa, tb):
    s_sc = torch.sigmoid(s_sc)
    pr_sc = []
    for b, sn1 in enumerate(sn):
        now_tb = tb[b]
        s_sc1 = s_sc[b]
        sa1 = sa[b]

        pr_sc1 = argsort(-s_sc1.data.cpu().numpy()).tolist()
        
        sc = [-1 for _ in range(sn1)]
        
        for sn11 in range(sn1):
            if sa1[sn11] != 0 and sa1[sn11] != 4:
                if sn11 == 0:
                    for sc1 in pr_sc1:
                        if get_col_type(now_tb, sc1) == 'real':
                            sc[sn11] = sc1
                            break
                else:
                    for sc1 in pr_sc1:
                        if sc1 not in sc[:sc1] and get_col_type(now_tb, sc1) == 'real':
                            sc[sn11] = sc1
                            break
            else:
                if sn11 == 0:
                    sc[sn11] = pr_sc1[0]
                else:
                    for sc1 in pr_sc1:
                        if sc1 not in sc[:sc1]:
                            sc[sn11] = sc1
                            break
        
        for i in range(len(sc)):
            if sc[i] == -1:
                sc[i] = pr_sc1[0]
        
        pr_sc.append(sc)
    return pr_sc

def pred_sc(sn, s_sc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    s_sc = torch.sigmoid(s_sc)
    pr_sc = []
    for b, sn1 in enumerate(sn):
        s_sc1 = s_sc[b]

        pr_sc1 = argsort(-s_sc1.data.cpu().numpy())[:sn1].tolist()
        pr_sc1.sort()#sort 可能是导致错位的罪魁祸首

        pr_sc.append(pr_sc1)
    return pr_sc

def pred_sc_without_sort(sn, s_sc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    s_sc = torch.sigmoid(s_sc)
    pr_sc = []
    for b, sn1 in enumerate(sn):
        s_sc1 = s_sc[b]

        pr_sc1 = argsort(-s_sc1.data.cpu().numpy())[:sn1].tolist()

        pr_sc.append(pr_sc1)
    return pr_sc


def pred_sc_beam(s_sc, beam_size):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_sc_beam = []


    for s_sc1 in s_sc:
        val, idxes = s_sc1.topk(k=beam_size)
        pr_sc_beam.append(idxes.tolist())

    return pr_sc_beam

def pred_sa(sn, s_sa):
    # s_wo = [B, 3, n_op] 3代表了select col的最大个数为4
    pr_sa_a = s_sa.argmax(dim=2)  # [B, 3]
    pr_sa = []
    # get g_num
    for b, pr_sa_a1 in enumerate(pr_sa_a):
        sn1 = sn[b]
        pr_sa.append(pr_sa_a1.data.cpu().numpy()[:sn1].tolist())

    return pr_sa


def pred_wn(s_wn):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_wn = []
    for s_wn1 in s_wn:
        pr_wn.append(s_wn1.argmax().item())
        # print(pr_wn, s_wn1)
        # if s_wn1.argmax().item() == 3:
        #     input('')

    return pr_wn

def pred_wr(wn, s_wr):
    return [s_wr1.argmax().item() if wn1 >= 2 else 0 for s_wr1, wn1 in zip(s_wr, wn)] #it can only predict "and" "or"

def re_pred_wr(wr, hrpc):
    return [2 if hrpc1 else wr1 for hrpc1, wr1 in zip(hrpc, wr)]

def pred_hrpc(s_hrpc):
    return [s_hrpc1.argmax().item() for s_hrpc1 in s_hrpc]

def re_pred_hrpc(wr, hrpc):
    return [0 if wr1 <= 1 else hrpc1 for hrpc1, wr1 in zip(hrpc, wr)]

def pred_dwn(wn, hrpc):
    return [wn1 if hrpc1 == 0 else 1 for wn1, hrpc1 in zip(wn, hrpc)]

def pred_wc_old(sql_i, s_wc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_wc = []
    for b, sql_i1 in enumerate(sql_i):
        wn = len(sql_i1['conds'])
        s_wc1 = s_wc[b]

        pr_wc1 = argsort(-s_wc1.data.cpu().numpy())[:wn].tolist()
        pr_wc1.sort()#sort 可能是导致错位的罪魁祸首

        pr_wc.append(pr_wc1)
    return pr_wc

def guide_pred_wc(hrpc, wn, s_wc, wo, tb, l_hs, wvi, nlu_t, engine):
    s_wc = torch.sigmoid(s_wc)
    pr_wc = []
    for b, wn1 in enumerate(wn):
        nlu_t1 = nlu_t[b]
        wvi1 = wvi[b]
        l_hs1 = l_hs[b]
        now_tb = tb[b]
        s_wc1 = s_wc[b]
        wo1 = wo[b]
        pr_wc1 = argsort(-s_wc1.data.cpu().numpy()).tolist()
        wc = [-1 for _ in range(wn1)]
        if hrpc[b] == 1:
            if 0 in wo1 or 1 in wo1:
                for pr_wc11 in pr_wc1:
                    if get_col_type(now_tb, pr_wc11) == 'real':
                        wc = [pr_wc11] * wn1
                        break
            else:
                wc = [pr_wc1[0]] * wn1
                        
        else:
            for wn11 in range(wn1):
                if wn11 == 0:
                    if wo1[wn11] <= 1:
                        for pr_wc11 in pr_wc1:
                            if pr_wc11 >= l_hs1:
                                continue
                            if get_col_type(now_tb, pr_wc11) == 'real':
                                wc[wn11] = pr_wc11
                                break
                    elif wo1[wn11] == 3:
                        wc[wn11] = pr_wc1[0]
                    else:
                        ok = False
                        st = wvi1[wn11][0]
                        ed = st + wvi1[wn11][1]
                        wv_str = single_wvi2str([st, ed], nlu_t1)
                        for col in range(l_hs1):
                            if engine.check_wc_wv(now_tb['id'], col, wv_str):
                                wc[wn11] = col
                                ok = True
                                break
                        if not ok:
                            wc[wn11] = pr_wc1[0]
                else:
                    if wo1[wn11] <= 1:
                        for pr_wc11 in pr_wc1:
                            if pr_wc11 >= l_hs1:
                                continue
                            if pr_wc11 not in wc[:wn11] and get_col_type(now_tb, pr_wc11) == 'real':
                                wc[wn11] = pr_wc11
                                break
                    elif wo1[wn11] == 3:
                        for pr_wc11 in pr_wc1:
                            if pr_wc11 >= l_hs1:
                                continue
                            if pr_wc11 not in wc[:wn11]:
                                wc[wn11] = pr_wc11                
                                break
                    else:
                        ok = False
                        st = wvi1[wn11][0]
                        ed = st + wvi1[wn11][1]
                        wv_str = single_wvi2str([st, ed], nlu_t1)
                        for col in range(l_hs1):
                            if col not in wc[:wn11] and engine.check_wc_wv(now_tb['id'], col, wv_str):
                                wc[wn11] = col
                                ok = True
                                break
                        if not ok:
                            for pr_wc11 in pr_wc1:
                                if pr_wc11 >= l_hs1:
                                    continue
                                if pr_wc11 not in wc[:wn11]:
                                    wc[wn11] = pr_wc11                
                                    break
        
        for i in range(len(wc)):
            if wc[i] == -1:
                wc[i] = pr_wc1[0]
        #print(wc)
        pr_wc.append(wc)
    return pr_wc

def pred_wc(dwn, s_wc, wn, pr_hrpc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    ! Returned index is sorted!
    """
    # get g_num
    s_wc = torch.sigmoid(s_wc)
    pr_wc = []
    for b, wn1 in enumerate(dwn):
        s_wc1 = s_wc[b]
        
        #print('batch id: ', b, '; swc1: ', s_wc1, '; pr_hrpc: ', pr_hrpc[b])

        pr_wc1 = argsort(-s_wc1.data.cpu().numpy())[:wn1].tolist()
        
        #print('A: pr_wc1: ', pr_wc1)
        if pr_hrpc[b]:
            pr_wc1 = pr_wc1 * wn[b]
        else:
            pr_wc1.sort()#sort 可能是导致错位的罪魁祸首
        
        #print('B: pr_wc1: ', pr_wc1)
        
        pr_wc.append(pr_wc1)
    return pr_wc

def pred_wc_without_sort(dwn, s_wc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    ! Returned index is sorted!
    """
    # get g_num
    s_wc = torch.sigmoid(s_wc)
    pr_wc = []
    for b, wn1 in enumerate(dwn):
        s_wc1 = s_wc[b]
        
        #print('batch id: ', b, '; swc1: ', s_wc1, '; pr_hrpc: ', pr_hrpc[b])

        pr_wc1 = argsort(-s_wc1.data.cpu().numpy())[:wn1].tolist()
        
        
        #print('B: pr_wc1: ', pr_wc1)
        
        pr_wc.append(pr_wc1)
    return pr_wc

def pred_wc_sorted_by_prob(s_wc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    ! Returned index is sorted by prob.
    All colume-indexes are returned here.
    """
    # get g_num
    bS = len(s_wc)
    pr_wc = []

    for b in range(bS):
        s_wc1 = s_wc[b]
        pr_wc1 = argsort(-s_wc1.data.cpu().numpy())
        pr_wc.append(pr_wc1.tolist())
    return pr_wc

def redirect_rpc(wv_list, tb1, l_hs1, engine):
    if len(wv_list) < 2:
        return -1
    col_rpc = -1
    for col in range(l_hs1):
        cnt = 0
        for wv1 in wv_list:
            if engine.check_wc_wv(tb1['id'], col, wv1):
                cnt += 1
        if cnt >= 2:
            col_rpc = col
            break
    return col_rpc

def greedy_wvi_normal(l_hs1, tb1, engine, nlu_t1, mvl, skip_dict, prob_list_h):
    result_list = []
    for col in prob_list_h:
        if col >= l_hs1:
            continue
        for st in range(0, len(nlu_t1)):
            for ed in range(st, min(st + mvl, len(nlu_t1))):
                wv_str = single_wvi2str([st, ed], nlu_t1)
                if col in skip_dict and len(skip_dict[col]) != 0:
                    need_continue = False
                    for wv_skip in skip_dict[col]:
                        if wv_skip == wv_str:
                            need_continue = True
                            break
                    if need_continue:
                        continue
                if engine.check_wc_wv(tb1['id'], col, wv_str):#wv 是否存在于 table
                    result_list.append([col, 2, wv_str])
                    skip_dict[col].add(wv_str)
    #clear
    clean_result = []
    for res in result_list:
        if not res[2].isdigit() and len(res[2]) == 1:
            continue
        clean_result.append(res)
    #if clean_result != []:
    #    print(clean_result)
    return clean_result

def greedy_wvi_hrpc(l_hs1, tb1, engine, nlu_t1, mvl, target, prob_list_h):
    col_res = -1
    str_res = []
    wvi_res = []
    cnt_max = 0
    cnt_t = 0
    str_list_t = []
    wvi_list_t = []
    #if '10&ZD109' in nlu_t1:
    #    print('target: ', target)
    
    for stt in range(0, len(nlu_t1)):
        for edd in range(stt, min(stt + mvl, len(nlu_t1))):
            wv_str_t = single_wvi2str([stt, edd], nlu_t1)
            #if '10&ZD109' in nlu_t1:
            #    print(wv_str_t)
            if engine.check_wc_wv(tb1['id'], target, wv_str_t):#wv 是否存在于 table
                #if not wv_str_t.isdigit() and len(wv_str_t) == 1:
                #    continue
                #if '10&ZD109' in nlu_t1:
                #    print('inside')
                wvi_list_t.append([stt, edd])
                str_list_t.append(wv_str_t)
    str_list_t = list(set(str_list_t))
    wvi_list_new_t = []
    for cur_str in str_list_t:
        for e in wvi_list_t:
            if single_wvi2str(e, nlu_t1) == cur_str:
                wvi_list_new_t.append(e)
                break
    wvi_list_t = wvi_list_new_t
    #if '10&ZD109' in nlu_t1:
    #    print('str_list_t: ', str_list_t)
    #    print('wvi_list_t: ', wvi_list_t)
    cnt_t = len(str_list_t)
    if cnt_t > cnt_max:
        cnt_max = cnt_t
        col_res = target
        wvi_res = wvi_list_t
        str_res = str_list_t
    
    for col in prob_list_h:
        if col >= l_hs1:
            continue
        cnt = 0
        str_list = []
        wvi_list = []
        for st in range(0, len(nlu_t1)):
            for ed in range(st, min(st + mvl, len(nlu_t1))):
                wv_str = single_wvi2str([st, ed], nlu_t1)
                #if '10&ZD109' in nlu_t1:
                #    print(wv_str)
                if engine.check_wc_wv(tb1['id'], col, wv_str):#wv 是否存在于 table
                    #if '10&ZD109' == wv_str:
                    #    print('inside')
                    if not wv_str.isdigit() and len(wv_str) == 1:
                        continue
                    wvi_list.append([st, ed])
                    str_list.append(wv_str)
        str_list = list(set(str_list))
        #if '10&ZD109' in nlu_t1:
        #    print('str_list: ', str_list)
        wvi_list_new = []
        for cur_str in str_list:
            for e in wvi_list:
                if single_wvi2str(e, nlu_t1) == cur_str:
                    wvi_list_new.append(e)
                    break
        wvi_list = wvi_list_new
        #if '10&ZD109' in nlu_t1:
        #    print('wvi_list: ', wvi_list)
        cnt = len(str_list)
        if cnt > cnt_max:
            cnt_max = cnt
            col_res = col
            wvi_res = wvi_list
            str_res = str_list
    # still col_res == -1, wn = 0; col != -1 and cnt_max == 1, wn = 1; col != -1 and cnt_max >= 2, wn = cnt_max
    return col_res, cnt_max, str_res, wvi_res


def pred_wo(wn, s_wo):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # s_wo = [B, 4, n_op] 4代表了where col的最大个数为4
    pr_wo_a = s_wo.argmax(dim=2)  # [B, 4]
    # get g_num
    pr_wo = []
    for b, pr_wo_a1 in enumerate(pr_wo_a):
        wn1 = wn[b]
        pr_wo.append(pr_wo_a1.data.cpu().numpy()[:wn1].tolist())

    return pr_wo

def pred_wvi1(wn, s_wvi1):
    pr_wvi1 = []
    for b, s_wvi11 in enumerate(s_wvi1):
        if wn[b]:
            pr_wvi1.append([e.argmax().item() for e in s_wvi11[:wn[b]]])
        else:
            pr_wvi1.append([])
        
    return pr_wvi1

def pred_wvi1_hrpc(wn, s_wvi1, hrpc, wr, l_hs):
    pr_wvi1 = []
    for b, s_wvi11 in enumerate(s_wvi1):
        if wn[b]:
            if hrpc[b] and wr[b] == 2:
                sub_list = []
                for i in range(wn[b]):
                    if i == 0:
                        sub_list.append(s_wvi11[0].argmax().item())
                    else:
                        prob_list = argsort(-s_wvi11[i].data.cpu().numpy())
                        ok = False
                        for e in prob_list:
                            if e >= l_hs[b]:
                                continue
                            if e not in sub_list[:i]:
                                sub_list.append(e)
                                
                                ok = True
                                break
                            print(e)
                        if not ok:
                            sub_list.append(s_wvi11[i].argmax().item())
                pr_wvi1.append(sub_list)
            else:
                pr_wvi1.append([e.argmax().item() for e in s_wvi11[:wn[b]]])
        else:
            pr_wvi1.append([])
            
    return pr_wvi1

def guide_pred_wvi1(wn, wc, s_wvi1, tb, nlu_t):
    pr_wvi1 = []
    for b, s_wvi11 in enumerate(s_wvi1):
        now_tb = tb[b]
        now_nlu_t = nlu_t[b]
        if wn[b]:
            sub_result = [-1 for _ in range(wn[b])]
            for wn11 in range(wn[b]):
                wc11 = wc[b][wn11]
                pr_wvi111 = argsort(-s_wvi11[wn11].data.cpu().numpy()).tolist()
                if get_col_type(now_tb, wc11) == 'real':
                    for houxuan_wvi in pr_wvi111:
                        wv_str = single_wvi2str([houxuan_wvi, houxuan_wvi], now_nlu_t)
                        if check_is_digits_without_head(wv_str):
                            sub_result[wn11] = houxuan_wvi
                            break
                else:
                    sub_result[wn11] = pr_wvi111[0]
                if sub_result[wn11] == -1:
                    sub_result[wn11] = pr_wvi111[0]
            pr_wvi1.append(sub_result)
        else:
            pr_wvi1.append([])
            
    return pr_wvi1

def same_wo_hrpc_pred_wvi1(wn, s_wvi1):
    pr_wvi1 = []
    for b, s_wvi11 in enumerate(s_wvi1):
        if wn[b]:
            myDict = defaultdict(int)
            for wn1 in range(wn[b]):
                k = argsort(-s_wvi11[wn1].data.cpu().numpy()).tolist()
                k = k[:wn[b]]
                for ik in k:
                    myDict[ik] += 1
            pairs = [[key, myDict[key]] for key in myDict]
            pairs.sort(key=lambda x: x[1], reverse=True)
            pairs = pairs[:wn[b]]
            single = [pair[0] for pair in pairs]
            single.sort()
            pr_wvi1.append(single)
        else:
            pr_wvi1.append([])
    return pr_wvi1

def full_matrix(m, max_wn=4):
    for b in range(len(m)):
        m[b] = m[b] + [0] * (max_wn - len(m[b]))
    r = torch.tensor(m)
    return r

def pred_wvi_se(wn, s_wv1, s_wv2, s_wv3, s_wv4, mvl):
    """
    s_wv: [B, 4, mL, 2]
    - predict best st-idx & ed-idx
    """
    p1 = F.softmax(s_wv1, dim=2)
    p3 = F.softmax(s_wv3, dim=2)
    maxwv1, _ = p1.max(dim=2)
    meanwv1 = p1.mean(dim=2)
    maxwv3, _ = p3.max(dim=2)
    meanwv3 = p3.mean(dim=2)
    #s_wv_st = s_wv_st.squeeze(3) # [B, 4, mL, 1] -> [B, 4, mL]
    #s_wv_ed = s_wv_ed.squeeze(3)
    pr_wv1 = full_matrix(pred_wvi1(wn, s_wv1)).to(device)
    pr_wv3 = full_matrix(pred_wvi1(wn, s_wv3)).to(device)
    pr_wvi_st_idx_s = pr_wv1 # [B, 4]
    pr_wvi_len_idx_s = s_wv2.argmax(dim=2)
    pr_wvi_len_idx_e = mvl - 1 - s_wv4.argmax(dim=2)
    pr_wvi_st_idx_e = pr_wv3 - pr_wvi_len_idx_e
    pr_wvi = []
    for b, wn1 in enumerate(wn):
        pr_wvi1 = []
        for i_wn in range(wn1):
            if maxwv1[b][i_wn].item() - meanwv1[b][i_wn].item() >= maxwv3[b][i_wn].item() - meanwv3[b][i_wn].item():
                #print('select wv12')
                pr_wvi_st_idx11 = pr_wvi_st_idx_s[b][i_wn]
                pr_wvi_len_idx11 = pr_wvi_len_idx_s[b][i_wn]
            else:
                #print('select wv34')
                pr_wvi_st_idx11 = pr_wvi_st_idx_e[b][i_wn]
                pr_wvi_len_idx11 = pr_wvi_len_idx_e[b][i_wn]
            pr_wvi1.append([pr_wvi_st_idx11.item(), pr_wvi_len_idx11.item()])
        pr_wvi.append(pr_wvi1)

    return pr_wvi

def length2end(pr_wvi):
    for ib in range(len(pr_wvi)):
        for wn in range(len(pr_wvi[0])):
            pr_wvi[ib][wn][1] += pr_wvi[ib][wn][0]
    return pr_wvi

def pred_wvi_se_beam(max_wn, s_wv1, s_wv3, beam_size, mvl, l_token):
    """
    s_wv: [B, 4, mL]
    - predict best st-idx & ed-idx


    output:
    pr_wvi_beam = [B, max_wn, n_pairs, 2]. 2 means [st, ed].
    prob_wvi_beam = [B, max_wn, n_pairs]
    """
    bS = s_wv1.shape[0]
    
    p1 = F.softmax(s_wv1, dim=2).detach().to('cpu').numpy()
    p3 = F.softmax(s_wv3, dim=2).detach().to('cpu').numpy()

    k_logit = int(ceil(sqrt(beam_size)))
    n_pairs = k_logit**2
    assert n_pairs >= beam_size
    range_list = []
    for ib in range(p1.shape[0]):
        range_list1 = []
        for n in range(p1.shape[1]):
            min_st = min(argsort(-p1[ib][n])[:n_pairs].tolist())
            max_ed = max(argsort(-p3[ib][n])[:n_pairs].tolist())
            #range_list1.append([min_st, max(max_ed, min_st + (mvl - 1))])#有可能要改为+mvl
            range_list1.append([0, l_token[ib] - 1])
        range_list.append(range_list1)

    return range_list#[b, wn, 2]

def check_conds_wv_last_not_same(conds, new_wv, nlu_t1):#注意这里是wv而不是wvi
    pre_wvi_list = [conds1[2].strip() for conds1 in conds]
    return new_wv not in pre_wvi_list

def check_is_digits_without_head(wv_string):
    wv_string = wv_string.strip()
    if len(wv_string) == 0:
        return False
    wv_string_list = wv_string.strip().split('.')
    if any([e == '' for e in wv_string_list]):
        return False
    if wv_string_list[0][0] == '-':
        wv_string_list[0] = wv_string_list[0][1:]
    if any([e == '' for e in wv_string_list]):
        return False
    return True if wv_string == '0' or all([component.isdigit() for component in wv_string_list]) else False

def check_is_digits(wv_string):
    wv_string = wv_string.strip()
    if len(wv_string) == 0:
        return False
    wv_string_list = wv_string.strip().split('.')
    if any([e == '' for e in wv_string_list]):
        return False
    if wv_string_list[0][0] == '-':
        wv_string_list[0] = wv_string_list[0][1:]
    if any([e == '' for e in wv_string_list]):
        return False
    return True if wv_string == '0' or all([component.isdigit() for component in wv_string_list]) and not (len(wv_string_list) == 1 and wv_string_list[0][0] == '0') else False#如果没有小数点的情况下，不能以0开头

def is_whitespace_g_wvi(c):
    # if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
    if c == " ":
        return True
    return False

def convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_wp_t, wp_to_wh_index):
    """
    - Convert to the string in whilte-space-separated tokens
    - Add-hoc addition.
    """
    pr_wv_str_wp = [] # word-piece version
    pr_wv_str = []
    for b, pr_wvi1 in enumerate(pr_wvi):
        #pr_wv_str_wp1 = []
        pr_wv_str1 = []
        #wp_to_wh_index1 = wp_to_wh_index[b]
        #nlu_wp_t1 = nlu_wp_t[b]
        nlu_t1 = nlu_t[b]

        for i_wn, pr_wvi11 in enumerate(pr_wvi1):
            st_idx, ed_idx = pr_wvi11

            # Ad-hoc modification of ed_idx to deal with wp-tokenization effect.
            # e.g.) to convert "butler cc (" ->"butler cc (ks)" (dev set 1st question).
            #pr_wv_str_wp11 = nlu_wp_t1[st_idx:ed_idx+1]
            #pr_wv_str_wp1.append(pr_wv_str_wp11)#最后一个词只有第一个字

            #st_wh_idx = wp_to_wh_index1[st_idx]
            #ed_wh_idx = wp_to_wh_index1[ed_idx]
            st_wh_idx = st_idx
            ed_wh_idx = ed_idx
            pr_wv_str11 = nlu_t1[st_wh_idx:ed_wh_idx+1]

            pr_wv_str1.append(pr_wv_str11)

        #pr_wv_str_wp.append(pr_wv_str_wp1)
        pr_wv_str.append(pr_wv_str1)

    return pr_wv_str, pr_wv_str_wp

def single_wvi2str(wvi1, nlu_t1):
    #print('wvi1 is: ', wvi1) 这个是st 和 ed
    return (''.join(nlu_t1[wvi1[0] : wvi1[1] + 1])).strip()

def caculate_std(wvi, target):
    return pow(wvi[0] - target[0], 2) + pow(wvi[1] - target[1], 2)

def find_std_min(wvi_list, target_wvi):
    min_std = caculate_std(wvi_list[0], target_wvi)
    min_idx = 0
    for i in range(1, len(wvi_list)):
        cur_std = caculate_std(wvi_list[i], target_wvi)
        if cur_std < min_std:
            min_std = cur_std
            min_idx = i
    return min_idx

def pred_sw_se(s_sn, s_sc, s_sa, s_wn, s_wr, s_hrpc, s_wc, s_wo, s_wv1, s_wv2, s_wv3, s_wv4, mvl):
    pr_sn = pred_sn(s_sn)
    pr_sc = pred_sc(pr_sn, s_sc)
    pr_sa = pred_sa(pr_sn, s_sa)
    pr_wn = pred_wn(s_wn)
    pr_wr = pred_wr(pr_wn, s_wr)
    pr_hrpc = pred_hrpc(s_hrpc)
    pr_dwn = pred_dwn(pr_wn, pr_hrpc)
    pr_wc = pred_wc(pr_dwn, s_wc, pr_wn, pr_hrpc)
    pr_wo = pred_wo(pr_wn, s_wo)
    pr_wvi = pred_wvi_se(pr_wn, s_wv1, s_wv2, s_wv3, s_wv4, mvl)

    return pr_sn, pr_sc, pr_sa, pr_wn, pr_wr, pr_hrpc, pr_wc, pr_wo, pr_wvi





def merge_wv_t1_eng(where_str_tokens, NLq):
    """
    Almost copied of SQLNet.
    The main purpose is pad blank line while combining tokens.
    """
    ret = ''
    for w_token in where_str_tokens:
        ret = ret + w_token

    return ret.strip()



def find_sql_where_op(gt_sql_tokens_part):
    """
    gt_sql_tokens_part: Between 'WHERE' and 'AND'(if exists).
    """
    # sql_where_op = ['=', 'EQL', '<', 'LT', '>', 'GT']
    sql_where_op = ['EQL','LT','GT'] # wv sometimes contains =, < or >.


    for sql_where_op in sql_where_op:
        if sql_where_op in gt_sql_tokens_part:
            found_sql_where_op = sql_where_op
            break

    return found_sql_where_op


def find_sub_list(sl, l):
    # from stack overflow.
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))

    return results

def get_g_wvi_bert(nlu, nlu_t, wh_to_wp_index, sql_i, sql_t, tokenizer, nlu_wp_t):
    """
    Generate SQuAD style start and end index of wv in nlu. Index is for of after WordPiece tokenization.

    Assumption: where_str always presents in the nlu.
    """
    g_wvi = []
    for b, sql_i1 in enumerate(sql_i):
        nlu1 = nlu[b]
        nlu_t1 = nlu_t[b]
        nlu_wp_t1 = nlu_wp_t[b]
        sql_t1 = sql_t[b]
        wh_to_wp_index1 = wh_to_wp_index[b]

        st = sql_t1.index('WHERE') + 1 if 'WHERE' in sql_t1 else len(sql_t1)
        g_wvi1 = []
        while st < len(sql_t1):
            if 'AND' not in sql_t1[st:]:
                ed = len(sql_t1)
            else:
                ed = sql_t1[st:].index('AND') + st
            sql_wop = find_sql_where_op(sql_t1[st:ed])  # sql where operator
            st_wop = st + sql_t1[st:ed].index(sql_wop)

            wv_str11_t = sql_t1[st_wop + 1:ed]
            results = find_sub_list(wv_str11_t, nlu_t1)
            st_idx, ed_idx = results[0]

            st_wp_idx = wh_to_wp_index1[st_idx]
            ed_wp_idx = wh_to_wp_index1[ed_idx]


            g_wvi11 = [st_wp_idx, ed_wp_idx]
            g_wvi1.append(g_wvi11)
            st = ed + 1
        g_wvi.append(g_wvi1)

    return g_wvi


def get_g_wvi_stidx_length_jian_yi(g_wvi_corenlp):
    return [[[e[0], e[1] - e[0]] for e in l] for l in g_wvi_corenlp]

def g_wvi_decoder_stidx_length_jian_yi(wvi_jian_yi):
    return [[[e[0], e[0] + e[1]] for e in l] for l in wvi_jian_yi]

def get_g_wvi_bert_from_g_wvi_corenlp(wh_to_wp_index, g_wvi_corenlp):
    """
    Generate SQuAD style start and end index of wv in nlu. Index is for of after WordPiece tokenization.

    Assumption: where_str always presents in the nlu.
    """
    g_wvi = []
    for b, g_wvi_corenlp1 in enumerate(g_wvi_corenlp):
        wh_to_wp_index1 = wh_to_wp_index[b]
        g_wvi1 = []
        for i_wn, g_wvi_corenlp11 in enumerate(g_wvi_corenlp1):

            st_idx, ed_idx = g_wvi_corenlp11

            st_wp_idx = wh_to_wp_index1[st_idx]#convert start with 第几个token 到 end with 第几个token to 具体的start and end index of smallest tokens
            ed_wp_idx = wh_to_wp_index1[ed_idx]

            g_wvi11 = [st_wp_idx, ed_wp_idx]
            g_wvi1.append(g_wvi11)

        g_wvi.append(g_wvi1)

    return g_wvi


def get_g_wvi_bert_from_sql_i(nlu, nlu_t, wh_to_wp_index, sql_i, sql_t, tokenizer, nlu_wp_t):
    """
    Generate SQuAD style start and end index of wv in nlu. Index is for of after WordPiece tokenization.

    Assumption: where_str always presents in the nlu.
    """
    g_wvi = []
    for b, sql_i1 in enumerate(sql_i):
        nlu1 = nlu[b]
        nlu_t1 = nlu_t[b]
        nlu_wp_t1 = nlu_wp_t[b]
        sql_t1 = sql_t[b]
        wh_to_wp_index1 = wh_to_wp_index[b]

        st = sql_t1.index('WHERE') + 1 if 'WHERE' in sql_t1 else len(sql_t1)
        g_wvi1 = []
        while st < len(sql_t1):
            if 'AND' not in sql_t1[st:]:
                ed = len(sql_t1)
            else:
                ed = sql_t1[st:].index('AND') + st
            sql_wop = find_sql_where_op(sql_t1[st:ed])  # sql where operator
            st_wop = st + sql_t1[st:ed].index(sql_wop)

            wv_str11_t = sql_t1[st_wop + 1:ed]
            results = find_sub_list(wv_str11_t, nlu_t1)
            st_idx, ed_idx = results[0]

            st_wp_idx = wh_to_wp_index1[st_idx]
            ed_wp_idx = wh_to_wp_index1[ed_idx]


            g_wvi11 = [st_wp_idx, ed_wp_idx]
            g_wvi1.append(g_wvi11)
            st = ed + 1
        g_wvi.append(g_wvi1)

    return g_wvi

def get_cnt_sc(g_sc, pr_sc):
    cnt = 0
    for b, g_sc1 in enumerate(g_sc):
        pr_sc1 = pr_sc[b]
        if pr_sc1 == g_sc1:
            cnt += 1

    return cnt

def get_cnt_sc_list(g_sc, pr_sc):
    cnt_list = []
    for b, g_sc1 in enumerate(g_sc):
        pr_sc1 = pr_sc[b]
        if pr_sc1 == g_sc1:
            cnt_list.append(1)
        else:
            cnt_list.append(0)

    return cnt_list

def get_cnt_sa(g_sa, pr_sa):
    cnt = 0
    for b, g_sa1 in enumerate(g_sa):
        pr_sa1 = pr_sa[b]
        if pr_sa1 == g_sa1:
            cnt += 1

    return cnt


def get_cnt_wn(g_wn, pr_wn):
    cnt = 0
    for b, g_wn1 in enumerate(g_wn):
        pr_wn1 = pr_wn[b]
        if pr_wn1 == g_wn1:
            cnt += 1

    return cnt

def get_cnt_wc(g_wc, pr_wc):
    cnt = 0
    for b, g_wc1 in enumerate(g_wc):

        pr_wc1 = pr_wc[b]
        pr_wn1 = len(pr_wc1)
        g_wn1 = len(g_wc1)

        if pr_wn1 != g_wn1:
            continue
        else:
            wc1 = array(g_wc1)
            wc1.sort()
            wc2 = array(pr_wc1)
            wc2.sort()

            if array_equal(wc2, wc1):
                cnt += 1

    return cnt

def get_cnt_wc_list(g_wc, pr_wc):
    cnt_list= []
    for b, g_wc1 in enumerate(g_wc):

        pr_wc1 = pr_wc[b]
        pr_wn1 = len(pr_wc1)
        g_wn1 = len(g_wc1)

        if pr_wn1 != g_wn1:
            cnt_list.append(0)
            continue
        else:
            wc1 = array(g_wc1)
            wc1.sort()
            wc2 = array(pr_wc1)
            wc2.sort()

            if array_equal(wc2, wc1):
                cnt_list.append(1)
            else:
                cnt_list.append(0)

    return cnt_list


def get_cnt_wo(g_wn, g_wc, g_wo, pr_wc, pr_wo, mode):
    """ pr's are all sorted as pr_wc are sorted in increasing order (in column idx)
        However, g's are not sorted.

        Sort g's in increasing order (in column idx)
    """
    cnt = 0
    for b, g_wo1 in enumerate(g_wo):
        g_wc1 = g_wc[b]
        pr_wc1 = pr_wc[b]
        pr_wo1 = pr_wo[b]
        pr_wn1 = len(pr_wo1)
        g_wn1 = g_wn[b]

        if g_wn1 != pr_wn1:
            continue
        else:
            # Sort based on wc sequence.
            if mode == 'test':
                idx = argsort(array(g_wc1))
                g_wo1_s = array(g_wo1)[idx].tolist()
            elif mode == 'train':
                # due to teacher forcing, no need to sort.
                g_wo1_s = g_wo1
            else:
                raise ValueError

            if type(pr_wo1) != list:
                raise TypeError
            if g_wo1_s == pr_wo1:
                cnt += 1
    return cnt

def get_cnt_wo_list(g_wn, g_wc, g_wo, pr_wc, pr_wo, mode):
    """ pr's are all sorted as pr_wc are sorted in increasing order (in column idx)
        However, g's are not sorted.

        Sort g's in increasing order (in column idx)
    """
    cnt_list=[]
    for b, g_wo1 in enumerate(g_wo):
        g_wc1 = g_wc[b]
        pr_wc1 = pr_wc[b]
        pr_wo1 = pr_wo[b]
        pr_wn1 = len(pr_wo1)
        g_wn1 = g_wn[b]

        if g_wn1 != pr_wn1:
            cnt_list.append(0)
            continue
        else:
            # Sort based wc sequence.
            if mode == 'test':
                idx = argsort(array(g_wc1))
                g_wo1_s = array(g_wo1)[idx].tolist()
            elif mode == 'train':
                # due to tearch forcing, no need to sort.
                g_wo1_s = g_wo1
            else:
                raise ValueError

            if type(pr_wo1) != list:
                raise TypeError
            if g_wo1_s == pr_wo1:
                cnt_list.append(1)
            else:
                cnt_list.append(0)
    return cnt_list


def get_cnt_wv(g_wn, g_wc, g_wvi, pr_wvi, mode):
    """ usalbe only when g_wc was used to find pr_wv

    g_wvi
    """
    cnt = 0
    for b, g_wvi1 in enumerate(g_wvi):
        pr_wvi1 = pr_wvi[b]
        g_wc1 = g_wc[b]
        pr_wn1 = len(pr_wvi1)
        g_wn1 = g_wn[b]

        # Now sorting.
        # Sort based wc sequence.
        if mode == 'test':
            idx1 = argsort(array(g_wc1))
            #idx1 = list( range( g_wn1) )
        elif mode == 'train':
            idx1 = list( range( g_wn1) )
        else:
            raise ValueError

        if g_wn1 != pr_wn1:
            continue
        else:
            flag = True
            for i_wn, idx11 in enumerate(idx1):
                g_wvi11 = g_wvi1[idx11]
                pr_wvi11 = pr_wvi1[i_wn]
                if g_wvi11 != pr_wvi11:
                    flag = False
                    # print(g_wv1, g_wv11)
                    # print(pr_wv1, pr_wv11)
                    # input('')
                    break
            if flag:
                cnt += 1

    return cnt


def get_cnt_wvi_list(g_wn, g_wc, g_wvi, pr_wvi, mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_list =[]
    for b, g_wvi1 in enumerate(g_wvi):
        g_wc1 = g_wc[b]
        pr_wvi1 = pr_wvi[b]
        pr_wn1 = len(pr_wvi1)
        g_wn1 = g_wn[b]

        # Now sorting.
        # Sort based wc sequence.
        if mode == 'test':
            #idx1 = list( range( g_wn1) )
            idx1 = argsort(array(g_wc1))
        elif mode == 'train':
            idx1 = list( range( g_wn1) )
        else:
            raise ValueError

        if g_wn1 != pr_wn1:
            cnt_list.append(0)
            continue
        else:
            flag = True
            for i_wn, idx11 in enumerate(idx1):
                g_wvi11 = g_wvi1[idx11]
                pr_wvi11 = pr_wvi1[i_wn]
                if g_wvi11 != pr_wvi11:
                    flag = False
                    # print(g_wv1, g_wv11)
                    # print(pr_wv1, pr_wv11)
                    # input('')
                    break
            if flag:
                cnt_list.append(1)
            else:
                cnt_list.append(0)

    return cnt_list




def get_cnt_wv_list(g_wn, g_wc, g_sql_i, pr_sql_i, g_wvi, pr_wvi, mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_list =[]
    for b, g_wc1 in enumerate(g_wc):
        pr_wn1 = len(pr_sql_i[b]["conds"])
        g_wn1 = g_wn[b]

        # Now sorting.
        # Sort based wc sequence.
        if mode == 'test':
            idx1 = argsort(array(g_wc1))
            #idx1 = list( range( g_wn1) )
        elif mode == 'train':
            idx1 = list( range( g_wn1) )
        else:
            raise ValueError

        if g_wn1 != pr_wn1:
            cnt_list.append(0)
            continue
        else:
            flag = True
            for i_wn, idx11 in enumerate(idx1):
                g_wvi_str11 = str(g_sql_i[b]["conds"][idx11][2])
                pr_wvi_str11 = str(pr_sql_i[b]["conds"][i_wn][2])
                # print(g_wvi_str11)
                # print(pr_wvi_str11)
                # print(g_wvi_str11==pr_wvi_str11)
                if g_wvi_str11 != pr_wvi_str11:
                    flag = False
                    '''
                    if g_wvi[b][idx11] == pr_wvi[b][i_wn]:#当wvi正确的情况下，wv还是错误的例子
                        print('----------------diff----------------')
                        print('g_wvi[b]: ', g_wvi[b])
                        print('pr_wvi[b]: ', pr_wvi[b])
                        print('g_conds: ', str(g_sql_i[b]['conds']))
                        print('pr_conds: ', str(pr_sql_i[b]['conds']))
                        print('g_wvi_str11: ', g_wvi_str11)
                        print('pr_wvi_str11: ', pr_wvi_str11)
                        print('----------------diff----------------')
                    '''
                    # input('')
                    break
            if flag:
                cnt_list.append(1)
            else:
                cnt_list.append(0)

    return cnt_list

def get_cnt_sw(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_sc = get_cnt_sc(g_sc, pr_sc)
    cnt_sa = get_cnt_sa(g_sa, pr_sa)
    cnt_wn = get_cnt_wn(g_wn, pr_wn)
    cnt_wc = get_cnt_wc(g_wc, pr_wc)
    cnt_wo = get_cnt_wo(g_wn, g_wc, g_wo, pr_wc, pr_wo, mode)
    cnt_wv = get_cnt_wv(g_wn, g_wc, g_wvi, pr_wvi, mode)

    return cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv

def get_cnt_sw_list(g_sn, g_sc, g_sa, g_wn, g_wr, g_wc, g_wo, g_wvi,
                    pr_sn, pr_sc, pr_sa, pr_wn, pr_wr, pr_wc, pr_wo, pr_wvi,
                    g_sql_i, pr_sql_i,
                    mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_sn = get_cnt_sc_list(g_sn, pr_sn)
    cnt_sc = get_cnt_wc_list(g_sc, pr_sc)
    cnt_sa = get_cnt_wc_list(g_sa, pr_sa)
    cnt_wn = get_cnt_sc_list(g_wn, pr_wn)
    cnt_wr = get_cnt_sc_list(g_wr, pr_wr)
    cnt_wc = get_cnt_wc_list(g_wc, pr_wc)
    cnt_wo = get_cnt_wo_list(g_wn, g_wc, g_wo, pr_wc, pr_wo, mode)
    if pr_wvi:
        cnt_wvi = get_cnt_wvi_list(g_wn, g_wc, g_wvi, pr_wvi, mode)
    else:
        cnt_wvi = [0]*len(cnt_sc)
    cnt_wv = get_cnt_wv_list(g_wn, g_wc, g_sql_i, pr_sql_i, g_wvi, pr_wvi, mode) # compare using wv-str which presented in original data.


    return cnt_sn, cnt_sc, cnt_sa, cnt_wn, cnt_wr, cnt_wc, cnt_wo, cnt_wvi, cnt_wv


def get_cnt_lx_list(cnt_sn1, cnt_sc1, cnt_sa1, cnt_wn1, cnt_wr1, cnt_wc1, cnt_wo1, cnt_wv1):
    # all cnt are list here.
    cnt_list = []
    for csn, csc, csa, cwn, cwr, cwc, cwo, cwv in zip(cnt_sn1, cnt_sc1, cnt_sa1, cnt_wn1, cnt_wr1, cnt_wc1, cnt_wo1, cnt_wv1):
        if csn and csc and csa and cwn and cwr and cwc and cwo and cwv:
            cnt_list.append(1)
        else:
            cnt_list.append(0)

    return cnt_list


def get_cnt_x_list(engine, tb, g_sc, g_sa, g_sql_i, pr_sc, pr_sa, pr_sql_i):
    cnt_x1_list = []
    g_ans = []
    pr_ans = []
    for b in range(len(g_sc)):
        g_ans1 = engine.execute(tb[b]['id'], g_sc[b], g_sa[b], g_sql_i[b]['conds'], g_sql_i[b]['cond_conn_op'])
        # print(f'cnt: {cnt}')
        # print(f"pr_sql_i: {pr_sql_i[b]['conds']}")
        try:
            pr_ans1 = engine.execute(tb[b]['id'], pr_sc[b], pr_sa[b], pr_sql_i[b]['conds'], pr_sql_i[b]['cond_conn_op'])

            if bool(pr_ans1):  # not empty due to lack of the data from incorretly generated sql
                if g_ans1 == pr_ans1:
                    cnt_x1 = 1
                else:
                    cnt_x1 = 0
            else:
                cnt_x1 = 0
        except:
            # type error etc... Execution-guided decoding may be used here.
            pr_ans1 = None
            cnt_x1 = 0
        cnt_x1_list.append(cnt_x1)
        g_ans.append(g_ans1)
        pr_ans.append(pr_ans1)

    return cnt_x1_list, g_ans, pr_ans

def get_mean_grad(named_parameters):
    """
    Get list of mean, std of grad of each parameters
    Code based on web searched result..
    """
    mu_list = []
    sig_list = []
    for name, param in named_parameters:
        if param.requires_grad: # and ("bias" not in name) :
            # bias makes std = nan as it is of single parameters
            magnitude = param.grad.abs()
            mu_list.append(magnitude.mean())
            if len(magnitude) == 1:
                # why nan for single param? Anyway to avoid that..
                sig_list.append(torch.tensor(0))
            else:
                sig_list.append(magnitude.std())

            # if "svp_se"

    return mu_list, sig_list


def generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wr, pr_wc, pr_wo, pr_wv_str, nlu):
    pr_sql_i = []
    for b, nlu1 in enumerate(nlu):
        conds = []
        #print('pr_wn: index', b, '; value: ', pr_wn[b], '; pr_wc: ', pr_wc[b])
        for i_wn in range(pr_wn[b]):
            conds1 = []
            conds1.append(pr_wc[b][i_wn])
            conds1.append(pr_wo[b][i_wn])
            merged_wv11 = merge_wv_t1_eng(pr_wv_str[b][i_wn], nlu[b])
            conds1.append(merged_wv11)
            conds.append(conds1)

        pr_sql_i1 = {'sel': pr_sc[b], 'agg': pr_sa[b], 'cond_conn_op': pr_wr[b], 'conds': conds}
        pr_sql_i.append(pr_sql_i1)
    return pr_sql_i


def save_for_evaluation(path_save, results, dset_name):
    path_save_file = os.path.join(path_save, f'results_{dset_name}.json')
    with open(path_save_file, 'w', encoding='utf-8') as f:
        for i, r1 in enumerate(results):
            json_str = json.dumps(r1, ensure_ascii=False, default=json_default_type_checker)
            json_str += '\n'

            f.writelines(json_str)

def save_for_evaluation_aux(path_save, results, dset_name):
    path_save_file = os.path.join(path_save, f'results_aux_{dset_name}.json')
    with open(path_save_file, 'w', encoding='utf-8') as f:
        for i, r1 in enumerate(results):
            json_str = json.dumps(r1, ensure_ascii=False, default=json_default_type_checker)
            json_str += '\n'

            f.writelines(json_str)

def get_next_large_index(arr_origin, cur):
    arr = list(arr_origin) if type(arr_origin) != type([]) else arr_origin
    if min(arr) != arr[cur]:
        prob_list = []
        value_list = []
        for i, e in enumerate(arr):
            if e < arr[cur]:
                prob_list.append(i)
                value_list.append(e)
        min_value = min(value_list)
        for i, e in enumerate(value_list):
            if e == min_value:
                return prob_list[i]
    return None

def get_col_type(tb, col):
    return tb['types'][col]

def get_max_sca_correct(tb, col, sa_prob):
    col_type = tb['types'][col]
    idx = sa_prob.argmax().item()
    if col_type == 'text':
        while idx != 0 and idx != 4:#只能是空或者count
            idx = get_next_large_index(sa_prob, idx)#一定有结果
    return idx

def get_max_wco_correct(tb, col, wo_prob):
    col_type = tb['types'][col]
    idx = wo_prob.argmax().item()
    if col_type == 'text':
        while idx != 2 and idx != 3:#只能是等于或者不等于
            idx = get_next_large_index(wo_prob, idx)
    return idx

def check_sc_sa_pairs(tb, pr_sc, pr_sa):
    """
    Check whether pr_sc, pr_sa are allowed pairs or not.
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']

    """
    bS = len(pr_sc)
    sn = len(pr_sc[0])
    check = [[False] * sn] * bS
    for b, pr_sc1 in enumerate(pr_sc):
        for n, pr_sc11 in enumerate(pr_sc1):
            pr_sa1 = pr_sa[b][n]
            hd_types1 = tb[b]['types']
            hd_types11 = hd_types1[pr_sc11]
            if hd_types11 == 'text':
                if pr_sa1 == 0 or pr_sa1 == 4: # ''和COUNT
                    check[b][n] = True
                else:
                    check[b][n] = False

            elif hd_types11 == 'real':
                check[b][n] = True
            else:
                raise Exception("New TYPE!!")
                
    return check


def remap_sc_idx(idxs, pr_sc_beam):
    for b, idxs1 in enumerate(idxs):
        for i_beam, idxs11 in enumerate(idxs1):
            sc_beam_idx = idxs[b][i_beam][0]
            sc_idx = pr_sc_beam[b][sc_beam_idx]
            idxs[b][i_beam][0] = sc_idx

    return idxs

def generate_pr(pr_sql_i):
    pr_sn = []#
    pr_sc = []#
    pr_sa = []#
    pr_wn = []#
    pr_wr = []#
    pr_wc = []#
    pr_wo = []#
    pr_wv = []#
    
    for b, pr_sql_i1 in enumerate(pr_sql_i):
        rela = pr_sql_i1['cond_conn_op']
        sel = pr_sql_i1['sel']
        agg = pr_sql_i1['agg']
        conds = pr_sql_i1["conds"]
        pr_wr.append(rela)
        pr_sn.append(len(sel))
        pr_wn.append(len(conds))
        pr_sc.append(sel)
        pr_sa.append(agg)
        pr_wc.append([cond[0] for cond in conds])
        pr_wo.append([cond[1] for cond in conds])
        pr_wv.append([cond[2] for cond in conds])
        
    return pr_sn, pr_sc, pr_sa, pr_wn, pr_wr, pr_wc, pr_wo, pr_wv

def sort_and_generate_pr_w(pr_sql_i):
    pr_wc = []
    pr_wo = []
    pr_wv = []
    for b, pr_sql_i1 in enumerate(pr_sql_i):
        conds1 = pr_sql_i1["conds"]
        pr_wc1 = []
        pr_wo1 = []
        pr_wv1 = []

        # Generate
        for i_wn, conds11 in enumerate(conds1):
            pr_wc1.append( conds11[0])
            pr_wo1.append( conds11[1])
            pr_wv1.append( conds11[2])

        # sort based on pr_wc1
        idx = argsort(pr_wc1)
        pr_wc1 = array(pr_wc1)[idx].tolist()
        pr_wo1 = array(pr_wo1)[idx].tolist()
        pr_wv1 = array(pr_wv1)[idx].tolist()

        conds1_sorted = []
        for i, idx1 in enumerate(idx):
            conds1_sorted.append( conds1[idx1] )


        pr_wc.append(pr_wc1)
        pr_wo.append(pr_wo1)
        pr_wv.append(pr_wv1)

        pr_sql_i1['conds'] = conds1_sorted

    return pr_wc, pr_wo, pr_wv, pr_sql_i

def generate_sql_q(sql_i, tb):
    sql_q = []
    for b, sql_i1 in enumerate(sql_i):
        tb1 = tb[b]
        sql_q1 = generate_sql_q1(sql_i1, tb1)
        sql_q.append(sql_q1)

    return sql_q

def generate_sql_q1(sql_i1, tb1):
    """
        sql = {'sel': [5], 'agg': [4], 'conds': [[3, 0, '59']], 'cond_conn_op': 0}
        agg_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM"}
        cond_op_dict = {0:">", 1:"<", 2:"=", 3:"!="}
        cond_rela_dict = {0:"",1:"and",2:"or"}

        Temporal as it can show only one-time conditioned case.
        sql_query: real sql_query
        sql_plus_query: More redable sql_query

        "PLUS" indicates, it deals with the some of db specific facts like PCODE <-> NAME
    """
    agg_ops = ['', 'AVG', 'MAX', 'MIN', 'COUNT', 'SUM']
    cond_ops = ['>', '<', '=', '!=']
    cond_rps = ['', 'AND', 'OR']

    headers = tb1["header"]
    # select_header = headers[sql['sel']].lower()
    # try:
    #     select_table = tb1["name"]
    # except:
    #     print(f"No table name while headers are {headers}")
    select_table = tb1["id"]
    
    sql_query_part1 = 'SELECT '
    
    for aggIdx1, headerIdx1 in zip(sql_i1['agg'], sql_i1['sel']):
        if headerIdx1 >= len(headers):
            print('-----------------------------------------------error--------------------------------------------')
            print('hIdx: ', headerIdx1, '; headerLen: ', len(headers))
            print('headers: ', headers)
            print('sql: ', sql_i1)
            print('-----------------------------------------------error--------------------------------------------')
        else:
            sql_query_part1 += agg_ops[aggIdx1]
            sql_query_part1 += '(' + headers[headerIdx1] + '),'
    
    sql_query_part1 = sql_query_part1[:-1] + ' '

    where_num = len(sql_i1['conds'])
    where_rela = cond_rps[sql_i1['cond_conn_op']]
    if where_num == 0:
        sql_query_part2 = f'FROM {select_table}'
        # sql_plus_query_part2 = f'FROM {select_table}'

    else:
        sql_query_part2 = f'FROM {select_table} WHERE'
        # sql_plus_query_part2 = f'FROM {select_table_refined} WHERE'
        # ----------------------------------------------------------------------------------------------------------
        for i in range(where_num):
            # check 'OR'
            # number_of_sub_conds = len(sql['conds'][i])
            where_header_idx, where_op_idx, where_str = sql_i1['conds'][i]
            if where_header_idx >= len(headers):
                print('wherehIdx: ', where_header_idx, '; headerLen: ', len(headers))
            else:
                where_header = headers[where_header_idx]
                where_op = cond_ops[where_op_idx]
                if i > 0:
                    if where_rela == 'OR':
                        sql_query_part2 += ' OR'
                    else:
                        sql_query_part2 += ' AND'
                        # sql_plus_query_part2 += ' AND'

                sql_query_part2 += f" {where_header} {where_op} {where_str}"

    sql_query = sql_query_part1 + sql_query_part2
    # sql_plus_query = sql_plus_query_part1 + sql_plus_query_part2

    return sql_query


def get_pnt_idx1(col_pool_type, st_ed):
    st, ed = st_ed
    if col_pool_type == 'start_tok':
        pnt_idx1 = st
    elif col_pool_type == 'end_tok':
        pnt_idx1 = ed
    elif col_pool_type == 'avg':
        pnt_idx1 = arange(st, ed, 1)
    return pnt_idx1


def gen_g_pnt_idx(g_wvi, sql_i, i_hds, i_sql_vocab, col_pool_type):
    """
    sql_vocab = (
        0.. "sql none", "sql max", "sql min", "sql count", "sql sum", "sql average", ..5
        6.. "sql select", "sql where", "sql and", .. 8
        9.. "sql equal", "sql greater than", "sql less than", .. 11
        12.. "sql start", "sql end" .. 13
    )
    """
    g_pnt_idxs = []



    for b, sql_i1 in enumerate(sql_i):
        i_sql_vocab1 = i_sql_vocab[b]
        i_hds1 = i_hds[b]
        g_pnt_idxs1 = []

        # start token
        pnt_idx1 = get_pnt_idx1(col_pool_type, i_sql_vocab1[-2])
        g_pnt_idxs1.append(pnt_idx1)

        # select token
        pnt_idx1 = get_pnt_idx1(col_pool_type, i_sql_vocab1[6])
        g_pnt_idxs1.append(pnt_idx1)

        # select agg
        idx_agg = sql_i1["agg"]
        pnt_idx1 = get_pnt_idx1(col_pool_type, i_sql_vocab1[idx_agg])
        g_pnt_idxs1.append(pnt_idx1)

        # select column
        idx_sc = sql_i1["sel"]
        pnt_idx1 = get_pnt_idx1(col_pool_type, i_hds1[idx_sc])
        g_pnt_idxs1.append(pnt_idx1)

        conds = sql_i1["conds"]
        wn = len(conds)
        if wn <= 0:
            pass
        else:
            # select where
            pnt_idx1 = get_pnt_idx1(col_pool_type, i_sql_vocab1[7])
            g_pnt_idxs1.append(pnt_idx1)

            for i_wn, conds1 in enumerate(conds):
                # where column
                idx_wc = conds1[0]
                pnt_idx1 = get_pnt_idx1(col_pool_type, i_hds1[idx_wc])
                g_pnt_idxs1.append(pnt_idx1)

                # where op
                idx_wo = conds1[1]
                pnt_idx1 = get_pnt_idx1(col_pool_type, i_sql_vocab1[idx_wo + 9])
                g_pnt_idxs1.append(pnt_idx1)

                # where val
                st, ed = g_wvi[b][i_wn]
                end_pos_of_sql_vocab = i_sql_vocab1[-1][-1]
                g_pnt_idxs1.append(st + 1 + end_pos_of_sql_vocab)  # due to inital [CLS] token in BERT-input vector
                g_pnt_idxs1.append(ed + 1 + end_pos_of_sql_vocab)  # due to inital [CLS] token in BERT-input vector

                # and token
                if i_wn < wn - 1:
                    pnt_idx1 = get_pnt_idx1(col_pool_type, i_sql_vocab1[8])
                    g_pnt_idxs1.append(pnt_idx1)

        # end token
        pnt_idx1 = get_pnt_idx1(col_pool_type, i_sql_vocab1[-1])
        g_pnt_idxs1.append(pnt_idx1)

        g_pnt_idxs.append(g_pnt_idxs1)

    return g_pnt_idxs


def pred_pnt_idxs(score, pnt_start_tok, pnt_end_tok):
    pr_pnt_idxs = []
    for b, score1 in enumerate(score):
        # score1 = [T, max_seq_length]
        pr_pnt_idxs1 = [pnt_start_tok]
        for t, score11 in enumerate(score1):
            pnt = score11.argmax().item()
            pr_pnt_idxs1.append(pnt)

            if pnt == pnt_end_tok:
                break
        pr_pnt_idxs.append(pr_pnt_idxs1)

    return pr_pnt_idxs


def generate_sql_q_s2s(pnt_idxs, tokens, tb):
    sql_q = []
    for b, pnt_idxs1 in enumerate(pnt_idxs):
        tb1 = tb[b]
        sql_q1 = generate_sql_q1_s2s(pnt_idxs1, tokens[b], tb1)
        sql_q.append(sql_q1)

    return sql_q


def generate_sql_q1_s2s(pnt_idxs1, tokens1, tb1):
    """
        agg_ops = ['', 'max', 'min', 'count', 'sum', 'avg']
        cond_ops = ['=', '>', '<', 'OP']

        Temporal as it can show only one-time conditioned case.
        sql_query: real sql_query
        sql_plus_query: More redable sql_query

        "PLUS" indicates, it deals with the some of db specific facts like PCODE <-> NAME
    """
    sql_query = ""
    for t, pnt_idxs11 in enumerate(pnt_idxs1):
        tok = tokens1[pnt_idxs11]
        sql_query += tok
        if t < len(pnt_idxs1)-1:
            sql_query += " "


    return sql_query


# Generate sql_i from pnt_idxs
def find_where_pnt_belong(pnt, vg):
    idx_sub = -1
    for i, st_ed in enumerate(vg):
        st, ed = st_ed
        if pnt < ed and pnt >= st:
            idx_sub = i

    return idx_sub


def gen_pnt_i_from_pnt(pnt, i_sql_vocab1, i_nlu1, i_hds1):
    # Find where it belong
    vg_list = [i_sql_vocab1, [i_nlu1], i_hds1] # as i_nlu has only single st and ed
    i_vg = -1
    i_vg_sub = -1
    for i, vg in enumerate(vg_list):
        idx_sub = find_where_pnt_belong(pnt, vg)
        if idx_sub > -1:
            i_vg = i
            i_vg_sub = idx_sub
            break
    return i_vg, i_vg_sub


def gen_i_vg_from_pnt_idxs(pnt_idxs, i_sql_vocab, i_nlu, i_hds):
    i_vg_list = []
    i_vg_sub_list = []
    for b, pnt_idxs1 in enumerate(pnt_idxs):
        # if properly generated,
        sql_q1_list = []
        i_vg_list1 = [] # index of (sql_vocab, nlu, hds)
        i_vg_sub_list1 = [] # index inside of each vocab group

        for t, pnt in enumerate(pnt_idxs1):
            i_vg, i_vg_sub = gen_pnt_i_from_pnt(pnt, i_sql_vocab[b], i_nlu[b], i_hds[b])
            i_vg_list1.append(i_vg)
            i_vg_sub_list1.append(i_vg_sub)

        # sql_q1 = sql_q1.join(' ')
        # sql_q.append(sql_q1)
        i_vg_list.append(i_vg_list1)
        i_vg_sub_list.append(i_vg_sub_list1)
    return i_vg_list, i_vg_sub_list


def gen_sql_q_from_i_vg(tokens, nlu, nlu_t, hds, tt_to_t_idx, pnt_start_tok, pnt_end_tok, pnt_idxs, i_vg_list, i_vg_sub_list):
    """
    (
        "none", "max", "min", "count", "sum", "average",
        "select", "where", "and",
        "equal", "greater than", "less than",
        "start", "end"
    ),
    """
    sql_q = []
    sql_i = []
    for b, nlu_t1 in enumerate(nlu_t):
        sql_q1_list = []
        sql_i1 = {}
        tt_to_t_idx1 = tt_to_t_idx[b]
        nlu_st_observed = False
        agg_observed = False
        wc_obs = False
        wo_obs = False
        conds = []

        for t, i_vg in enumerate(i_vg_list[b]):
            i_vg_sub = i_vg_sub_list[b][t]
            pnt = pnt_idxs[b][t]
            if i_vg == 0:
                # sql_vocab
                if pnt == pnt_start_tok or pnt == pnt_end_tok:
                    pass
                else:
                    tok = tokens[b][pnt]
                    if tok in ["none", "max", "min", "count", "sum", "average"]:
                        agg_observed = True
                        if tok == "none":
                            pass
                        sql_i1["agg"] = ["none", "max", "min", "count", "sum", "average"].index(tok)
                    else:
                        if tok in ["greater", "less", "equal"]:
                            if tok == 'greater':
                                tok = '>'
                            elif tok == 'less':
                                tok = '<'
                            elif tok == 'equal':
                                tok = '='

                            # gen conds1
                            if wc_obs:
                                conds1.append( ['=','>','<'].index(tok) )
                                wo_obs = True

                        sql_q1_list.append(tok)

            elif i_vg == 1:
                # nlu case
                if not nlu_st_observed:
                    idx_nlu_st = pnt
                    nlu_st_observed = True
                else:
                    # now to wrap up
                    idx_nlu_ed = pnt
                    st_wh_idx = tt_to_t_idx1[idx_nlu_st - pnt_end_tok - 2]
                    ed_wh_idx = tt_to_t_idx1[idx_nlu_ed - pnt_end_tok - 2]
                    pr_wv_str11 = nlu_t1[st_wh_idx:ed_wh_idx + 1]
                    merged_wv11 = merge_wv_t1_eng(pr_wv_str11, nlu[b])
                    sql_q1_list.append(merged_wv11)
                    nlu_st_observed = False

                    if wc_obs and wo_obs:
                        conds1.append(merged_wv11)
                        conds.append(conds1)

                        wc_obs = False
                        wo_obs = False


            elif i_vg == 2:
                # headers
                tok = hds[b][i_vg_sub]
                if agg_observed:
                    sql_q1_list.append(f"({tok})")
                    sql_i1["sel"] = i_vg_sub
                    agg_observed = False
                else:
                    wc_obs = True
                    conds1 = [i_vg_sub]

                    sql_q1_list.append(tok)

        # insert table name between.
        sql_i1["conds"] = conds
        sql_i.append(sql_i1)
        sql_q1 = ' '.join(sql_q1_list)
        sql_q.append(sql_q1)

    return sql_q, sql_i


def get_cnt_lx_list_s2s(g_pnt_idxs, pr_pnt_idxs):
    # all cnt are list here.
    cnt_list = []
    for b, g_pnt_idxs1 in enumerate(g_pnt_idxs):
        pr_pnt_idxs1 = pr_pnt_idxs[b]

        if g_pnt_idxs1 == pr_pnt_idxs1:
            cnt_list.append(1)
        else:
            cnt_list.append(0)

    return cnt_list


def get_wemb_h_FT_Scalar_1(i_hds, l_hs, hS, all_encoder_layer, col_pool_type='start_tok'):
    """
    As if
    [ [table-1-col-1-tok1, t1-c1-t2, ...],
       [t1-c2-t1, t1-c2-t2, ...].
       ...
       [t2-c1-t1, ...,]
    ]

    # i_hds = [ [  Batch 1 ] [  Batch 2  ] ]
    #  [Batch 1] = [ (col1_st_idx, col1_ed_idx), (col2_st_idx, col2_ed_idx), ...]
    # i_hds = [[(11, 14), (15, 19), (20, 21), (22, 24), (25, 27), (28, 29)],
            #  [(16, 19), (20, 24), (25, 26), (27, 29), (30, 32), (33, 34)]]

    pool_type = 'start_tok', 'end_tok', 'avg'

    """
    bS = len(l_hs)
    l_hs_max = max(l_hs)
    wemb_h = torch.zeros([bS, l_hs_max, hS]).to(device)
    for b, i_hds1 in enumerate(i_hds):
        for i_hd, st_ed_pair in enumerate(i_hds1):
            st, ed = st_ed_pair
            if col_pool_type == 'start_tok':
                vec = all_encoder_layer[-1][b, st,:]
            elif col_pool_type == 'end_tok':
                vec = all_encoder_layer[-1][b, ed, :]
            elif col_pool_type == 'avg':
                vecs = all_encoder_layer[-1][b, st:ed,:]
                vec = vecs.mean(dim=1, keepdim=True)
            else:
                raise ValueError
            wemb_h[b, i_hd, :] = vec

    return wemb_h


def cal_prob(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi):
    """

    :param s_sc: [B, l_h]
    :param s_sa: [B, l_a] # 16
    :param s_wn: [B, 5]
    :param s_wc: [B, l_h]
    :param s_wo: [B, 4, l_o] #
    :param s_wv: [B, 4, 22]
    :return:
    """
    # First get selected index

    #

    # Predict prob
    p_sc = cal_prob_sc(s_sc, pr_sc)
    p_sa = cal_prob_sa(s_sa, pr_sa)
    p_wn = cal_prob_wn(s_wn, pr_wn)
    p_wc = cal_prob_wc(s_wc, pr_wc)
    p_wo = cal_prob_wo(s_wo, pr_wo)
    p_wvi = cal_prob_wvi_se(s_wv, pr_wvi)

    # calculate select-clause probability
    p_select = cal_prob_select(p_sc, p_sa)

    # calculate where-clause probability
    p_where  = cal_prob_where(p_wn, p_wc, p_wo, p_wvi)

    # calculate total probability
    p_tot = cal_prob_tot(p_select, p_where)

    return p_tot, p_select, p_where, p_sc, p_sa, p_wn, p_wc, p_wo, p_wvi

def cal_prob_tot(p_select, p_where):
    p_tot = []
    for b, p_select1 in enumerate(p_select):
        p_where1 = p_where[b]
        p_tot.append( p_select1 * p_where1 )

    return p_tot

def cal_prob_select(p_sc, p_sa):
    p_select = []
    for b, p_sc1 in enumerate(p_sc):
        p1 = 1.0
        p1 *= p_sc1
        p1 *= p_sa[b]

        p_select.append(p1)
    return p_select

def cal_prob_where(p_wn, p_wc, p_wo, p_wvi):
    p_where = []
    for b, p_wn1 in enumerate(p_wn):
        p1 = 1.0
        p1 *= p_wn1
        p_wc1 = p_wc[b]

        for i_wn, p_wc11 in enumerate(p_wc1):
            p_wo11 = p_wo[b][i_wn]
            p_wv11_st, p_wv11_ed = p_wvi[b][i_wn]

            p1 *= p_wc11
            p1 *= p_wo11
            p1 *= p_wv11_st
            p1 *= p_wv11_ed

        p_where.append(p1)

    return p_where


def cal_prob_sc(s_sc, pr_sc):
    ps = F.softmax(s_sc, dim=1)
    p = []
    for b, ps1 in enumerate(ps):
        pr_sc1 = pr_sc[b]
        p1 = ps1[pr_sc1]
        p.append(p1.item())

    return p

def cal_prob_sa(s_sa, pr_sa):
    ps = F.softmax(s_sa, dim=1)
    p = []
    for b, ps1 in enumerate(ps):
        pr_sa1 = pr_sa[b]
        p1 = ps1[pr_sa1]
        p.append(p1.item())

    return p

def cal_prob_wn(s_wn, pr_wn):
    ps = F.softmax(s_wn, dim=1)
    p = []
    for b, ps1 in enumerate(ps):
        pr_wn1 = pr_wn[b]
        p1 = ps1[pr_wn1]
        p.append(p1.item())

    return p

def cal_prob_wc(s_wc, pr_wc):
    ps = torch.sigmoid(s_wc)
    ps_out = []
    for b, pr_wc1 in enumerate(pr_wc):
        ps1 = array(ps[b].cpu())
        ps_out1 = ps1[pr_wc1].tolist()
        ps_out.append(ps_out1)

    return ps_out

def cal_prob_wo(s_wo, pr_wo):
    # assume there is always at least single condition.
    ps = F.softmax(s_wo, dim=2)
    ps_out = []


    for b, pr_wo1 in enumerate(pr_wo):
        ps_out1 = []
        for n, pr_wo11 in enumerate(pr_wo1):
            ps11 = ps[b][n]
            ps_out1.append( ps11[pr_wo11].item() )


        ps_out.append(ps_out1)

    return ps_out


def cal_prob_wvi_se(s_wv, pr_wvi):
    prob_wv = F.softmax(s_wv, dim=-2).detach().to('cpu').numpy()
    p_wv = []
    for b, pr_wvi1 in enumerate(pr_wvi):
        p_wv1 = []
        for i_wn, pr_wvi11 in enumerate(pr_wvi1):
            st, ed = pr_wvi11
            p_st = prob_wv[b, i_wn, st, 0]
            p_ed = prob_wv[b, i_wn, ed, 1]
            p_wv1.append([p_st, p_ed])
        p_wv.append(p_wv1)

    return p_wv

def generate_inputs_s2s(tokenizer, nlu1_tt, hds1, sql_vocab1):
    """
    [CLS] sql_vocab [SEP] question [SEP] headers
    To make sql_vocab in a fixed position.
    """

    tokens = []
    segment_ids = []

    tokens.append("[CLS]")


    # sql_vocab
    i_sql_vocab = []
    # for doc
    for i, sql_vocab11 in enumerate(sql_vocab1):
        i_st_sql = len(tokens)
        sub_tok = tokenizer.tokenize(sql_vocab11)
        tokens += sub_tok
        i_ed_sql = len(tokens)
        i_sql_vocab.append((i_st_sql, i_ed_sql))
        segment_ids += [1] * len(sub_tok)
        if i < len(sql_vocab1) - 1:
            tokens.append("[SEP]")
            segment_ids.append(0)
        elif i == len(sql_vocab1) - 1:
            tokens.append("[SEP]")
            segment_ids.append(1)
        else:
            raise EnvironmentError


    # question
    i_st_nlu = len(tokens)  # to use it later

    segment_ids.append(0)
    for token in nlu1_tt:
        tokens.append(token)
        segment_ids.append(0)
    i_ed_nlu = len(tokens)
    tokens.append("[SEP]")
    segment_ids.append(0)
    i_nlu = (i_st_nlu, i_ed_nlu)


    # headers
    i_hds = []
    # for doc
    for i, hds11 in enumerate(hds1):
        i_st_hd = len(tokens)
        sub_tok = tokenizer.tokenize(hds11)
        tokens += sub_tok
        i_ed_hd = len(tokens)
        i_hds.append((i_st_hd, i_ed_hd))
        segment_ids += [1] * len(sub_tok)
        if i < len(hds1)-1:
            tokens.append("[SEP]")
            segment_ids.append(0)
        elif i == len(hds1)-1:
            tokens.append("[SEP]")
            segment_ids.append(1)
        else:
            raise EnvironmentError


    return tokens, segment_ids, i_sql_vocab, i_nlu, i_hds


def sort_pr_wc(pr_wc, g_wc):
    """
    Input: list
    pr_wc = [B, n_conds]
    g_wc = [B, n_conds]


    Return: list
    pr_wc_sorted = [B, n_conds]
    """
    pr_wc_sorted = []
    for b, pr_wc1 in enumerate(pr_wc):
        g_wc1 = g_wc[b]
        pr_wc1_sorted = []

        if set(g_wc1) == set(pr_wc1) and len(g_wc1) == len(pr_wc1):
            pr_wc1_sorted = deepcopy(g_wc1)
        else:
            # no sorting when g_wc1 and pr_wc1 are different.
            pr_wc1_sorted = deepcopy(pr_wc1)

        pr_wc_sorted.append(pr_wc1_sorted)
    return pr_wc_sorted

