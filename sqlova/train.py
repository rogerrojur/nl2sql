# Copyright 2019-present NAVER Corp.
# Apache License v2.0

# Wonseok Hwang
# Sep30, 2018

#execute : python train.py --seed 1 --bS 4 --accumulate_gradients 2 --bert_type_abb zhS --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_leng 400
from pytorch_pretrained_bert import BertModel, BertTokenizer

import numpy as np
import os, sys, argparse, re, json

from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as python_random
# import torchvision.datasets as dsets

# BERT
import bert.tokenization as tokenization
from bert.modeling import BertConfig

from sqlova.utils.utils_wikisql import *
from sqlova.model.nl2sql.wikisql_models import *
from sqlnet.dbengine import DBEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################
# 设置logging,同时输出到文件和屏幕
import logging

logger = logging.getLogger()  # 不加名称设置root logger
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

# 使用FileHandler输出到文件
if not os.path.exists('log'):
    os.makedirs('log')
fh = logging.FileHandler('log/log.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

# 使用StreamHandler输出到屏幕
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

# 添加两个Handler
logger.addHandler(ch)
logger.addHandler(fh)
# logger.info('this is info message')
################################################################

def construct_hyper_param(parser):
    parser.add_argument('--tepoch', default=200, type=int)
    parser.add_argument("--bS", default=32, type=int,
                        help="Batch size")
    parser.add_argument("--user", default=0, type=int,
                        help="0: luokai, 1: jinhao, 2: liuchao")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--fine_tune',
                        default=False,
                        action='store_true',
                        help="If present, BERT is trained.")

    parser.add_argument("--model_type", default='Seq2SQL_v1', type=str,
                        help="Type of model.")

    # 1.2 BERT Parameters
    parser.add_argument("--vocab_file",
                        default='vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--max_seq_length",
                        default=222, type=int, # Set based on maximum length of input tokens.
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_target_layers",
                        default=2, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no_pretraining', action='store_true', help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='uS', type=str,
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")

    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
    parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")

    # 1.4 Execution-guided decoding beam-size. It is used only in test.py
    parser.add_argument('--EG',
                        default=False,
                        action='store_true',
                        help="If present, Execution guided decoding is used in test.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=4,
                        help="The size of beam for smart decoding")

    args = parser.parse_args()

    map_bert_type_abb = {'uS': 'uncased_L-12_H-768_A-12',
                         'zhS': 'bert-base-chinese',#中文，我们需要的
                         'uL': 'uncased_L-24_H-1024_A-16',
                         'cS': 'cased_L-12_H-768_A-12',
                         'cL': 'cased_L-24_H-1024_A-16',
                         'mcS': 'multi_cased_L-12_H-768_A-12'}
    args.bert_type = map_bert_type_abb[args.bert_type_abb]
    print(f"BERT-type: {args.bert_type}")

    # Decide whether to use lower_case.
    if args.bert_type_abb == 'cS' or args.bert_type_abb == 'cL' or args.bert_type_abb == 'mcS':
        args.do_lower_case = False
    else:
        args.do_lower_case = True

    # Seeds for random number generation
    seed(args.seed)
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    #args.toy_model = not torch.cuda.is_available()
    args.toy_model = False
    args.toy_size = 12

    return args


def get_bert(BERT_PT_PATH, bert_type, do_lower_case):
    
    bert_config_file = os.path.join(BERT_PT_PATH, f'bert_config_{bert_type}.json')
    vocab_file = os.path.join(BERT_PT_PATH, f'vocab_{bert_type}.txt')
    #init_checkpoint = os.path.join(BERT_PT_PATH, f'pytorch_model_{bert_type}.bin')
    #init_checkpoint = os.path.join(BERT_PT_PATH, f'bert_model_{bert_type}.ckpt.data')



    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=False)
    bert_config.print_status()
    model_bert = BertModel.from_pretrained(bert_type)
    #model_bert.eval()
    '''
    if no_pretraining:
        pass
    else:
        model_bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
    '''
    print("Load pre-trained parameters.")
    model_bert.to(device)

    return model_bert, tokenizer, bert_config

def get_opt(model, model_bert, fine_tune):
    if fine_tune:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)

        opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                    lr=args.lr_bert, weight_decay=0)
    else:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)
        opt_bert = None

    return opt, opt_bert

def get_models(args, BERT_PT_PATH, trained=False, path_model_bert=None, path_model=None):
    # some constants
    agg_ops = ['', 'AVG', 'MAX', 'MIN', 'COUNT', 'SUM']
    cond_ops = ['>', '<', '==', '!=']
    rela_ops = ['', 'and', 'or']

    print(f"Batch_size = {args.bS * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: {args.fine_tune}")

    # Get BERT
    model_bert, tokenizer, bert_config = get_bert(BERT_PT_PATH, args.bert_type, args.do_lower_case)
    args.iS = bert_config.hidden_size * args.num_target_layers  # Seq-to-SQL input vector dimenstion

    # Get Seq-to-SQL

    n_cond_ops = len(cond_ops)
    n_agg_ops = len(agg_ops)
    n_cond_rps = len(rela_ops)
    print(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    model = Seq2SQL_v1(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_cond_rps, n_agg_ops)
    model = model.to(device)

    if trained:
        assert path_model_bert != None
        assert path_model != None

        if torch.cuda.is_available():
            res = torch.load(path_model_bert)
        else:
            res = torch.load(path_model_bert, map_location='cpu')
        model_bert.load_state_dict(res['model_bert'])
        model_bert.to(device)

        if torch.cuda.is_available():
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')

        model.load_state_dict(res['model'])

    return model, model_bert, tokenizer, bert_config

def get_data(path_wikisql, args):
    train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(path_wikisql, args.toy_model, args.toy_size, no_w2i=True, no_hs_tok=True)
    train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, args.bS, shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader


def train(train_loader, train_table, model, model_bert, opt, bert_config, tokenizer,
          max_seq_length, num_target_layers, accumulate_gradients=1, check_grad=True,
          st_pos=0, opt_bert=None, path_db=None, dset_name='train', mvl=2):#max value length
    model.train()
    model_bert.train()
    #train table is a dict, key is table id, value is the whole table
    
    ave_loss = 0
    cnt = 0 # count the # of examples
    
    cnt_sn = 0 # count select number
    
    cnt_sc = 0 # count the # of correct predictions of select column
    cnt_sa = 0 # of selectd aggregation
    cnt_wn = 0 # of where number
    
    cnt_wr = 0
    
    #where relation number = cnt_wn - 1
    
    cnt_wc = 0 # of where column
    cnt_wo = 0 # of where operator
    cnt_wv = 0 # of where-value
    cnt_wvi = 0 # of where-value index (on question tokens)
    cnt_lx = 0  # of logical form acc
    cnt_x = 0   # of execution acc

    # Engine for SQL querying.
    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))

    for iB, t in enumerate(train_loader):#generate each data batch
        cnt += len(t)

        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True, generate_mode=False)
        # nlu  : natural language utterance
        # nlu_t: tokenized nlu
        # sql_i: canonical form of SQL query
        # sql_q: full SQL query text. Not used.
        # sql_t: tokenized SQL query
        # tb   : table
        # hs_t : tokenized headers. Not used.
        '''
        print('nlu: ', nlu)
        print('nlu_t: ', nlu_t)
        print('sql_i: ', sql_i)
        print('sql_q: ', sql_q)
        print('sql_t: ', sql_t)
        #print('tb: ', tb)
        print('hs_t: ', hs_t)
        print('hds: ', hds)
        '''
        g_sn, g_sc, g_sa, g_wn, g_wr, g_dwn, g_wc, g_wo, g_wv, g_wrcn, wvi_change_index = get_g(sql_i)#get the where values
        '''
        print('g_sn: ', g_sn)
        print('g_sc: ', g_sc)
        print('g_sa: ', g_sa)
        print('g_wn: ', g_wn)
        print('g_wr: ', g_wr)
        print('g_dwn: ', g_dwn)
        print('g_wc: ', g_wc)
        print('g_wo: ', g_wo)
        print('g_wv: ', g_wv)
        print('g_wrcn: ', g_wrcn)
        '''
        
        #g_sn: (a list of double) number of select column;
        #g_sc: (a list of list) select column names;
        #g_sa: (a list of list) agg for each col;
        #g_wr: (a list of double) if value=0, then there is only one condition, else there are two conditions;
        #g_wc: (a list of list) where col;
        #g_wo: (a list of list) where op;
        #g_wv: (a list of list) where val;
        # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
        g_wvi_corenlp = get_g_wvi_corenlp(t, wvi_change_index)
        # this function is to get the indices of where values from the question token

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx, wemb_v, l_npu, l_token \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers, num_out_layers_v=num_target_layers)
        '''
        print('wemb_n: ', torch.tensor(wemb_n).size())
        print('wemb_h: ', torch.tensor(wemb_h).size())
        '''
        #print('l_n: ', l_n[0])
        #print('l_hpu: ', l_hpu)
        #print('l_hs: ', l_hs)
        #print('nlu_tt: ', nlu_tt[0])
        
        #print('t_to_tt_idx: ', t_to_tt_idx)
        #print('tt_to_t_idx: ', tt_to_t_idx)
        #print('g_wvi_corenlp', g_wvi_corenlp)
        
        # wemb_n: natural language embedding
        # wemb_h: header embedding
        # l_n: token lengths of each question
        # l_hpu: header token lengths
        # l_hs: the number of columns (headers) of the tables.
        try:
            #
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)#if not exist, it will not train not include the length, so the end value is the start index of this word, not the end index of this word, so it need to add sth
            g_wvi = g_wvi_corenlp
            if g_wvi:
                for L in g_wvi:
                    for e in L:
                        if e[1] - e[0] + 1 > mvl:
                            cnt -= len(t)
                            print('error: ', e)
                            raise RuntimeError('invalid training set')#only train length no larger than 8 of where value
            g_wvi = get_g_wvi_stidx_length_jian_yi(g_wvi)#不能sort，sort会导致两者对应不上
            #print('g_wvi', g_wvi[0][0])
        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            # e.g. train: 32.
            continue
        # score
        s_sn, s_sc, s_sa, s_wn, s_wr, s_hrpc, s_wc, s_wo, s_wv1, s_wv2, s_wv3, s_wv4 = model(mvl, wemb_n, l_n, wemb_h, l_hpu, l_hs, wemb_v, l_npu, l_token,
                                                   g_sn=g_sn, g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_dwn=g_dwn, g_wr=g_wr, g_wc=g_wc, g_wo=g_wo, g_wvi=g_wvi, g_wrcn=g_wrcn)
        
        #print('g_wvi: ', g_wvi[0])
        '''
        print('s_sn: ', s_sn)
        print('s_sc: ', s_sc)
        print('s_sa: ', s_sa)
        print('s_wn: ', s_wn)
        print('s_wr: ', s_wr)
        print('s_hrpc: ', s_hrpc)
        print('s_wrpc', s_wrpc)
        print('s_nrpc: ', s_nrpc)
        print('s_wc: ', s_wc)
        print('s_wo: ', s_wo)
        print('s_wv1: ', s_wv1)
        print('s_wv2: ', s_wv2)
        '''
        
        # Calculate loss & step
        loss = Loss_sw_se(s_sn, s_sc, s_sa, s_wn, s_wr, s_hrpc, s_wc, s_wo, s_wv1, s_wv2, s_wv3, s_wv4, g_sn, g_sc, g_sa, g_wn, g_dwn, g_wr, g_wc, g_wo, g_wvi, g_wrcn, mvl)
        '''
        print('ave_loss', ave_loss)
        print('loss: ', loss.item())
        print('cnt: ', cnt)
        '''
        # Calculate gradient
        if iB % accumulate_gradients == 0: # mode
            # at start, perform zero_grad
            opt.zero_grad()
            if opt_bert:
                opt_bert.zero_grad()
            loss.backward()
            if accumulate_gradients == 1:
                opt.step()
                if opt_bert:
                    opt_bert.step()
        elif iB % accumulate_gradients == (accumulate_gradients-1):
            # at the final, take step with accumulated graident
            loss.backward()
            opt.step()
            if opt_bert:
                opt_bert.step()
        else:
            # at intermediate stage, just accumulates the gradients
            loss.backward()
        
        #print('grad finish')
        
        # Prediction
        #print('s_wc: ', s_wc.size())
        pr_sn, pr_sc, pr_sa, pr_wn, pr_wr, pr_hrpc, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sn, s_sc, s_sa, s_wn, s_wr, s_hrpc, s_wc, s_wo, s_wv1, s_wv2, s_wv3, s_wv4, mvl)
        '''
        print('pr_sn: ', pr_sn)
        print('pr_sc: ', pr_sc)
        print('pr_sa: ', pr_sa)
        print('pr_wn: ', pr_wn)
        print('pr_wr: ', pr_wr)
        print('pr_hrpc: ', pr_hrpc)
        print('pr_wrpc', pr_wrpc)
        print('pr_nrpc: ', pr_nrpc)
        print('pr_wc: ', pr_wc)
        print('pr_wo: ', pr_wo)
        print('pr_wvi: ', pr_wvi)
        '''
        pr_wvi_decode = g_wvi_decoder_stidx_length_jian_yi(pr_wvi)
        #print('pr_wvi_decode: ', pr_wvi_decode)
        pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi_decode, nlu_t, nlu_tt, tt_to_t_idx)
        #print('pr_wv_str: ', pr_wv_str)
        #print('pr_wv_str_wp: ', pr_wv_str_wp)
        # Sort pr_wc:
        #   Sort pr_wc when training the model as pr_wo and pr_wvi are predicted using ground-truth where-column (g_wc)
        #   In case of 'dev' or 'test', it is not necessary as the ground-truth is not used during inference.
        pr_sc_sorted = sort_pr_wc(pr_sc, g_sc)
        pr_wc_sorted = sort_pr_wc(pr_wc, g_wc)
        #print('pr_wc: ', pr_wc)
        #print('g_wc: ', g_wc)
        pr_sql_i = generate_sql_i(pr_sc_sorted, pr_sa, pr_wn, pr_wr, pr_wc_sorted, pr_wo, pr_wv_str, nlu)
        
        #print('pr_sql_i: ', pr_sql_i)
        

        # Cacluate accuracy
        cnt_sn1_list, cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wr1_list, cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sn, g_sc, g_sa, g_wn, g_wr, g_wc, g_wo, g_wvi,
                                                                   pr_sn, pr_sc, pr_sa, pr_wn, pr_wr, pr_wc, pr_wo, pr_wvi,
                                                                   sql_i, pr_sql_i,
                                                                   mode='train')
        
        '''
        print('cnt_sn1_list: ', cnt_sn1_list)
        print('cnt_sc1_list: ', cnt_sc1_list)
        print('cnt_sa1_list: ', cnt_sa1_list)
        print('cnt_wn1_list: ', cnt_wn1_list)
        print('cnt_wr1_list: ', cnt_wr1_list)
        print('cnt_wc1_list: ', cnt_wc1_list)
        print('cnt_wo1_list', cnt_wo1_list)
        print('cnt_wvi1_list: ', cnt_wvi1_list)
        print('cnt_wv1_list: ', cnt_wv1_list)
        '''
        
        cnt_lx1_list = get_cnt_lx_list(cnt_sn1_list, cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wr1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)
        # lx stands for logical form accuracy

        # Execution accuracy test.
        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

        # statistics
        ave_loss += loss.item()
        
        '''
        print('cnt_lx1_list: ', cnt_lx1_list)
        print('cnt_x1_list: ', cnt_x1_list)
        print('g_ans: ', g_ans)
        print('pr_ans: ', pr_ans)
        print('ave_loss: ', ave_loss)
        '''
        

        # count
        cnt_sn += sum(cnt_sn1_list)
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wr += sum(cnt_wr1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)
        if iB % 200 == 0:
            logger.info('%d - th data batch -> loss: %.4f; acc_sn: %.4f; acc_sc: %.4f; acc_sa: %.4f; acc_wn: %.4f; acc_wr: %.4f; acc_wc: %.4f; acc_wo: %.4f; acc_wvi: %.4f; acc_wv: %.4f; acc_lx: %.4f; acc_x %.4f;' % 
                (iB, ave_loss / cnt, cnt_sn / cnt, cnt_sc / cnt, cnt_sa / cnt, cnt_wn / cnt, cnt_wr / cnt, cnt_wc / cnt, cnt_wo / cnt, cnt_wvi / cnt, cnt_wv / cnt, cnt_lx / cnt, cnt_x / cnt))
            #print('train: [ ', iB, '- th data batch -> loss:', ave_loss / cnt, '; acc_sn: ', cnt_sn / cnt, '; acc_sc: ', cnt_sc / cnt, '; acc_sa: ', cnt_sa / cnt, '; acc_wn: ', cnt_wn / cnt, '; acc_wr: ', cnt_wr / cnt, '; acc_wc: ', cnt_wc / cnt, '; acc_wo: ', cnt_wo / cnt, '; acc_wvi: ', cnt_wvi / cnt, '; acc_wv: ', cnt_wv / cnt, '; acc_lx: ', cnt_lx / cnt, '; acc_x: ', cnt_x / cnt, ' ]')

    
    ave_loss = ave_loss / cnt
    acc_sn = cnt_sn / cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wr = cnt_wr / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sn, acc_sc, acc_sa, acc_wn, acc_wr, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]

    aux_out = 1
    
    return acc, aux_out

def report_detail(hds, nlu,
                  g_sn, g_sc, g_sa, g_wn, g_wr, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                  pr_sn, pr_sc, pr_sa, pr_wn, pr_wr, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                  cnt_list, current_cnt):
    cnt_tot, cnt, cnt_sn, cnt_sc, cnt_sa, cnt_wn, cnt_wr, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x = current_cnt

    print(f'cnt = {cnt} / {cnt_tot} ===============================')

    print(f'headers: {hds}')
    print(f'nlu: {nlu}')

    # print(f's_sc: {s_sc[0]}')
    # print(f's_sa: {s_sa[0]}')
    # print(f's_wn: {s_wn[0]}')
    # print(f's_wc: {s_wc[0]}')
    # print(f's_wo: {s_wo[0]}')
    # print(f's_wv: {s_wv[0][0]}')
    print(f'===============================')
    print(f'g_sn : {g_sn}')
    print(f'pr_sn : {pr_sn}')
    print(f'g_sc : {g_sc}')
    print(f'pr_sc: {pr_sc}')
    print(f'g_sa : {g_sa}')
    print(f'pr_sa: {pr_sa}')
    print(f'g_wn : {g_wn}')
    print(f'pr_wn: {pr_wn}')
    print(f'g_wr : {g_wr}')
    print(f'pr_wr : {pr_wr}')
    print(f'g_wc : {g_wc}')
    print(f'pr_wc: {pr_wc}')
    print(f'g_wo : {g_wo}')
    print(f'pr_wo: {pr_wo}')
    print(f'g_wv : {g_wv}')
    # print(f'pr_wvi: {pr_wvi}')
    print('g_wv_str:', g_wv_str)
    print('p_wv_str:', pr_wv_str)
    print(f'g_sql_q:  {g_sql_q}')
    print(f'pr_sql_q: {pr_sql_q}')
    print(f'g_ans: {g_ans}')
    print(f'pr_ans: {pr_ans}')
    print(f'--------------------------------')

    print(cnt_list)

    print(f'acc_lx = {cnt_lx/cnt:.3f}, acc_x = {cnt_x/cnt:.3f}\n',
          f'acc_sn = {cnt_sn/cnt:.3f}, acc_sc = {cnt_sc/cnt:.3f}, acc_sa = {cnt_sa/cnt:.3f}, acc_wn = {cnt_wn/cnt:.3f}\n',
          f'acc_wr = {cnt_wr/cnt:.3f}, acc_wc = {cnt_wc/cnt:.3f}, acc_wo = {cnt_wo/cnt:.3f}, acc_wv = {cnt_wv/cnt:.3f}')
    print(f'===============================')

def test(data_loader, data_table, model, model_bert, bert_config, tokenizer,
         max_seq_length,
         num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
         path_db=None, dset_name='test', mvl=2):
    model.eval()
    model_bert.eval()

    ave_loss = 0
    cnt = 0
    cnt_sn = 0
    cnt_sc = 0
    cnt_sa = 0
    cnt_wn = 0
    cnt_wr = 0
    cnt_wc = 0
    cnt_wo = 0
    cnt_wv = 0
    cnt_wvi = 0
    cnt_lx = 0
    cnt_x = 0

    cnt_list = []

    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    results = []
    for iB, t in enumerate(data_loader):

        cnt += len(t)
        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True, generate_mode=False)

        g_sn, g_sc, g_sa, g_wn, g_wr, g_dwn, g_wc, g_wo, g_wv, g_r_c_n, wvi_change_index = get_g(sql_i)
        g_wrcn = g_r_c_n
        

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx, wemb_v, l_npu, l_token \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers, num_out_layers_v=num_target_layers)
        try:#here problem
            #print('ok')
            g_wvi_corenlp = get_g_wvi_corenlp(t, wvi_change_index)
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
            #print('no')
            
            g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx)
            g_wvi = get_g_wvi_stidx_length_jian_yi(g_wvi_corenlp)
            #print('gogogo:', g_wvi)
            #这里需要连同脏数据一起计算准确率
        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            for b in range(len(nlu)):
                results1 = {}
                results1["error"] = "Skip happened"
                results1["nlu"] = nlu[b]
                results1["table_id"] = tb[b]["id"]
                results.append(results1)
            continue

        # model specific part
        # score
        if not EG:
            # No Execution guided decoding
            s_sn, s_sc, s_sa, s_wn, s_wr, s_hrpc, s_wc, s_wo, s_wv1, s_wv2, s_wv3, s_wv4 = model(mvl, wemb_n, l_n, wemb_h, l_hpu, l_hs, wemb_v, l_npu, l_token)

            # get loss & step
            #loss = Loss_sw_se(s_sn, s_sc, s_sa, s_wn, s_wr, s_hrpc, s_wrpc, s_nrpc, s_wc, s_wo, s_wv1, s_wv2, g_sn, g_sc, g_sa, g_wn, g_dwn, g_wr, g_wc, g_wo, g_wvi, g_wrcn)
            #unable for loss
            loss = torch.tensor([0])
            # prediction
            pr_sn, pr_sc, pr_sa, pr_wn, pr_wr, pr_hrpc, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sn, s_sc, s_sa, s_wn, s_wr, s_hrpc, s_wc, s_wo, s_wv1, s_wv2, s_wv3, s_wv4, mvl)
            pr_wvi_decode = g_wvi_decoder_stidx_length_jian_yi(pr_wvi)
            pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi_decode, nlu_t, nlu_tt, tt_to_t_idx)
            # g_sql_i = generate_sql_i(g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_str, nlu)
            pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wr, pr_wc, pr_wo, pr_wv_str, nlu)
        else:
            # Execution guided decoding
            prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                            l_hs, engine, tb,
                                                                                            nlu_t, nlu_tt,
                                                                                            tt_to_t_idx, nlu,
                                                                                            beam_size=beam_size)
            # sort and generate
            pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)

            # Follosing variables are just for the consistency with no-EG case.
            pr_wvi = None # not used
            pr_wv_str=None
            pr_wv_str_wp=None
            loss = torch.tensor([0])



        g_sql_q = generate_sql_q(sql_i, tb)
        pr_sql_q = generate_sql_q(pr_sql_i, tb)


        # Saving for the official evaluation later.
        for b, pr_sql_i1 in enumerate(pr_sql_i):
            results1 = {}
            results1["query"] = pr_sql_i1
            results1["table_id"] = tb[b]["id"]
            results1["nlu"] = nlu[b]
            results.append(results1)

        cnt_sn1_list, cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wr1_list, cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sn, g_sc, g_sa, g_wn, g_wr, g_wc, g_wo, g_wvi,
                                                                   pr_sn, pr_sc, pr_sa, pr_wn, pr_wr, pr_wc, pr_wo, pr_wvi,
                                                                   sql_i, pr_sql_i,
                                                                   mode='test')

        cnt_lx1_list = get_cnt_lx_list(cnt_sn1_list, cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wr1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)

        # Execution accura y test
        cnt_x1_list = []
        # lx stands for logical form accuracy

        # Execution accuracy test.
        cnt_x1_list, g_ans, pr_ans = get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

        # stat
        ave_loss += loss.item()
        
        #print('loss: ', ave_loss / cnt)

        # count
        cnt_sn += sum(cnt_sn1_list)
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wr += sum(cnt_wr1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

        current_cnt = [cnt_tot, cnt, cnt_sn, cnt_sc, cnt_sa, cnt_wn, cnt_wr, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x]
        cnt_list1 = [cnt_sn1_list, cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wr1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_lx1_list,
                     cnt_x1_list]
        cnt_list.append(cnt_list1)
        # report
        if detail:
            report_detail(hds, nlu,
                          g_sn, g_sc, g_sa, g_wn, g_wr, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                          pr_sn, pr_sc, pr_sa, pr_wn, pr_wr, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                          cnt_list1, current_cnt)

    ave_loss /= cnt
    acc_sn = cnt_sn / cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wr = cnt_wr / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sn, acc_sc, acc_sa, acc_wn, acc_wr, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
    return acc, results, cnt_list


def print_result(epoch, acc, dname):
    ave_loss, acc_sn, acc_sc, acc_sa, acc_wn, acc_wr, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sn: {acc_sn:.3f}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wr: {acc_wr:.3f}, acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )

if __name__ == '__main__':

    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    path_h = 'D:\\tianChi\\nl2sql\\sqlova\\wikisql'
    if args.user == 1:
        path_h = './wikisql'
    path_wikisql = os.path.join(path_h, 'data', 'tianchi')
    BERT_PT_PATH = path_wikisql

    path_save_for_evaluation = './'

    ## 3. Load data
    train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_data(path_wikisql, args)
    print("train_data: ", len(train_data))
    print("train_table: ", len(train_table))
    print("dev_data: ", len(dev_data))
    print("dev_table: ", len(dev_table))
    

    # test_data, test_table = load_wikisql_data(path_wikisql, mode='test', toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)
    # test_loader = torch.utils.data.DataLoader(
    #     batch_size=args.bS,
    #     dataset=test_data,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=lambda x: x  # now dictionary values are not merged!
    # )
    ## 4. Build & Load models
    # model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH)

    ## 4.1.
    # To start from the pre-trained models, un-comment following lines.
    path_model_bert = 'model_bert_best.pt'
    path_model = 'model_best.pt'
    model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True, path_model_bert=path_model_bert, path_model=path_model)

    ## 5. Get optimizers
    opt, opt_bert = get_opt(model, model_bert, args.fine_tune)
    #print('ok')
    ## 6. Train
    acc_lx_t_best = -1
    epoch_best = -1
    for epoch in range(args.tepoch):
        # train
        
        acc_train, aux_out_train = train(train_loader,
                                         train_table,
                                         model,
                                         model_bert,
                                         opt,
                                         bert_config,
                                         tokenizer,
                                         args.max_seq_length,
                                         args.num_target_layers,
                                         args.accumulate_gradients,
                                         opt_bert=opt_bert,
                                         st_pos=0,
                                         path_db=path_wikisql,
                                         dset_name='train',
                                         mvl=2)
        
        # check DEV
        with torch.no_grad():
            acc_dev, results_dev, cnt_list = test(dev_loader,
                                                dev_table,
                                                model,
                                                model_bert,
                                                bert_config,
                                                tokenizer,
                                                args.max_seq_length,
                                                args.num_target_layers,
                                                detail=False,
                                                path_db=path_wikisql,
                                                st_pos=0,
                                                dset_name='val', EG=args.EG,
                                                mvl=2)


        print_result(epoch, acc_train, 'train')
        print_result(epoch, acc_dev, 'val')

        # save results for the official evaluation
        save_for_evaluation(path_save_for_evaluation, results_dev, 'val')



        # save best model
        # Based on Dev Set logical accuracy lx
        acc_lx_t = acc_dev[-2]
        if acc_lx_t > acc_lx_t_best:
            acc_lx_t_best = acc_lx_t
            epoch_best = epoch
            # save best model
            state = {'model': model.state_dict()}
            torch.save(state, os.path.join('.', 'model_best.pt') )

            state = {'model_bert': model_bert.state_dict()}
            torch.save(state, os.path.join('.', 'model_bert_best.pt'))

        print(f" Best Val lx acc: {acc_lx_t_best} at epoch: {epoch_best}")
        
