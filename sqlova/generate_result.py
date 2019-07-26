# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 20:59:53 2019

@author: 63184
"""

# Copyright 2019-present NAVER Corp.
# Apache License v2.0

# Wonseok Hwang
# Sep30, 2018

#execute : python generate_result.py --seed 1 --bS 4 --accumulate_gradients 2 --bert_type_abb zhS --fine_tune --lr 0.001 --lr_bert 0.00001 --max_seq_leng 400
from bert.modeling import BertConfig

import numpy as np
import os, argparse

# from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as python_random
# import torchvision.datasets as dsets

# BERT
import bert.tokenization as tokenization
from pytorch_pretrained_bert import BertModel, BertTokenizer

from sqlova.utils.utils_wikisql import *
from sqlova.model.nl2sql.wikisql_models import *
from sqlnet.dbengine import DBEngine

import token_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def construct_hyper_param(parser):
    parser.add_argument('--tepoch', default=200, type=int)
    parser.add_argument("--bS", default=32, type=int,
                        help="Batch size")
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
    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=False)
    bert_config.print_status()
    model_bert = BertModel.from_pretrained(bert_type)
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
    test_data, test_table = load_wikisql_data(path_wikisql, mode='test', toy_model=args.toy_model, no_hs_tok=True)
    
    test_loader = torch.utils.data.DataLoader(
        batch_size=args.bS,
        dataset=test_data,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    return test_data, test_table, test_loader

def test(data_loader, data_table, model, model_bert, bert_config, tokenizer,
         max_seq_length,
         num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
         path_db=None, dset_name='test', mvl=2):
    model.eval()
    model_bert.eval()

    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    results = []
    for iB, records in enumerate(data_loader):
        #print('iB: ', iB)#to locate the error

        # Invoke the token_test_each() function, which transfer each record in test to record_tok
        # print(record)
        # # print(data_table)
        # print(record['table_id'])
        t = []
        for record in records:
            t.append(token_utils.token_test_each(record, data_table))
       
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True, generate_mode=True)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx, wemb_v, l_npu, l_token \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers, num_out_layers_v=num_target_layers)

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

        pr_sql_q = generate_sql_q(pr_sql_i, tb)


        # Saving for the official evaluation later.
        for b, pr_sql_i1 in enumerate(pr_sql_i):
            results1 = pr_sql_i1
            results.append(results1)
            
    return results


if __name__ == '__main__':

    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    path_h = './wikisql'
    path_wikisql = os.path.join(path_h, 'data', 'tianchi')
    BERT_PT_PATH = path_wikisql

    path_save_for_evaluation = './result'

    ## 3. Load data
    test_data, test_table, test_loader = get_data(path_wikisql, args)
    print("test_data: ", len(test_data))
    print("test_table: ", len(test_table))
    

    # test_data, test_table = load_wikisql_data(path_wikisql, mode='test', toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)
    # test_loader = torch.utils.data.DataLoader(
    #     batch_size=args.bS,
    #     dataset=test_data,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=lambda x: x  # now dictionary values are not merged!
    # )
    ## 4. Build & Load models
    model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True, path_model_bert='./model_bert_best.pt', path_model='model_best.pt')

    ## 4.1.
    # To start from the pre-trained models, un-comment following lines.
    # path_model_bert =
    # path_model =
    # model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True, path_model_bert=path_model_bert, path_model=path_model)

    ## 5. Get optimizers
    opt, opt_bert = get_opt(model, model_bert, args.fine_tune)
    #print('ok')
    ## 6. Train
    with torch.no_grad():
        results_test = test(test_loader,
                            test_table,
                            model,
                            model_bert,
                            bert_config,
                            tokenizer,
                            args.max_seq_length,
                            args.num_target_layers,
                            detail=False,
                            path_db=path_wikisql,
                            st_pos=0,
                            dset_name='test', EG=args.EG,
                            mvl=2)


        # save results for the official evaluation
    save_for_evaluation(path_save_for_evaluation, results_test, 'test')
        
