# Copyright 2019-present NAVER Corp.
# Apache License v2.0

# Wonseok Hwang

import os, json
from copy import deepcopy
from matplotlib.pylab import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sqlova.utils.utils import topk_multi_dim
from sqlova.utils.utils_wikisql import *

class Seq2SQL_v1(nn.Module):
    def __init__(self, iS, hS, lS, dr, n_cond_ops, n_cond_rps, n_agg_ops, old=False):
        super(Seq2SQL_v1, self).__init__()
        self.iS = iS
        self.hS = hS
        self.ls = lS
        self.dr = dr
        
        self.max_sn = 3
        self.max_wn = 4
        self.n_cond_ops = n_cond_ops
        self.n_agg_ops = n_agg_ops
        self.n_cond_rps = n_cond_rps

        self.snp = SNP(iS, hS, lS, dr)
        
        self.scp = SCP(iS, hS, lS, dr)
        self.sap = SAP(iS, hS, lS, dr, n_agg_ops, old=old)
        self.wnp = WNP(iS, hS, lS, dr)
        
        self.wrp = WRP(iS, hS, lS, dr, n_cond_rps)
        
        self.hrpc = HRPC(iS, hS, lS, dr)# have repeated column
        
        self.wcp = WCP(iS, hS, lS, dr)
        self.wop = WOP(iS, hS, lS, dr, n_cond_ops)
        self.wvp1 = WVP_se(iS, hS, lS, dr, n_cond_ops, old=old) # start-end-search-discriminative model
        self.wvp2 = WVP_se2(iS, hS, lS, dr, n_cond_ops, old=old)
        
        self.wvp3 = WVP_se3(iS, hS, lS, dr, n_cond_ops, old=old)
        self.wvp4 = WVP_se4(iS, hS, lS, dr, n_cond_ops, old=old)


    def forward(self, mvl, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wemb_v, l_npu, l_token,
                g_sn=None, g_sc=None, g_sa=None, g_wr=None, g_wn=None, g_dwn=None, g_wc=None, g_wo=None, g_wvi=None, g_wrcn=None,
                show_p_sn=False, show_p_sc=False, show_p_sa=False, show_p_wr=False, show_p_hrpc=False, show_p_wrpc=False, show_p_nrpc=False,
                show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False):
        
        # sn
        s_sn = self.snp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sn=show_p_sn)
        
        if g_sn:
            pr_sn = g_sn
        else:
            pr_sn = pred_sn(s_sn)

        # sc
        s_sc = self.scp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sc=show_p_sc)#wemb_hpu is wemb_h

        if g_sc:
            pr_sc = g_sc
        else:
            pr_sc = pred_sc(pr_sn, s_sc)

        # sa
        s_sa = self.sap(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_sn, pr_sc, show_p_sa=show_p_sa)
        if g_sa:
            # it's not necessary though.
            pr_sa = g_sa
        else:
            pr_sa = pred_sa(pr_sn, s_sa)


        # wn
        s_wn = self.wnp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wn=show_p_wn)

        if g_wn:
            pr_wn = g_wn
        else:
            pr_wn = pred_wn(s_wn)
            
        s_wr = self.wrp(wemb_n, l_n, pr_wn, show_p_wr=show_p_wr)
        
        if g_wr:
            pr_wr = g_wr
        else:
            pr_wr = pred_wr(pr_wn, s_wr)
        #for all elements in pr_wn with value 1, it will simply let pr_wr becomes "", otherwise, it will caculate whether it is "and" or "or"

        s_hrpc = self.hrpc(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_wr, show_p_hrpc=show_p_hrpc)
        
        if g_wrcn:
            pr_hrpc = [0 if e[0] == -1 else 1 for e in g_wrcn]
        else:
            pr_hrpc = pred_hrpc(s_hrpc)
            
        if g_dwn:
            pr_dwn = g_dwn
        else:
            pr_dwn = pred_dwn(pr_wn, pr_hrpc)
        
        # wc
        s_wc = self.wcp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wc=show_p_wc, penalty=True)
        
        
        
        if g_wc:
            pr_wc = g_wc
        else:
            pr_wc = pred_wc(pr_dwn, s_wc, pr_wn, pr_hrpc)
        '''
        print('pr_hrpc: ', pr_hrpc)
        print('pr_wrpc: ', pr_wrpc)
        print('pr_nrpc: ', pr_nrpc)
        '''
        # wo
        s_wo = self.wop(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, show_p_wo=show_p_wo)

        if g_wo:
            pr_wo = g_wo
        else:
            pr_wo = pred_wo(pr_wn, s_wo)

        # wv
        s_wv1 = self.wvp1(wemb_v, l_npu, l_token, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, wo=pr_wo, show_p_wv=show_p_wv)
        
        if g_wvi:
            pr_wvi1 = [[e[0] for e in l] for l in g_wvi]
        else:
            pr_wvi1 = pred_wvi1(pr_wn, s_wv1)
            #pr_wvi1 = pred_wvi1_hrpc(pr_wn, s_wv1, pr_hrpc, pr_wr, l_hs)
        
        s_wv2 = self.wvp2(wemb_v, l_npu, l_token, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, wo=pr_wo, wvi1=pr_wvi1, mvl=mvl, show_p_wv=show_p_wv)#it represent 长度-1 the g_wvi will convert to [start_index, length-1]
        
        s_wv3 = self.wvp3(wemb_v, l_npu, l_token, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, wo=pr_wo, show_p_wv=show_p_wv)
        
        if g_wvi:
            pr_wvi3 = [[e[0] + e[1] for e in l] for l in g_wvi]#end index
        else:
            pr_wvi3 = pred_wvi1(pr_wn, s_wv3)#it is the same prediction
            #pr_wvi3 = pred_wvi1_hrpc(pr_wn, s_wv3, pr_hrpc, pr_wr, l_hs)
            
        s_wv4 = self.wvp4(wemb_v, l_npu, l_token, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, wo=pr_wo, wvi3=pr_wvi3, mvl=mvl, show_p_wv=show_p_wv)
        
        

        return s_sn, s_sc, s_sa, s_wn, s_wr, s_hrpc, s_wc, s_wo, s_wv1, s_wv2, s_wv3, s_wv4

    def beam_forward(self, normal_sql_i, mvl, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wemb_v, l_npu, l_token, engine, tb,
                     nlu_t,
                     beam_size=4,
                     show_p_sn=False, show_p_sc=False, show_p_sa=False,
                     show_p_wn=False, show_p_wr=False, show_p_hrpc=False, show_p_wc=False, show_p_wo=False, show_p_wv=False):
        """
        Execution-guided beam decoding.
        """
        
        s_sn = self.snp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sn=show_p_sn)
        
        pr_sn = pred_sn(s_sn)
        
        s_sc = self.scp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sc=show_p_sc)
        
        pr_sc = pred_sc(pr_sn, s_sc)
        
        s_sa = self.sap(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_sn, pr_sc, show_p_sa=show_p_sa)
        
        pr_sa = pred_sa(pr_sn, s_sa)
        
        #pr_sc = guide_pred_sc(pr_sn, s_sc, pr_sa, tb)
        
        s_wn = self.wnp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wn=show_p_wn)
        
        pr_wn = pred_wn(s_wn)
        
        s_wr = self.wrp(wemb_n, l_n, pr_wn, show_p_wr=show_p_wr)
        
        pr_wr = pred_wr(pr_wn, s_wr)
        
        s_hrpc = self.hrpc(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_wr, show_p_hrpc=show_p_hrpc)
        
        pr_hrpc = pred_hrpc(s_hrpc)
        
        
        #pr_wr = re_pred_wr(pr_wr, pr_hrpc)
        
        #pr_hrpc = re_pred_hrpc(pr_wr, pr_hrpc)
        
        pr_dwn = pred_dwn(pr_wn, pr_hrpc)
        
        s_wc = self.wcp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wc=show_p_wc, penalty=True)
        
        pr_wc = pred_wc(pr_dwn, s_wc, pr_wn, pr_hrpc)
        
        s_wo = self.wop(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, show_p_wo=show_p_wo)
        
        pr_wo = pred_wo(pr_wn, s_wo)
        
        #pr_wc = guide_pred_wc(pr_sc, pr_hrpc, pr_wn, s_wc, pr_wo, tb)
        
        #s_wv1 = self.wvp1(wemb_v, l_npu, l_token, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, wo=pr_wo, show_p_wv=show_p_wv)
        
        #pr_wvi1 = guide_pred_wvi1(pr_wn, pr_wc, s_wv1, tb, nlu_t)
        
        #pr_wvi1 = pred_wvi1(pr_wn, s_wv1)
        
        #s_wv2 = self.wvp2(wemb_v, l_npu, l_token, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, wo=pr_wo, wvi1=pr_wvi1, mvl=mvl, show_p_wv=show_p_wv)
        
        #s_wv3 = self.wvp3(wemb_v, l_npu, l_token, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, wo=pr_wo, show_p_wv=show_p_wv)
        
        #pr_wvi3 = guide_pred_wvi1(pr_wn, pr_wc, s_wv3, tb, nlu_t)
        
        #pr_wvi3 = pred_wvi1(pr_wn, s_wv3)
        
        #s_wv4 = self.wvp4(wemb_v, l_npu, l_token, wemb_hpu, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, wo=pr_wo, wvi3=pr_wvi3, mvl=mvl, show_p_wv=show_p_wv)
        
        #pr_wvi_all = pred_wvi_se(pr_wn, s_wv1, s_wv2, s_wv3, s_wv4, mvl)
        
        #pr_wc = guide_pred_wc(pr_hrpc, pr_wn, s_wc, pr_wo, tb, l_hs, pr_wvi_all, nlu_t, engine)
        
        bS = len(normal_sql_i)
        
        pr_sql_list = [{} for _ in range(bS)]#result
        exe_error = 0
        still_error = 0
        for ib in range(bS):
            cur_conds = normal_sql_i[ib]['conds']
            not_repeated_conds = []
            for cond in cur_conds:
                if cond not in not_repeated_conds:
                    not_repeated_conds.append(cond)
            if len(cur_conds) != len(not_repeated_conds):
                if len(not_repeated_conds) <= 1:
                    normal_sql_i[ib]['cond_conn_op'] = 0
                    normal_sql_i[ib]['conds'] = not_repeated_conds
                    cur_conds = not_repeated_conds
                else:
                    normal_sql_i[ib]['conds'] = not_repeated_conds
                    cur_conds = not_repeated_conds
            cur_scas = [[e1, e2] for e1, e2 in zip(normal_sql_i[ib]['sel'], normal_sql_i[ib]['agg'])]
            not_repeated_scas = []
            for sca in cur_scas:
                if sca not in not_repeated_scas:
                    not_repeated_scas.append(sca)
            if len(cur_scas) != len(not_repeated_scas):
                new_sel = [e[0] for e in not_repeated_scas]
                new_agg = [e[1] for e in not_repeated_scas]
                normal_sql_i[ib]['sel'] = new_sel
                normal_sql_i[ib]['agg'] = new_agg
                cur_scas = not_repeated_scas
            
            prob_sc1 = s_sc[ib]
            rank_sc = argsort(-prob_sc1.data.cpu().numpy())
            good_sca = []
            for sc_now, sa_now in zip(normal_sql_i[ib]['sel'], normal_sql_i[ib]['agg']):
                if sa_now == 0 or sa_now == 4 or tb[ib]['types'][sc_now] == 'real':
                    good_sca.append([sc_now, sa_now])
                else:
                    for sc_cur in rank_sc:
                        if sc_cur >= l_hs[ib]:
                            continue
                        if tb[ib]['types'][sc_cur] == 'real':
                            good_sca.append([sc_cur, sa_now])
                            break
            if good_sca:
                normal_sql_i[ib]['sel'] = [e[0] for e in good_sca]
                normal_sql_i[ib]['agg'] = [e[1] for e in good_sca]
            
            colss = [e[0] for e in cur_conds]
            prob_list_h = argsort(-s_wc[ib].data.cpu().numpy()).tolist()
            if len(list(set(colss))) != len(colss):
                col_res, cnt_max, str_res, wvi_res = greedy_wvi_hrpc(l_hs[ib], tb[ib], engine, nlu_t[ib], mvl, colss[0], prob_list_h)
                if col_res != colss[0]:
                    if col_res == -1:#此时直接为wn=0
                        #print('no')
                        #do nothing
                        pass
                    if col_res != -1 and cnt_max == 1:
                        #print('single')
                        pr_wn1 = [4] * bS
                        pr_wc1 = [[col_res for _1 in range(self.max_wn)] for _ in range(bS)]
                        s_wo1 = self.wop(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn1, wc=pr_wc1, show_p_wo=show_p_wo)
                        pr_wo1 = pred_wo(pr_wn1, s_wo1)
                        normal_sql_i[ib]['cond_conn_op'] = 0
                        normal_sql_i[ib]['conds'] = [[col_res, pr_wo1[ib][0], str_res[0]]]
                    else:
                        #print(cnt_max)
                        correct_cnt_max = min(4, cnt_max)
                        pr_wn1 = [correct_cnt_max] * bS
                        pr_wc1 = [[col_res for _1 in range(correct_cnt_max)] for _ in range(bS)]
                        s_wr1 = self.wrp(wemb_n, l_n, pr_wn1, show_p_wr=show_p_wr)
                        pr_wr1 = pred_wr(pr_wn1, s_wr1)
                        s_wo1 = self.wop(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn1, wc=pr_wc1, show_p_wo=show_p_wo)
                        pr_wo1 = pred_wo(pr_wn1, s_wo1)
                        conds_new = []
                        for i, str_single in enumerate(str_res):
                            if i >= correct_cnt_max:
                                break
                            conds_new.append([col_res, pr_wo1[ib][i], str_single])
                        normal_sql_i[ib]['conds'] = conds_new
                        normal_sql_i[ib]['cond_conn_op'] = pr_wr1[ib]
            
            if len(list(set(colss))) == len(colss):
                skip_dict = defaultdict(set)
                nb_of_conversion = 0
                conversion_idxs = []
                for i, cond in enumerate(cur_conds):
                    if engine.check_wc_wv(tb[ib]['id'], cond[0], cond[2]):
                        skip_dict[cond[0]].add(cond[2])
                        continue
                    if cond[1] != 2:
                        continue
                    else:
                        nb_of_conversion += 1
                        conversion_idxs.append(i)
                #调用
                if nb_of_conversion:
                    conversion_elements = greedy_wvi_normal(l_hs[ib], tb[ib], engine, nlu_t[ib], mvl, skip_dict, prob_list_h)
                    if len(conversion_elements) >= nb_of_conversion:
                        #rest_of_ce = conversion_elements[nb_of_conversion:]
                        #rest_of_ce = rest_of_ce[:self.max_wn - len(normal_sql_i[ib]['conds'])]
                        for i, conversion_element in enumerate(conversion_elements[:nb_of_conversion]):
                            normal_sql_i[ib]['conds'][conversion_idxs[i]] = conversion_element
                        #normal_sql_i[ib]['conds'] += rest_of_ce
                        myDict = defaultdict(int)
                        for cond in normal_sql_i[ib]['conds']:
                            myDict[cond[0]] += 1
                        rpc = -1
                        for key in myDict:
                            if myDict[key] >= 2:
                                rpc = key
                                break
                        if rpc != -1:
                            #print(normal_sql_i[ib]['conds'])
                            clean_conds = []
                            for cond in normal_sql_i[ib]['conds']:
                                if cond[0] == rpc:
                                    clean_conds.append(cond)
                            if clean_conds:
                                normal_sql_i[ib]['conds'] = clean_conds
                                normal_sql_i[ib]['cond_conn_op'] = 2
                    else:
                        abandon_part = conversion_idxs[-(nb_of_conversion - len(conversion_elements)):]
                        conversion_idxs = conversion_idxs[:len(conversion_elements)]
                        for i, conversion_element in enumerate(conversion_elements):
                            normal_sql_i[ib]['conds'][conversion_idxs[i]] = conversion_element
                        change_conds = []
                        for i, cond in enumerate(normal_sql_i[ib]['conds']):
                            if i not in abandon_part:
                                change_conds.append(cond)
                        if change_conds:
                            normal_sql_i[ib]['conds'] = change_conds
                        
                        myDict = defaultdict(int)
                        for cond in normal_sql_i[ib]['conds']:
                            myDict[cond[0]] += 1
                        rpc = -1
                        for key in myDict:
                            if myDict[key] >= 2:
                                rpc = key
                                break
                        if rpc != -1:
                            #print(normal_sql_i[ib]['conds'])
                            clean_conds = []
                            for cond in normal_sql_i[ib]['conds']:
                                if cond[0] == rpc:
                                    clean_conds.append(cond)
                            if clean_conds:
                                normal_sql_i[ib]['conds'] = clean_conds
                                normal_sql_i[ib]['cond_conn_op'] = 2
                            
                        if len(normal_sql_i[ib]['conds']) <= 1:
                            normal_sql_i[ib]['cond_conn_op'] = 0
            
            if engine.execute(tb[ib]['id'], normal_sql_i[ib]['sel'], normal_sql_i[ib]['agg'], normal_sql_i[ib]['conds'], normal_sql_i[ib]['cond_conn_op']):
                pr_sql_list[ib] = normal_sql_i[ib]
                continue
            
            exe_error += 1
            conds = normal_sql_i[ib]['conds']
            rela = normal_sql_i[ib]['cond_conn_op']
            new_conds = []
            for i, cond in enumerate(conds):
                if engine.check_wc_wv(tb[ib]['id'], cond[0], cond[2]):
                    new_conds.append(cond)
                else:
                    ok = False
                    prob_wc1 = s_wc[ib]
                    rank_hs = argsort(-prob_wc1.data.cpu().numpy())
                    
                    if not ok:
                        if cond[1] <= 1 and tb[ib]['types'][cond[0]] == 'real':
                            if not ok and check_is_digits(cond[2]):
                                new_conds.append([cond[0], cond[1], cond[2]])
                                ok = True
                    
                    if not ok:
                        for col in rank_hs:
                            if col >= l_hs[ib]:
                                continue
                            if engine.check_wc_wv(tb[ib]['id'], col, cond[2]):
                                if tb[ib]['types'][col] == 'text':
                                    new_conds.append([col, 2, cond[2]])
                                else:
                                    new_conds.append([col, cond[1], cond[2]])
                                ok = True
                                break
                    
                    if not ok and cond[1] == 2:
                        pr_wn1 = [self.max_wn] * bS
                        pr_wc1 = [[cond[0] for _1 in range(self.max_wn)] for _ in range(bS)]
                        s_wo1 = self.wop(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn1, wc=pr_wc1, show_p_wo=show_p_wo)
                        pr_wo1 = pred_wo(pr_wn1, s_wo1)
                        s_wv1_temp = self.wvp1(wemb_v, l_npu, l_token, wemb_hpu, l_hpu, l_hs, wn=pr_wn1, wc=pr_wc1, wo=pr_wo1, show_p_wv=show_p_wv)
                        prob_wv_temp = s_wv1_temp[ib][i]
                        rank_wv = argsort(-prob_wv_temp.data.cpu().numpy())
                        for st in rank_wv:
                            if ok:
                                break
                            if st >= l_token[ib]:
                                continue
                            for ed in range(st, min(st + mvl, len(nlu_t[ib]))):
                                wv_str = single_wvi2str([st, ed], nlu_t[ib])
                                if not wv_str.isdigit() and len(wv_str) == 1:
                                    continue
                                if engine.check_wc_wv(tb[ib]['id'], cond[0], wv_str):
                                    new_conds.append([cond[0], cond[1], wv_str])
                                    ok = True
                                    break
                    
                    if not ok:
                        if tb[ib]['types'][cond[0]] == 'real':
                            if not ok and check_is_digits(cond[2]):
                                new_conds.append([cond[0], cond[1], cond[2]])
                                ok = True
                                
                    if not ok and cond[1] <= 1:
                        for col1 in rank_hs:
                            if ok:
                                break
                            if col1 >= l_hs[ib]:
                                continue
                            if tb[ib]['types'][col1] == 'real':
                                
                                if not ok and check_is_digits(cond[2]):
                                    new_conds.append([col1, cond[1], cond[2]])
                                    ok = True
                                    break
                                pr_wn1 = [self.max_wn] * bS
                                pr_wc1 = [[col1 for _1 in range(self.max_wn)] for _ in range(bS)]
                                s_wo1 = self.wop(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn1, wc=pr_wc1, show_p_wo=show_p_wo)
                                pr_wo1 = pred_wo(pr_wn1, s_wo1)
                                #print(s_wo1.size())
                                s_wv1_temp = self.wvp1(wemb_v, l_npu, l_token, wemb_hpu, l_hpu, l_hs, wn=pr_wn1, wc=pr_wc1, wo=pr_wo1, show_p_wv=show_p_wv)
                                prob_wv_temp = s_wv1_temp[ib][i]
                                rank_wv = argsort(-prob_wv_temp.data.cpu().numpy())
                                for st in rank_wv:
                                    if ok:
                                        break
                                    if st >= l_token[ib]:
                                        continue
                                    for ed in range(st, min(st + mvl, len(nlu_t[ib]))):
                                        wv_str = single_wvi2str([st, ed], nlu_t[ib])
                                        if not wv_str.isdigit() and len(wv_str) == 1:
                                            continue
                                        if engine.check_wc_wv(tb[ib]['id'], col1, wv_str):#wv 是否存在于 table
                                            new_conds.append([col1, cond[1], wv_str])
                                            ok = True
                                            break
                                
                    if not ok:
                        new_conds.append(cond)
            if not new_conds:
                new_conds = normal_sql_i[ib]['conds']
            pr_sql_list[ib] = {'sel': normal_sql_i[ib]['sel'], 'agg': normal_sql_i[ib]['agg'], 'cond_conn_op': rela, 'conds': new_conds}
            if not engine.execute(tb[ib]['id'], normal_sql_i[ib]['sel'], normal_sql_i[ib]['agg'], new_conds, rela):
                still_error += 1
        return pr_sql_list, exe_error, still_error

class SNP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
        super(SNP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.mL_w = 3  # max select condition number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att_h = nn.Linear(hS, 1)#linear 保留前面的维度，只对最后一个维度进行全连接
        self.W_hidden = nn.Linear(hS, lS * hS)
        self.W_cell = nn.Linear(hS, lS * hS)

        self.W_att_n = nn.Linear(hS, 1)
        self.sn_out = nn.Sequential(nn.Linear(hS, hS),
                                    nn.Tanh(),
                                    nn.Linear(hS, self.mL_w))  # max number 3 it is imposible to have 0 selected col

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sn=False):
        # Encode

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, mL_hs, dim]

        bS = len(l_hs)
        mL_n = max(l_n)
        mL_hs = max(l_hs)
        # mL_h = max(l_hpu)

        #   (self-attention?) column Embedding?
        #   [B, mL_hs, 100] -> [B, mL_hs, 1] -> [B, mL_hs], last two may inverse
        att_h = self.W_att_h(wenc_hs).squeeze(2)

        #   Penalty
        for b, l_hs1 in enumerate(l_hs):
            att_h[b, l_hs1:] = -10000000000.0
        p_h = self.softmax_dim1(att_h)

        if show_p_sn:
            if p_h.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001);
            subplot(7,2,5)
            cla()
            plot(p_h[0].data.numpy(), '--rs', ms=7)
            title('wn: header_weight')
            grid(True)
            fig.canvas.draw()
            show()
            # input('Type Eenter to continue.')

        #   [B, mL_hs, 100] * [ B, mL_hs, 1] -> [B, mL_hs, 100] -> [B, 100]
        c_hs = torch.mul(wenc_hs, p_h.unsqueeze(2)).sum(1)

        #   [B, 100] --> [B, 2*100] Enlarge because there are two layers.
        hidden = self.W_hidden(c_hs)  # [B, 4, 200/2]
        hidden = hidden.view(bS, self.lS * 2, int(
            self.hS / 2))  # [4, B, 100/2] # number_of_layer_layer * (bi-direction) # lstm input convention.
        hidden = hidden.transpose(0, 1).contiguous()

        cell = self.W_cell(c_hs)  # [B, 4, 100/2]
        cell = cell.view(bS, self.lS * 2, int(self.hS / 2))  # [4, B, 100/2] the impact on view funciton is to reshape
        cell = cell.transpose(0, 1).contiguous()

        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=(hidden, cell),
                        last_only=False)  # [b, n, dim]

        att_n = self.W_att_n(wenc_n).squeeze(2)  # [B, max_len, 100] -> [B, max_len, 1] -> [B, max_len]

        #    Penalty
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att_n[b, l_n1:] = -10000000000.0
        p_n = self.softmax_dim1(att_n)

        if show_p_sn:
            if p_n.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001);
            subplot(7,2,6)
            cla()
            plot(p_n[0].data.numpy(), '--rs', ms=7)
            title('wn: nlu_weight')
            grid(True)
            fig.canvas.draw()

            show()
            # input('Type Enter to continue.')

        #    [B, mL_n, 100] *([B, mL_n] -> [B, mL_n, 1] -> [B, mL_n, 100] ) -> [B, 100]
        c_n = torch.mul(wenc_n, p_n.unsqueeze(2).expand_as(wenc_n)).sum(dim=1)
        
        s_sn = torch.cat([torch.tensor([-10000000000.0] * c_n.size()[0]).view(-1, 1).to(device), self.sn_out(c_n)], dim=1)#add 0 nb of selected col is imposible
        
        # Penalty
        mL_hs = max(l_hs)
        for b, l_hs1 in enumerate(l_hs):
            #l_hs1 is the number of col in this table, ml_hs is the max nb of col in all tables
            s_sn[b, l_hs1 + 1:] = -10000000000.0#reset the padding value as -inf
        
        #s_sn = self.softmax_dim1(s_sn)

        return s_sn

class SCP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
        super(SCP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr


        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.sc_out = nn.Sequential(nn.Tanh(), nn.Linear(2 * hS, 1))#ouput的时候feature从200合并为1

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sc=False):
        # Encode
        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=None,
                        last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        mL_n = max(l_n)

        #   [bS, mL_hs, 100] * [bS, 100, mL_n] -> [bS, mL_hs, mL_n]
        att_h = torch.bmm(wenc_hs, self.W_att(wenc_n).transpose(1, 2))

        #   Penalty on blank parts
        for b, l_n1 in enumerate(l_n):
            att_h[b, :, l_n1:] = -10000000000.0

        p_n = self.softmax_dim2(att_h)
        if show_p_sc:
            # p = [b, hs, n]
            if p_n.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001, figsize=(12,3.5))
            # subplot(6,2,7)
            subplot2grid((7,2), (3, 0), rowspan=2)
            cla()
            _color='rgbkcm'
            _symbol='.......'
            for i_h in range(l_hs[0]):
                color_idx = i_h % len(_color)
                plot(p_n[0][i_h][:].data.numpy() - i_h, '--'+_symbol[color_idx]+_color[color_idx], ms=7)

            title('sc: p_n for each h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()



        #   p_n [ bS, mL_hs, mL_n]  -> [ bS, mL_hs, mL_n, 1]
        #   wenc_n [ bS, mL_n, 100] -> [ bS, 1, mL_n, 100]
        #   -> [bS, mL_hs, mL_n, 100] -> [bS, mL_hs, 100]
        c_n = torch.mul(p_n.unsqueeze(3), wenc_n.unsqueeze(1)).sum(dim=2)#做惩罚，是element wise的乘法，最后还是关于header的matrix

        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs)], dim=2)#在feature那一个维度进行concatenate
        s_sc = self.sc_out(vec).squeeze(2) # [bS, mL_hs, 1] -> [bS, mL_hs]

        # Penalty
        mL_hs = max(l_hs)
        for b, l_hs1 in enumerate(l_hs):
            #l_hs1 is the number of col in this table, ml_hs is the max nb of col in all tables
            s_sc[b, l_hs1:] = -10000000000.0#reset the padding value as -inf
                
        #s_sc = self.softmax_dim1(s_sc)
        
        return s_sc


class SAP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_agg_ops=6, old=False):
        super(SAP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
             
        self.mL_w = 3 #max select column number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.sa_out = nn.Sequential(nn.Linear(2*hS, hS), #original is nn.Linear(hS, hS)
                                    nn.Tanh(),
                                    nn.Linear(hS, n_agg_ops))  # Fixed number of aggregation operator.

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

        if old:
            # for backwoard compatibility
            self.W_c = nn.Linear(hS, hS)
            self.W_hs = nn.Linear(hS, hS)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_sn, pr_sc, show_p_sa=False):
        # Encode
        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=None,
                        last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        bS = len(l_hs)
        mL_n = max(l_n)

        # list, so one sample for each batch. need change
        wenc_hs_ob = [] # observed hs
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            real = [wenc_hs[b, col] for col in pr_sc[b]]#[selected col, dim]
            pad = (self.mL_w - pr_sn[b]) * [wenc_hs.new_zeros(wenc_hs.size()[-1])] # this padding could be wrong. Test with zero padding later. wn[b] is an int to indicate how many cols are selected
            wenc_hs_ob1 = torch.stack(real + pad) # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 3, dim] tensor.
        wenc_hs_ob = torch.stack(wenc_hs_ob) # list to tensor.
        wenc_hs_ob = wenc_hs_ob.to(device)
        
        # [B, 1, mL_n, dim] * [B, 3, dim, 1]
        #  -> [B, 3, mL_n, 1] -> [B, 3, mL_n]
        # multiplication bewteen NLq-tokens and  selected column
        att = torch.matmul(self.W_att(wenc_n).unsqueeze(1),
                              wenc_hs_ob.unsqueeze(3)
                              ).squeeze(3)

        # Penalty for blank part.
        mL_n = max(l_n)
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b, :, l_n1:] = -10000000000.0

        p = self.softmax_dim2(att)  # p( n| selected_col )
        if show_p_sa:
            # p = [b, hs, n]
            if p.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001)
            # subplot(6,2,7)
            subplot2grid((7,2), (5, 0), rowspan=2)
            cla()
            _color='rgbkcm'
            _symbol='.......'
            for i_wn in range(self.mL_w):
                color_idx = i_wn % len(_color)
                plot(p[0][i_wn][:].data.numpy() - i_wn, '--'+_symbol[color_idx]+_color[color_idx], ms=7)

            title('wo: p_n for selected h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()

        # [B, 1, mL_n, dim] * [B, 3, mL_n, 1]
        #  --> [B, 3, mL_n, dim]
        #  --> [B, 3, dim]
        c_n = torch.mul(wenc_n.unsqueeze(1), p.unsqueeze(3)).sum(dim=2)

        # [bS, 3, dim] -> [bS, 3, n_agg_ops]

        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob)], dim=2)
        s_sa = self.sa_out(vec)
        #s_sa[:, :, 2:4] = -10000000000.0
        #s_sa = self.softmax_dim2(s_sa) # loss 可能有问题，这里没有把out of range的selected col mask掉

        return s_sa


class WNP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
        super(WNP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.mL_w = 4  # max where condition number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att_h = nn.Linear(hS, 1)
        self.W_hidden = nn.Linear(hS, lS * hS)
        self.W_cell = nn.Linear(hS, lS * hS)

        self.W_att_n = nn.Linear(hS, 1)
        self.wn_out = nn.Sequential(nn.Linear(hS, hS),
                                    nn.Tanh(),
                                    nn.Linear(hS, self.mL_w + 1))  # max number (4 + 1)

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wn=False):
        # Encode

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, mL_hs, dim]

        bS = len(l_hs)
        mL_n = max(l_n)
        mL_hs = max(l_hs)
        # mL_h = max(l_hpu)

        #   (self-attention?) column Embedding?
        #   [B, mL_hs, 100] -> [B, mL_hs, 1] -> [B, mL_hs]
        att_h = self.W_att_h(wenc_hs).squeeze(2)

        #   Penalty
        for b, l_hs1 in enumerate(l_hs):
            if l_hs1 < mL_hs:
                att_h[b, l_hs1:] = -10000000000.0
        p_h = self.softmax_dim1(att_h)

        if show_p_wn:
            if p_h.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001);
            subplot(7,2,5)
            cla()
            plot(p_h[0].data.numpy(), '--rs', ms=7)
            title('wn: header_weight')
            grid(True)
            fig.canvas.draw()
            show()
            # input('Type Eenter to continue.')

        #   [B, mL_hs, 100] * [ B, mL_hs, 1] -> [B, mL_hs, 100] -> [B, 100]
        c_hs = torch.mul(wenc_hs, p_h.unsqueeze(2)).sum(1)

        #   [B, 100] --> [B, 2*100] Enlarge because there are two layers.
        hidden = self.W_hidden(c_hs)  # [B, 4, 200/2]
        hidden = hidden.view(bS, self.lS * 2, int(
            self.hS / 2))  # [4, B, 100/2] # number_of_layer_layer * (bi-direction) # lstm input convention.
        hidden = hidden.transpose(0, 1).contiguous()

        cell = self.W_cell(c_hs)  # [B, 4, 100/2]
        cell = cell.view(bS, self.lS * 2, int(self.hS / 2))  # [4, B, 100/2] the impact on view funciton is to reshape
        cell = cell.transpose(0, 1).contiguous()

        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=(hidden, cell),
                        last_only=False)  # [b, n, dim]

        att_n = self.W_att_n(wenc_n).squeeze(2)  # [B, max_len, 100] -> [B, max_len, 1] -> [B, max_len]

        #    Penalty
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att_n[b, l_n1:] = -10000000000.0
        p_n = self.softmax_dim1(att_n)

        if show_p_wn:
            if p_n.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001);
            subplot(7,2,6)
            cla()
            plot(p_n[0].data.numpy(), '--rs', ms=7)
            title('wn: nlu_weight')
            grid(True)
            fig.canvas.draw()

            show()
            # input('Type Enter to continue.')

        #    [B, mL_n, 100] *([B, mL_n] -> [B, mL_n, 1] -> [B, mL_n, 100] ) -> [B, 100]
        c_n = torch.mul(wenc_n, p_n.unsqueeze(2).expand_as(wenc_n)).sum(dim=1)
        s_wn = self.wn_out(c_n)# [B, 1+4]
        
        # Penalty
        mL_hs = max(l_hs)
        for b, l_hs1 in enumerate(l_hs):
            #l_hs1 is the number of col in this table, ml_hs is the max nb of col in all tables
            s_wn[b, l_hs1 + 1:] = -10000000000.0#reset the padding value as -inf include 0 so it need to plus 1
        
        #s_wn = self.softmax_dim1(s_wn) # [B, 1 + 4]
        
        s_wn[:, 0] = -10000000000.0

        return s_wn

class WRP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, rpnb=3):
        super(WRP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        
        self.rpnb = rpnb
        

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)


        self.W_att_n = nn.Linear(hS, 1)
        self.wr_out = nn.Sequential(nn.Linear(hS, hS),
                                    nn.Tanh(),
                                    nn.Linear(hS, self.rpnb - 1))  # it is only two, but we need to append one at the first , so it will be 0: "and", 1: "or", after appending it will become 0: "", 1: "and", 2: "or"

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, pr_wn, show_p_wr=False):
        # Encode

        wenc_n = encode(self.enc_n, wemb_n, l_n)  # [b, mL_n, dim]

        bS = len(l_n)
        mL_n = max(l_n)

        #   (self-attention?) column Embedding?
        #   [B, mL_n, dim] -> [B, mL_n, dim] -> [B, mL_hs]
        att_n = self.W_att_n(wenc_n).squeeze(2)

        #   Penalty
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att_n[b, l_n1:] = -10000000000.0
        p_n = self.softmax_dim1(att_n)

        if show_p_wr:
            if p_n.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001);
            subplot(7,2,5)
            cla()
            plot(p_h[0].data.numpy(), '--rs', ms=7)
            title('wn: header_weight')
            grid(True)
            fig.canvas.draw()
            show()
            # input('Type Eenter to continue.')

        #   [B, mL_n, dim] * [ B, mL_n, 1] -> [B, mL_n, dim] -> [B, dim]
        c_n = torch.mul(wenc_n, p_n.unsqueeze(2)).sum(1)
        
        c_n = self.wr_out(c_n)# [B, dim] -> [B, 2]
        
        s_wr = torch.cat([torch.tensor([-10000000000.0] * c_n.size()[0]).view(-1, 1).to(device), c_n], dim=1)# add a padding to represent "" at the front

        for b, wn1 in enumerate(pr_wn):
            if wn1 <= 1:
                s_wr[b][1:] = -20000000000.0
                
        #s_wr = self.softmax_dim1(s_wr)
        
        return s_wr
    
class HRPC(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
        super(HRPC, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr


        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att_h = nn.Linear(hS, 1)
        self.W_hidden = nn.Linear(hS, lS * hS)
        self.W_cell = nn.Linear(hS, lS * hS)

        self.W_att_n = nn.Linear(hS, 1)
        self.hrpc_out = nn.Sequential(nn.Linear(hS, hS),
                                    nn.Tanh(),
                                    nn.Linear(hS, 2))  # index 0 indicates not having repeated cols, index 1 indicates having repeated cols

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wr, show_p_hrpc=False):
        # Encode

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, mL_hs, dim]

        bS = len(l_hs)
        mL_n = max(l_n)
        mL_hs = max(l_hs)
        # mL_h = max(l_hpu)

        #   (self-attention?) column Embedding?
        #   [B, mL_hs, 100] -> [B, mL_hs, 1] -> [B, mL_hs]
        att_h = self.W_att_h(wenc_hs).squeeze(2)

        #   Penalty
        for b, l_hs1 in enumerate(l_hs):
            if l_hs1 < mL_hs:
                att_h[b, l_hs1:] = -10000000000.0
        p_h = self.softmax_dim1(att_h)

        if show_p_hrpc:
            if p_h.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001);
            subplot(7,2,5)
            cla()
            plot(p_h[0].data.numpy(), '--rs', ms=7)
            title('wn: header_weight')
            grid(True)
            fig.canvas.draw()
            show()
            # input('Type Eenter to continue.')

        #   [B, mL_hs, 100] * [ B, mL_hs, 1] -> [B, mL_hs, 100] -> [B, 100]
        c_hs = torch.mul(wenc_hs, p_h.unsqueeze(2)).sum(1)

        #   [B, 100] --> [B, 2*100] Enlarge because there are two layers.
        hidden = self.W_hidden(c_hs)  # [B, 2 * 100]
        hidden = hidden.view(bS, self.lS * 2, int(
            self.hS / 2))  # [4, B, 100/2] # number_of_layer_layer * (bi-direction) # lstm input convention.
        hidden = hidden.transpose(0, 1).contiguous()

        cell = self.W_cell(c_hs)  # [B, 4, 100/2]
        cell = cell.view(bS, self.lS * 2, int(self.hS / 2))  # [4, B, 100/2] the impact on view funciton is to reshape
        cell = cell.transpose(0, 1).contiguous()

        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=(hidden, cell),
                        last_only=False)  # [b, n, dim]

        att_n = self.W_att_n(wenc_n).squeeze(2)  # [B, max_len, 100] -> [B, max_len, 1] -> [B, max_len]

        #    Penalty
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att_n[b, l_n1:] = -10000000000.0
        p_n = self.softmax_dim1(att_n)

        if show_p_hrpc:
            if p_n.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001);
            subplot(7,2,6)
            cla()
            plot(p_n[0].data.numpy(), '--rs', ms=7)
            title('wn: nlu_weight')
            grid(True)
            fig.canvas.draw()

            show()
            # input('Type Enter to continue.')

        #    [B, mL_n, 100] *([B, mL_n] -> [B, mL_n, 1] -> [B, mL_n, 100] ) -> [B, 100]
        c_n = torch.mul(wenc_n, p_n.unsqueeze(2).expand_as(wenc_n)).sum(dim=1)
        s_hrpc = self.hrpc_out(c_n) # [B, 2]
        
        
        for B, wr1 in enumerate(wr):
            if wr1 == 0:# if wr is not "or" it means it will not have repeated column
                s_hrpc[B][1] = -10000000000.0
                
        #s_hrpc = self.softmax_dim1(c_n) # [B, 2]

        return s_hrpc

class WCP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
        super(WCP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.W_out = nn.Sequential(
            nn.Linear(2 * hS, hS),
            nn.Tanh(), 
            nn.Linear(hS, 1)
        )

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wc=False, penalty=True):
        # Encode
        wenc_n = encode(self.enc_n, wemb_n, l_n,
                        return_hidden=False,
                        hc0=None,
                        last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        # attention
        # wenc = [bS, mL, hS]
        # att = [bS, mL_hs, mL_n]
        # att[b, i_h, j_n] = p(j_n| i_h)
        att = torch.bmm(wenc_hs, self.W_att(wenc_n).transpose(1, 2))

        # penalty to blank part.
        mL_n = max(l_n)
        for b_n, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b_n, :, l_n1:] = -10000000000.0

        # make p(j_n | i_h)
        p = self.softmax_dim2(att)

        if show_p_wc:
            # p = [b, hs, n]
            if p.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001);
            # subplot(6,2,7)
            subplot2grid((7,2), (3, 1), rowspan=2)
            cla()
            _color='rgbkcm'
            _symbol='.......'
            for i_h in range(l_hs[0]):
                color_idx = i_h % len(_color)
                plot(p[0][i_h][:].data.numpy() - i_h, '--'+_symbol[color_idx]+_color[color_idx], ms=7)

            title('wc: p_n for each h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()
        # max nlu context vectors
        # [bS, mL_hs, mL_n]*[bS, mL_hs, mL_n]
        wenc_n = wenc_n.unsqueeze(1)  # [ b, n, dim] -> [b, 1, n, dim]
        p = p.unsqueeze(3)  # [b, hs, n] -> [b, hs, n, 1]
        c_n = torch.mul(wenc_n, p).sum(2)  # -> [b, hs, dim], c_n for each header.

        y = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs)], dim=2)  # [b, hs, 2*dim]
        score = self.W_out(y).squeeze(2)  # [b, hs]
        

        if penalty:
            for b, l_hs1 in enumerate(l_hs):
                score[b, l_hs1:] = -10000000000.0
                
        #score = self.softmax_dim1(score)
        
        return score


class WOP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_cond_ops=4):
        super(WOP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr

        self.mL_w = 4 # max where condition number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.wo_out = nn.Sequential(
            nn.Linear(2*hS, hS),
            nn.Tanh(),
            nn.Linear(hS, n_cond_ops)
        )

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn, wc, wenc_n=None, show_p_wo=False):
        # Encode
        if not wenc_n:
            wenc_n = encode(self.enc_n, wemb_n, l_n,
                            return_hidden=False,
                            hc0=None,
                            last_only=False)  # [b, n, dim]

        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        bS = len(l_hs)
        # wn


        wenc_hs_ob = [] # observed hs
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            #print(wc)
            real = [wenc_hs[b, col] for col in wc[b]]
            pad = (self.mL_w - wn[b]) * [wenc_hs.new_zeros(wenc_hs.size()[-1])] # this padding could be wrong. Test with zero padding later.
            wenc_hs_ob1 = torch.stack(real + pad) # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 4, dim] tensor.
        '''
        print('wn: ', wn)
        print('wc: ', wc)
        print('wenc_hs_ob', wenc_hs_ob)
        '''
        wenc_hs_ob = torch.stack(wenc_hs_ob) # list to tensor.
        wenc_hs_ob = wenc_hs_ob.to(device)

        # [B, 1, mL_n, dim] * [B, 4, dim, 1]
        #  -> [B, 4, mL_n, 1] -> [B, 4, mL_n]
        # multiplication bewteen NLq-tokens and  selected column
        att = torch.matmul(self.W_att(wenc_n).unsqueeze(1),
                              wenc_hs_ob.unsqueeze(3)
                              ).squeeze(3)

        # Penalty for blank part.
        mL_n = max(l_n)
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b, :, l_n1:] = -10000000000.0

        p = self.softmax_dim2(att)  # p( n| selected_col )
        if show_p_wo:
            # p = [b, hs, n]
            if p.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001)
            # subplot(6,2,7)
            subplot2grid((7,2), (5, 0), rowspan=2)
            cla()
            _color='rgbkcm'
            _symbol='.......'
            for i_wn in range(self.mL_w):
                color_idx = i_wn % len(_color)
                plot(p[0][i_wn][:].data.numpy() - i_wn, '--'+_symbol[color_idx]+_color[color_idx], ms=7)

            title('wo: p_n for selected h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()

        # [B, 1, mL_n, dim] * [B, 4, mL_n, 1]
        #  --> [B, 4, mL_n, dim]
        #  --> [B, 4, dim]
        c_n = torch.mul(wenc_n.unsqueeze(1), p.unsqueeze(3)).sum(dim=2)

        # [bS, 5-1, dim] -> [bS, 5-1, 4]

        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob)], dim=2)
        s_wo = self.wo_out(vec)
        s_wo[:, :, 3] = -10000000000.0
        #s_wo = self.softmax_dim2(s_wo)

        return s_wo

class WVP_se(nn.Module):
    """
    Discriminative model
    Get start and end.
    Here, classifier for [ [투수], [팀1], [팀2], [연도], ...]
    Input:      Encoded nlu & selected column.
    Algorithm: Encoded nlu & selected column. -> classifier -> mask scores -> ...
    """
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_cond_ops=4, old=False):
        super(WVP_se, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.n_cond_ops = n_cond_ops

        self.mL_w = 4  # max where condition number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.W_op = nn.Linear(n_cond_ops, hS)

        # self.W_n = nn.Linear(hS, hS)
        if old:
            self.wv_out =  nn.Sequential(
            nn.Linear(4 * hS, 2)
            )
        else:
            self.wv_out = nn.Sequential(
                nn.Linear(4 * hS, hS),
                nn.GroupNorm(2, 4),
                nn.Tanh(),
                nn.Linear(hS, 1)
            )
        # self.wv_out = nn.Sequential(
        #     nn.Linear(3 * hS, hS),
        #     nn.Tanh(),
        #     nn.Linear(hS, self.gdkL)
        # )

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)
        self.softmax_dim3 = nn.Softmax(dim=3)

    def forward(self, wemb_v, l_npu, l_token, wemb_hpu, l_hpu, l_hs, wn, wc, wo, wenc_n=None, show_p_wv=False):

        # Encode
        if not wenc_n:
            wenc_n = encode_npu(self.enc_n, wemb_v, l_npu, l_token)
        
        #print('wenc_n: ', wenc_n.size(), '; wemb_n: ', wemb_n.size(), '; l_n: ', l_n)#l_n is how many 字 in this each question
        
        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        bS = len(l_hs)

        wenc_hs_ob = []  # observed hs

        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            real = [wenc_hs[b, col] for col in wc[b]]
            pad = (self.mL_w - wn[b]) * [wenc_hs[b, 0]]  # this padding could be wrong. Test with zero padding later.
            wenc_hs_ob1 = torch.stack(real + pad)  # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 4, dim] tensor.
        wenc_hs_ob = torch.stack(wenc_hs_ob)  # list to tensor.
        wenc_hs_ob = wenc_hs_ob.to(device)
        #print('wenc_hs_ob: ', wenc_hs_ob.size())

        # 学！结尾词对开头词的attention
        # Column attention
        # [B, 1, mL_n, dim] * [B, 4, dim, 1]
        #  -> [B, 4, mL_n, 1] -> [B, 4, mL_n]
        # multiplication bewteen NLq-tokens and  selected column
        att = torch.matmul(self.W_att(wenc_n).unsqueeze(1),
                           wenc_hs_ob.unsqueeze(3)
                           ).squeeze(3)
        # Penalty for blank part.
        mL_n = max(l_token)#字的长度
        for b, l_n1 in enumerate(l_token):
            if l_n1 < mL_n:
                att[b, :, l_n1:] = -10000000000.0

        p = self.softmax_dim2(att)  # p( n| selected_col )

        if show_p_wv:
            # p = [b, hs, n]
            if p.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001)
            # subplot(6,2,7)
            subplot2grid((7,2), (5, 1), rowspan=2)
            cla()
            _color='rgbkcm'
            _symbol='.......'
            for i_wn in range(self.mL_w):
                color_idx = i_wn % len(_color)
                plot(p[0][i_wn][:].data.numpy() - i_wn, '--'+_symbol[color_idx]+_color[color_idx], ms=7)

            title('wv: p_n for selected h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()


        # [B, 1, mL_n, dim] * [B, 4, mL_n, 1]
        #  --> [B, 4, mL_n, dim]
        #  --> [B, 4, dim]
        c_n = torch.mul(wenc_n.unsqueeze(1), p.unsqueeze(3)).sum(dim=2)
        
        #print('c_n: ', c_n.size())

        # Select observed headers only.
        # Also generate one_hot vector encoding info of the operator
        # [B, 4, dim]
        wenc_op = []
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            wenc_op1 = torch.zeros(self.mL_w, self.n_cond_ops)
            wo1 = wo[b]
            idx_scatter = []
            l_wo1 = len(wo1)
            for i_wo11 in range(self.mL_w):
                if i_wo11 < l_wo1:
                    wo11 = wo1[i_wo11]
                    idx_scatter.append([int(wo11)])
                else:
                    idx_scatter.append([0]) # not used anyway

            wenc_op1 = wenc_op1.scatter(1, torch.tensor(idx_scatter), 1)

            wenc_op.append(wenc_op1)
        
        #print(wenc_op[0].size())
        # list to [B, 4, dim] tensor.
        wenc_op = torch.stack(wenc_op)  # list to tensor.
        wenc_op = wenc_op.to(device)
        #print('wo: ', wo)
        #print('wenc_op: ', wenc_op.size())

        # Now after concat, calculate logits for each token
        # [bS, 5-1, 3*hS] = [bS, 4, 300]
        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob), self.W_op(wenc_op)], dim=2)
        #print('vec: ', vec.size())

        # Make extended vector based on encoded nl token containing column and operator information.
        # wenc_n = [bS, mL, 100]
        # vec2 = [bS, 4, mL, 400]
        vec1e = vec.unsqueeze(2).expand(-1,-1, mL_n, -1) # [bS, 4, 1, 300]  -> [bS, 4, mL, 300]
        wenc_ne = wenc_n.unsqueeze(1).expand(-1, 4, -1, -1) # [bS, 1, mL, 100] -> [bS, 4, mL, 100]
        vec2 = torch.cat( [vec1e, wenc_ne], dim=3)
        #print('vec1e: ', vec1e.size())
        #print('vec2: ', vec2.size())
        #print('---------------------------------------------------------------------------------------------------')
        # now make logits
        s_wv1 = self.wv_out(vec2).squeeze(3) # [bS, 4, mL, 400] -> [bS, 4, mL, 1] -> [bS, 4, mL]
        
        # penalty for spurious tokens
        for b, l_n1 in enumerate(l_token):
            if l_n1 < mL_n:
                s_wv1[b, :, l_n1:] = -10000000000.0
                
        #s_wv = self.softmax_dim2(s_wv)
        #print('s_wv: ', [e[0] for e in s_wv[0][0].tolist()])
                
        return s_wv1
    
class WVP_se2(nn.Module):
    """
    Discriminative model
    Get start and end.
    Here, classifier for [ [투수], [팀1], [팀2], [연도], ...]
    Input:      Encoded nlu & selected column.
    Algorithm: Encoded nlu & selected column. -> classifier -> mask scores -> ...
    """
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_cond_ops=4, old=False):
        super(WVP_se2, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.n_cond_ops = n_cond_ops

        self.mL_w = 4  # max where condition number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.W_op = nn.Linear(n_cond_ops, hS)

        # self.W_n = nn.Linear(hS, hS)
        if old:
            self.wv_out =  nn.Sequential(
            nn.Linear(4 * hS, 2)
            )
        else:
            self.wv_out = nn.Sequential(
                nn.Linear(4 * hS, hS),
                nn.GroupNorm(2, 4),
                nn.Tanh(),
                nn.Linear(hS, 1)
            )
        # self.wv_out = nn.Sequential(
        #     nn.Linear(3 * hS, hS),
        #     nn.Tanh(),
        #     nn.Linear(hS, self.gdkL)
        # )

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)
        self.softmax_dim3 = nn.Softmax(dim=3)

    def forward(self, wemb_v, l_npu, l_token, wemb_hpu, l_hpu, l_hs, wn, wc, wo, wvi1, mvl,  wenc_n=None, show_p_wv=False):

        # Encode
        if not wenc_n:
            wenc_n = encode_npu(self.enc_n, wemb_v, l_npu, l_token)
        
        #print('wenc_n: ', wenc_n.size(), '; wemb_n: ', wemb_n.size(), '; l_n: ', l_n)#l_n is how many 字 in this each question
        
        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        bS = len(l_hs)

        wenc_hs_ob = []  # observed hs

        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            real = [wenc_hs[b, col] for col in wc[b]]
            pad = (self.mL_w - wn[b]) * [wenc_hs[b, 0]]  # this padding could be wrong. Test with zero padding later.
            wenc_hs_ob1 = torch.stack(real + pad)  # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 4, dim] tensor.
        wenc_hs_ob = torch.stack(wenc_hs_ob)  # list to tensor.
        wenc_hs_ob = wenc_hs_ob.to(device)
        #print('wenc_hs_ob: ', wenc_hs_ob.size())

        # 学！结尾词对开头词的attention
        # Column attention
        # [B, 1, mL_n, dim] * [B, 4, dim, 1]
        #  -> [B, 4, mL_n, 1] -> [B, 4, mL_n]
        # multiplication bewteen NLq-tokens and  selected column
        att = torch.matmul(self.W_att(wenc_n).unsqueeze(1),
                           wenc_hs_ob.unsqueeze(3)
                           ).squeeze(3)
        # Penalty for blank part.
        mL_n = max(l_token)#字的长度
        for b, l_n1 in enumerate(l_token):
            if l_n1 < mL_n:
                att[b, :, l_n1:] = -10000000000.0

        p = self.softmax_dim2(att)  # p( n| selected_col )

        if show_p_wv:
            # p = [b, hs, n]
            if p.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001)
            # subplot(6,2,7)
            subplot2grid((7,2), (5, 1), rowspan=2)
            cla()
            _color='rgbkcm'
            _symbol='.......'
            for i_wn in range(self.mL_w):
                color_idx = i_wn % len(_color)
                plot(p[0][i_wn][:].data.numpy() - i_wn, '--'+_symbol[color_idx]+_color[color_idx], ms=7)

            title('wv: p_n for selected h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()


        # [B, 1, mL_n, dim] * [B, 4, mL_n, 1]
        #  --> [B, 4, mL_n, dim]
        #  --> [B, 4, dim]
        c_n = torch.mul(wenc_n.unsqueeze(1), p.unsqueeze(3)).sum(dim=2)
        
        #print('c_n: ', c_n.size())

        # Select observed headers only.
        # Also generate one_hot vector encoding info of the operator
        # [B, 4, dim]
        wenc_op = []
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            wenc_op1 = torch.zeros(self.mL_w, self.n_cond_ops)
            wo1 = wo[b]
            idx_scatter = []
            l_wo1 = len(wo1)
            for i_wo11 in range(self.mL_w):
                if i_wo11 < l_wo1:
                    wo11 = wo1[i_wo11]
                    idx_scatter.append([int(wo11)])
                else:
                    idx_scatter.append([0]) # not used anyway

            wenc_op1 = wenc_op1.scatter(1, torch.tensor(idx_scatter), 1)

            wenc_op.append(wenc_op1)
        
        #print(wenc_op[0].size())
        # list to [B, 4, dim] tensor.
        wenc_op = torch.stack(wenc_op)  # list to tensor.
        wenc_op = wenc_op.to(device)
        #print('wo: ', wo)
        #print('wenc_op: ', wenc_op.size())

        # Now after concat, calculate logits for each token
        # [bS, 5-1, 3*hS] = [bS, 4, 300]
        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob), self.W_op(wenc_op)], dim=2)
        #print('vec: ', vec.size())

        # Make extended vector based on encoded nl token containing column and operator information.
        # wenc_n = [bS, mL, 100]
        # vec2 = [bS, 4, mL, 400]
        vec1e = vec.unsqueeze(2).expand(-1,-1, mvl, -1) # [bS, 4, 1, 300]  -> [bS, 4, mvl, 300]
        
        wenc_vs_ob = [] # observed hs
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            big_real = []
            for wc in range(wn[b]):
                real = []
                for idx in range(wvi1[b][wc], l_token[b]):
                    real.append(wenc_n[b, idx])
                real = real[:mvl]
                pad = (mvl - len(real)) * [wenc_n.new_zeros(wenc_n.size()[-1])] # this padding could be wrong. Test with zero padding later. wn[b] is an int to indicate how many cols are selected
                big_real1 = torch.stack(real + pad) # It is not used in the loss function.
                big_real.append(big_real1)
            big_pad = (self.mL_w - wn[b]) * [wenc_n.new_zeros(mvl, wenc_n.size()[-1])]
            wenc_vs_ob1 = torch.stack(big_real + big_pad)
            wenc_vs_ob.append(wenc_vs_ob1)

        # list to [B, 4, mvl, dim] tensor.
        wenc_vs_ob = torch.stack(wenc_vs_ob) # list to tensor.
        wenc_ne = wenc_vs_ob.to(device)
        
        vec2 = torch.cat( [vec1e, wenc_ne], dim=3)
        #print('vec1e: ', vec1e.size())
        #print('vec2: ', vec2.size())
        #print('---------------------------------------------------------------------------------------------------')
        # now make logits
        s_wv2 = self.wv_out(vec2).squeeze(3) # [bS, 4, mvl, 400] -> [bS, 4, mvl, 1] -> [bS, 4, mvl]
        #print(s_wv2)
        
        # penalty for spurious tokens
        for b, l_n1 in enumerate(l_token):
            for wc in range(wn[b]):
                if l_n1 - wvi1[b][wc] < mvl:
                    s_wv2[b, wc, l_n1 - wvi1[b][wc]:] = -10000000000.0#mask all of invalid words
            s_wv2[b, wn[b]:] = -10000000000.0
                
        #s_wv = self.softmax_dim2(s_wv)
        #print('s_wv: ', [e[0] for e in s_wv[0][0].tolist()])
                
        return s_wv2

'''
draft
l is the index of the last token
pn is the prob of token can be choosen when n is the token's index
if wv1 = [p0, p1, p2, p3, p4, ..., pl] wv3 = [p0, p1, p2, ..., pl]
method 1: if max(wv1) >= max(wv3) then choose wv1 else choose wv3
method 2: if max(wv1) - secondlarge(wv1) >= max(wv3) - secondlarge(wv3) then choose wv1 else choose wv3
method 3: if max(wv1) - mean(wv1) >= max(wv3) - mean(wv3) then choose wv1 else choose wv3
'''

class WVP_se3(nn.Module):
    """
    Discriminative model
    Get start and end.
    Here, classifier for [ [투수], [팀1], [팀2], [연도], ...]
    Input:      Encoded nlu & selected column.
    Algorithm: Encoded nlu & selected column. -> classifier -> mask scores -> ...
    """
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_cond_ops=4, old=False):
        super(WVP_se3, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.n_cond_ops = n_cond_ops

        self.mL_w = 4  # max where condition number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.W_op = nn.Linear(n_cond_ops, hS)

        # self.W_n = nn.Linear(hS, hS)
        if old:
            self.wv_out =  nn.Sequential(
            nn.Linear(4 * hS, 2)
            )
        else:
            self.wv_out = nn.Sequential(
                nn.Linear(4 * hS, hS),
                nn.GroupNorm(2, 4),
                nn.Tanh(),
                nn.Linear(hS, 1)
            )
        # self.wv_out = nn.Sequential(
        #     nn.Linear(3 * hS, hS),
        #     nn.Tanh(),
        #     nn.Linear(hS, self.gdkL)
        # )

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)
        self.softmax_dim3 = nn.Softmax(dim=3)

    def forward(self, wemb_v, l_npu, l_token, wemb_hpu, l_hpu, l_hs, wn, wc, wo, wenc_n=None, show_p_wv=False):

        # Encode
        if not wenc_n:
            wenc_n = encode_npu(self.enc_n, wemb_v, l_npu, l_token)
        
        #print('wenc_n: ', wenc_n.size(), '; wemb_n: ', wemb_n.size(), '; l_n: ', l_n)#l_n is how many 字 in this each question
        
        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        bS = len(l_hs)

        wenc_hs_ob = []  # observed hs

        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            real = [wenc_hs[b, col] for col in wc[b]]
            pad = (self.mL_w - wn[b]) * [wenc_hs[b, 0]]  # this padding could be wrong. Test with zero padding later.
            wenc_hs_ob1 = torch.stack(real + pad)  # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 4, dim] tensor.
        wenc_hs_ob = torch.stack(wenc_hs_ob)  # list to tensor.
        wenc_hs_ob = wenc_hs_ob.to(device)
        #print('wenc_hs_ob: ', wenc_hs_ob.size())

        # 学！结尾词对开头词的attention
        # Column attention
        # [B, 1, mL_n, dim] * [B, 4, dim, 1]
        #  -> [B, 4, mL_n, 1] -> [B, 4, mL_n]
        # multiplication bewteen NLq-tokens and  selected column
        att = torch.matmul(self.W_att(wenc_n).unsqueeze(1),
                           wenc_hs_ob.unsqueeze(3)
                           ).squeeze(3)
        # Penalty for blank part.
        mL_n = max(l_token)#字的长度
        for b, l_n1 in enumerate(l_token):
            if l_n1 < mL_n:
                att[b, :, l_n1:] = -10000000000.0

        p = self.softmax_dim2(att)  # p( n| selected_col )

        if show_p_wv:
            # p = [b, hs, n]
            if p.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001)
            # subplot(6,2,7)
            subplot2grid((7,2), (5, 1), rowspan=2)
            cla()
            _color='rgbkcm'
            _symbol='.......'
            for i_wn in range(self.mL_w):
                color_idx = i_wn % len(_color)
                plot(p[0][i_wn][:].data.numpy() - i_wn, '--'+_symbol[color_idx]+_color[color_idx], ms=7)

            title('wv: p_n for selected h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()


        # [B, 1, mL_n, dim] * [B, 4, mL_n, 1]
        #  --> [B, 4, mL_n, dim]
        #  --> [B, 4, dim]
        c_n = torch.mul(wenc_n.unsqueeze(1), p.unsqueeze(3)).sum(dim=2)
        
        #print('c_n: ', c_n.size())

        # Select observed headers only.
        # Also generate one_hot vector encoding info of the operator
        # [B, 4, dim]
        wenc_op = []
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            wenc_op1 = torch.zeros(self.mL_w, self.n_cond_ops)
            wo1 = wo[b]
            idx_scatter = []
            l_wo1 = len(wo1)
            for i_wo11 in range(self.mL_w):
                if i_wo11 < l_wo1:
                    wo11 = wo1[i_wo11]
                    idx_scatter.append([int(wo11)])
                else:
                    idx_scatter.append([0]) # not used anyway

            wenc_op1 = wenc_op1.scatter(1, torch.tensor(idx_scatter), 1)

            wenc_op.append(wenc_op1)
        
        #print(wenc_op[0].size())
        # list to [B, 4, dim] tensor.
        wenc_op = torch.stack(wenc_op)  # list to tensor.
        wenc_op = wenc_op.to(device)
        #print('wo: ', wo)
        #print('wenc_op: ', wenc_op.size())

        # Now after concat, calculate logits for each token
        # [bS, 5-1, 3*hS] = [bS, 4, 300]
        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob), self.W_op(wenc_op)], dim=2)
        #print('vec: ', vec.size())

        # Make extended vector based on encoded nl token containing column and operator information.
        # wenc_n = [bS, mL, 100]
        # vec2 = [bS, 4, mL, 400]
        vec1e = vec.unsqueeze(2).expand(-1,-1, mL_n, -1) # [bS, 4, 1, 300]  -> [bS, 4, mL, 300]
        wenc_ne = wenc_n.unsqueeze(1).expand(-1, 4, -1, -1) # [bS, 1, mL, 100] -> [bS, 4, mL, 100]
        vec2 = torch.cat( [vec1e, wenc_ne], dim=3)
        #print('vec1e: ', vec1e.size())
        #print('vec2: ', vec2.size())
        #print('---------------------------------------------------------------------------------------------------')
        # now make logits
        s_wv3 = self.wv_out(vec2).squeeze(3) # [bS, 4, mL, 400] -> [bS, 4, mL, 1] -> [bS, 4, mL]
        
        # penalty for spurious tokens
        for b, l_n1 in enumerate(l_token):
            if l_n1 < mL_n:
                s_wv3[b, :, l_n1:] = -10000000000.0
                
        #s_wv = self.softmax_dim2(s_wv)
        #print('s_wv: ', [e[0] for e in s_wv[0][0].tolist()])
                
        return s_wv3
    
class WVP_se4(nn.Module):
    """
    Discriminative model
    Get start and end.
    Here, classifier for [ [투수], [팀1], [팀2], [연도], ...]
    Input:      Encoded nlu & selected column.
    Algorithm: Encoded nlu & selected column. -> classifier -> mask scores -> ...
    """
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_cond_ops=4, old=False):
        super(WVP_se4, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.n_cond_ops = n_cond_ops

        self.mL_w = 4  # max where condition number

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.W_op = nn.Linear(n_cond_ops, hS)

        # self.W_n = nn.Linear(hS, hS)
        if old:
            self.wv_out =  nn.Sequential(
            nn.Linear(4 * hS, 2)
            )
        else:
            self.wv_out = nn.Sequential(
                nn.Linear(4 * hS, hS),
                nn.GroupNorm(2, 4),
                nn.Tanh(),
                nn.Linear(hS, 1)
            )
        # self.wv_out = nn.Sequential(
        #     nn.Linear(3 * hS, hS),
        #     nn.Tanh(),
        #     nn.Linear(hS, self.gdkL)
        # )

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)
        self.softmax_dim3 = nn.Softmax(dim=3)

    def forward(self, wemb_v, l_npu, l_token, wemb_hpu, l_hpu, l_hs, wn, wc, wo, wvi3, mvl, wenc_n=None, show_p_wv=False):

        # Encode
        if not wenc_n:
            wenc_n = encode_npu(self.enc_n, wemb_v, l_npu, l_token)
        
        #print('wenc_n: ', wenc_n.size(), '; wemb_n: ', wemb_n.size(), '; l_n: ', l_n)#l_n is how many 字 in this each question
        
        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)  # [b, hs, dim]

        bS = len(l_hs)

        wenc_hs_ob = []  # observed hs

        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            real = [wenc_hs[b, col] for col in wc[b]]
            pad = (self.mL_w - wn[b]) * [wenc_hs[b, 0]]  # this padding could be wrong. Test with zero padding later.
            wenc_hs_ob1 = torch.stack(real + pad)  # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 4, dim] tensor.
        wenc_hs_ob = torch.stack(wenc_hs_ob)  # list to tensor.
        wenc_hs_ob = wenc_hs_ob.to(device)
        #print('wenc_hs_ob: ', wenc_hs_ob.size())

        # 学！结尾词对开头词的attention
        # Column attention
        # [B, 1, mL_n, dim] * [B, 4, dim, 1]
        #  -> [B, 4, mL_n, 1] -> [B, 4, mL_n]
        # multiplication bewteen NLq-tokens and  selected column
        att = torch.matmul(self.W_att(wenc_n).unsqueeze(1),
                           wenc_hs_ob.unsqueeze(3)
                           ).squeeze(3)
        # Penalty for blank part.
        mL_n = max(l_token)#字的长度
        for b, l_n1 in enumerate(l_token):
            if l_n1 < mL_n:
                att[b, :, l_n1:] = -10000000000.0

        p = self.softmax_dim2(att)  # p( n| selected_col )

        if show_p_wv:
            # p = [b, hs, n]
            if p.shape[0] != 1:
                raise Exception("Batch size should be 1.")
            fig=figure(2001)
            # subplot(6,2,7)
            subplot2grid((7,2), (5, 1), rowspan=2)
            cla()
            _color='rgbkcm'
            _symbol='.......'
            for i_wn in range(self.mL_w):
                color_idx = i_wn % len(_color)
                plot(p[0][i_wn][:].data.numpy() - i_wn, '--'+_symbol[color_idx]+_color[color_idx], ms=7)

            title('wv: p_n for selected h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()


        # [B, 1, mL_n, dim] * [B, 4, mL_n, 1]
        #  --> [B, 4, mL_n, dim]
        #  --> [B, 4, dim]
        c_n = torch.mul(wenc_n.unsqueeze(1), p.unsqueeze(3)).sum(dim=2)
        
        #print('c_n: ', c_n.size())

        # Select observed headers only.
        # Also generate one_hot vector encoding info of the operator
        # [B, 4, dim]
        wenc_op = []
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            wenc_op1 = torch.zeros(self.mL_w, self.n_cond_ops)
            wo1 = wo[b]
            idx_scatter = []
            l_wo1 = len(wo1)
            for i_wo11 in range(self.mL_w):
                if i_wo11 < l_wo1:
                    wo11 = wo1[i_wo11]
                    idx_scatter.append([int(wo11)])
                else:
                    idx_scatter.append([0]) # not used anyway

            wenc_op1 = wenc_op1.scatter(1, torch.tensor(idx_scatter), 1)

            wenc_op.append(wenc_op1)
        
        #print(wenc_op[0].size())
        # list to [B, 4, dim] tensor.
        wenc_op = torch.stack(wenc_op)  # list to tensor.
        wenc_op = wenc_op.to(device)
        #print('wo: ', wo)
        #print('wenc_op: ', wenc_op.size())

        # Now after concat, calculate logits for each token
        # [bS, 5-1, 3*hS] = [bS, 4, 300]
        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob), self.W_op(wenc_op)], dim=2)
        #print('vec: ', vec.size())

        # Make extended vector based on encoded nl token containing column and operator information.
        # wenc_n = [bS, mL, 100]
        # vec2 = [bS, 4, mL, 400]
        vec1e = vec.unsqueeze(2).expand(-1,-1, mvl, -1) # [bS, 4, 1, 300]  -> [bS, 4, mvl, 300]
        
        wenc_vs_ob = [] # observed hs
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            big_real = []
            for wc in range(wn[b]):
                real = []
                for idx in range(max(0, wvi3[b][wc] - (mvl - 1)), wvi3[b][wc] + 1):
                    real.append(wenc_n[b, idx])
                pad = (mvl - len(real)) * [wenc_n.new_zeros(wenc_n.size()[-1])] # this padding could be wrong. Test with zero padding later. wn[b] is an int to indicate how many cols are selected
                big_real1 = torch.stack(pad + real) # It is not used in the loss function.
                big_real.append(big_real1)
            big_pad = (self.mL_w - wn[b]) * [wenc_n.new_zeros(mvl, wenc_n.size()[-1])]
            wenc_vs_ob1 = torch.stack(big_real + big_pad)
            wenc_vs_ob.append(wenc_vs_ob1)

        # list to [B, 4, mvl, dim] tensor.
        wenc_vs_ob = torch.stack(wenc_vs_ob) # list to tensor.
        wenc_ne = wenc_vs_ob.to(device)
        
        vec2 = torch.cat( [vec1e, wenc_ne], dim=3)
        #print('vec1e: ', vec1e.size())
        #print('vec2: ', vec2.size())
        #print('---------------------------------------------------------------------------------------------------')
        # now make logits
        s_wv4 = self.wv_out(vec2).squeeze(3) # [bS, 4, mvl, 400] -> [bS, 4, mvl, 1] -> [bS, 4, mvl]
        #print(s_wv2)
        
        # penalty for spurious tokens
        for b, l_n1 in enumerate(l_token):
            for wc in range(wn[b]):
                if wvi3[b][wc] + 1 < mvl:
                    s_wv4[b, wc, : mvl - (wvi3[b][wc] + 1)] = -10000000000.0#mask all of invalid words
            s_wv4[b, wn[b]:] = -10000000000.0
                
        #s_wv = self.softmax_dim2(s_wv)
        #print('s_wv: ', [e[0] for e in s_wv[0][0].tolist()])
                
        return s_wv4

def Loss_sw_se(s_sn, s_sc, s_sa, s_wn, s_wr, s_hrpc, s_wc, s_wo, s_wv1, s_wv2, s_wv3, s_wv4, g_sn, g_sc, g_sa, g_wn, g_dwn, g_wr, g_wc, g_wo, g_wvi, g_wrcn, mvl):
    """

    :param s_wv: score  [ B, n_conds, T, score]
    :param g_wn: [ B ]
    :param g_wvi: [B, conds, pnt], e.g. [[[0, 6, 7, 8, 15], [0, 1, 2, 3, 4, 15]], [[0, 1, 2, 3, 16], [0, 7, 8, 9, 16]]]
    :return:
    """
    loss = 0
    loss += Loss_sn(s_sn, g_sn)
    loss += Loss_sc(s_sc, g_sc)
    loss += Loss_sa(s_sa, g_sn, g_sa)
    loss += Loss_wn(s_wn, g_wn)
    loss += Loss_wr(s_wr, g_wr)
    loss += Loss_hrpc(s_hrpc, [0 if e[0] == -1 else 1 for e in g_wrcn])
    loss += Loss_wc(s_wc, g_wc)
    loss += Loss_wo(s_wo, g_wn, g_wo)
    loss += Loss_wv_se(s_wv1, s_wv2, g_wn, g_wvi)
    loss += Loss_wv_se_ed(s_wv3, s_wv4, g_wn, g_wvi, mvl)

    return loss

def Loss_sn(s_sn, g_sn):
    #p = torch.sigmoid(s_sn)
    return F.cross_entropy(s_sn, torch.tensor(g_sn).to(device))

def Loss_sc(s_sc, g_sc):
    
    # Construct index matrix
    bS, max_h_len = s_sc.shape
    im = torch.zeros([bS, max_h_len]).to(device)
    for b, g_sc1 in enumerate(g_sc):
        for g_sc11 in g_sc1:
            im[b, g_sc11] = 1.0
    # Construct prob.
    p = torch.sigmoid(s_sc)
    loss = F.binary_cross_entropy(p, im)
    
    return loss


def Loss_sa(s_sa, g_sn, g_sa):
    
    loss = 0
    
    for b, g_sn1 in enumerate(g_sn):
        if g_sn1 == 0:
            continue
        g_sa1 = g_sa[b]
        s_sa1 = s_sa[b]
        #p = torch.sigmoid(s_sa1[:g_sn1])
        loss += F.cross_entropy(s_sa1[:g_sn1], torch.tensor(g_sa1).to(device))
    
    return loss

def Loss_wn(s_wn, g_wn):
    #p = torch.sigmoid(s_wn)
    loss = F.cross_entropy(s_wn, torch.tensor(g_wn).to(device))
    return loss

def Loss_wr(s_wr, g_wr):
    #p = torch.sigmoid(s_wr)
    loss = F.cross_entropy(s_wr, torch.tensor(g_wr).to(device))
    return loss

def Loss_hrpc(s_hrpc, g_hrpc):
    #p = torch.sigmoid(s_hrpc)
    p = s_hrpc
    loss = F.cross_entropy(p, torch.tensor(g_hrpc).to(device))
    return loss

def Loss_wc(s_wc, g_wc):

    # Construct index matrix
    bS, max_h_len = s_wc.size()
    im = s_wc.new_zeros([bS, max_h_len]).to(device)
    for b, g_wc1 in enumerate(g_wc):
        for g_wc11 in g_wc1:
            im[b, g_wc11] = 1.0#如果有重复col也不用怕，因为只是重复覆盖1.0而已，结果矩阵还是一样的
    # Construct prob.
    p = torch.sigmoid(s_wc)
    loss = F.binary_cross_entropy(p, im)

    return loss


def Loss_wo(s_wo, g_wn, g_wo):

    # Construct index matrix
    loss = 0
    for b, g_wn1 in enumerate(g_wn):
        if g_wn1 == 0:
            continue
        g_wo1 = g_wo[b]
        s_wo1 = s_wo[b]
        #p = torch.sigmoid(s_wo1[:g_wn1])
        loss += F.cross_entropy(s_wo1[:g_wn1], torch.tensor(g_wo1).to(device))

    return loss

def Loss_wv_se(s_wv1, s_wv2, g_wn, g_wvi):
    """
    s_wv:   [bS, 4, mL, 2], 4 stands for maximum # of condition, 2 tands for start & end logits.
    g_wvi:  [ [1, 3, 2], [4,3] ] (when B=2, wn(b=1) = 3, wn(b=2) = 2).
    """
    loss = 0
    loss_begin = 0
    loss_len = 0
    # g_wvi = torch.tensor(g_wvi).to(device)
    #batchSize = len(g_wvi)
    for b, g_wvi1 in enumerate(g_wvi):
        # for i_wn, g_wvi11 in enumerate(g_wvi1):

        g_wn1 = g_wn[b]
        if g_wn1 == 0:
            continue
        g_wvi1 = torch.tensor(g_wvi1).to(device)
        g_st1 = g_wvi1[:,0]
        g_len1 = g_wvi1[:,1]
        # loss from the start position
        #p = torch.sigmoid(s_wv[b,:g_wn1,:,0])
        loss_begin += F.cross_entropy(s_wv1[b,:g_wn1], g_st1)

        # print("st_login: ", s_wv[b,:g_wn1], g_st1, loss)
        # loss from the end position
        #p = torch.sigmoid(s_wv[b,:g_wn1,:,1])
        loss_len += F.cross_entropy(s_wv2[b,:g_wn1], g_len1)
        # print("len_login: ", s_wv[b,:g_wn1], g_len1, loss)
    
    loss = loss_begin + loss_len
    #print('Loss_wv_se: ', loss.item() / batchSize, '; loss_begin: ', loss_begin.item() / batchSize, '; loss_len: ', loss_len.item() / batchSize)
    
    return loss

def Loss_wv_se_ed(s_wv3, s_wv4, g_wn, g_wvi, mvl):
    loss = 0
    loss_end = 0
    loss_x = 0
    #batchSize = len(g_wvi)
    for b, g_wvi1 in enumerate(g_wvi):
        # for i_wn, g_wvi11 in enumerate(g_wvi1):

        g_wn1 = g_wn[b]
        if g_wn1 == 0:
            continue
        g_wvi1 = torch.tensor(g_wvi1).to(device)
        g_ed1 = g_wvi1[:,0] + g_wvi1[:,1]
        g_x1 = mvl - 1 - g_wvi1[:,1]
        # loss from the start position
        #p = torch.sigmoid(s_wv[b,:g_wn1,:,0])
        loss_end += F.cross_entropy(s_wv3[b,:g_wn1], g_ed1)

        # print("st_login: ", s_wv[b,:g_wn1], g_st1, loss)
        # loss from the end position
        #p = torch.sigmoid(s_wv[b,:g_wn1,:,1])
        loss_x += F.cross_entropy(s_wv4[b,:g_wn1], g_x1)
        # print("len_login: ", s_wv[b,:g_wn1], g_len1, loss)
    
    loss = loss_x + loss_end
    #print('Loss_wv_se_ed: ', loss.item() / batchSize, '; loss_end: ', loss_end.item() / batchSize, '; loss_x: ', loss_x.item() / batchSize)
    
    return loss



# ========= Decoder-Layer ===========
class FT_s2s_1(nn.Module):
    """ Decoder-Layer """
    def __init__(self, iS, hS, lS, dr, max_seq_length, n_cond_ops, n_agg_ops, old=False):
        super(FT_s2s_1, self).__init__()
        self.iS = iS # input_size
        self.hS = hS # hidden_size
        self.ls = lS
        self.dr = dr

        self.n_cond_ops = n_cond_ops
        self.n_agg_ops = n_agg_ops
        self.n_where_num = 4

        self.decoder_s2s = Decoder_s2s(iS, hS, lS, dr, max_seq_length)


    def forward(self, wenc_s2s, l_input, cls_vec, pnt_start_tok, g_pnt_idxs=None):
        score = self.decoder_s2s(wenc_s2s, l_input, cls_vec, pnt_start_tok, g_pnt_idxs)
        return score


    def EG_forward(self, wenc_s2s, l_input, cls_vec,
                   pnt_start_tok, pnt_end_tok,
                   i_sql_vocab, i_nlu, i_hds,  # for EG
                   tokens, nlu, nlu_t, hds, tt_to_t_idx,  # for EG
                   tb, engine,
                   beam_size=4, beam_only=True):
        """ EG-guided beam-search """

        score = self.decoder_s2s.EG_forward(wenc_s2s, l_input, cls_vec,
                                            pnt_start_tok, pnt_end_tok,
                                            i_sql_vocab, i_nlu, i_hds,  # for EG
                                            tokens, nlu, nlu_t, hds, tt_to_t_idx,  # for EG
                                            tb, engine,
                                            beam_size, beam_only)
        return score


class Decoder_s2s(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, max_seq_length=222, n_cond_ops=3):
        super(Decoder_s2s, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.mL = max_seq_length

        self.Tmax = 200

        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2),
                             num_layers=lS, batch_first=True,
                             dropout=dr, bidirectional=True)

        self.decode_pn = nn.LSTM(input_size=max_seq_length, hidden_size=hS,
                                 num_layers=lS, batch_first=True,
                                 dropout=dr)

        self.W_s2s = nn.Linear(iS, hS)
        self.W_pnt = nn.Linear(hS, hS)

        self.wv_out = nn.Sequential(nn.Tanh(), nn.Linear(hS, 1))


    def forward(self, wenc_s2s, l_input, cls_vec, pnt_start_tok, g_pnt_idxs=None,):

        # Encode
        bS, mL_input, iS = wenc_s2s.shape

        # Now, pointer network.
        ipnt = wenc_s2s.new_zeros(bS, 1, mL_input).to(device)  # [B, 1, 200]
        ipnt[:, 0, pnt_start_tok] = 1 # 27 is of start token under current tokenization scheme

        # initial (current) pointer
        cpnt = ipnt

        # reshape wenc_s2s to incorporate T later
        wenc_s2s = wenc_s2s.unsqueeze(1)
        # h_0 and c_0 from cls_vec
        # They are not bidirectional.
        h_0 = torch.zeros([self.lS, bS, self.hS]).to(device)
        c_0 = torch.zeros([self.lS, bS, self.hS]).to(device)
        for i_layer in range(self.lS):
            h_st = (2*i_layer)*self.hS
            h_ed = h_st + self.hS

            c_st = (2*i_layer+1)*self.hS
            c_ed = c_st + self.hS

            h_0[i_layer] = cls_vec[:, h_st:h_ed] # [ # of layers, batch, dim]
            c_0[i_layer] = cls_vec[:, c_st:c_ed] # [ # of layers, batch, dim]

        if g_pnt_idxs:

            pnt_n = torch.zeros(bS, self.Tmax, mL_input).to(device)  # one hot
            # assign index
            for b, g_pnt_idxs1 in enumerate(g_pnt_idxs):
                for t, g_pnt_idx in enumerate(g_pnt_idxs1):
                    pnt_n[b, t, g_pnt_idx] = 1

            # Encode
            dec_pn, _ = self.decode_pn(pnt_n, (h_0, c_0))
            dec_pn = dec_pn.contiguous()

            # [bS, T, iS]
            dec_pn = dec_pn.unsqueeze(2)

            # Calculate score
            s_wv = self.wv_out(
                self.W_s2s(wenc_s2s)
                + self.W_pnt(dec_pn)
            ).squeeze(3) # [B, T, mL_input, dim] -> [B, T, mL_input, 1] -> [B, T, mL_input]
            # s_wv = [B, 4, T, mL_n] = [batch, conds, token idx, score]

            # penalty
            for b, l_input1 in enumerate(l_input):
                if l_input1 < mL_input:
                    s_wv[b, :, l_input1:] = -10000000000

        else:
            t = 0
            s_wv_list = []
            cpnt_h = (h_0, c_0)
            while t < self.Tmax:
                dec_pn, cpnt_h = self.decode_pn(cpnt, cpnt_h)  # lstm

                # [B, 1, 100] -> [B, 1, 1, 100]
                dec_pn = dec_pn.unsqueeze(2)
                # [bS, T, iS]

                # get score
                s_wv1 = self.wv_out(
                    self.W_s2s(wenc_s2s)  # [B, 1,   mL_input, dim]
                    + self.W_pnt(dec_pn)  # [B, T=1,        1, dim]   Now, T=1
                ).squeeze(3)
                # s_wv = [B, 4, 1, mL_n, 1] = [batch, conds, token idx, score]
                # -> [B, 4, mL_n]

                # Masking --
                for b, l_input1 in enumerate(l_input):
                    if l_input1 < mL_input:
                        s_wv1[b, :, l_input1:] = -10000000000

                # Collect score--
                s_wv_list.append(s_wv1)

                # [B, 1, mL_input] -> [B, mL_n] -> [bS*(5-1)]
                # (max_val, max_indices)
                _val, pnt_n = s_wv1.view(bS, -1).max(dim=1)

                # formatting pnt_n as a one-hot input.
                cpnt = torch.zeros(bS, mL_input).to(device)
                # cpnt = cpnt.scatter_(dim=1, index=pnt_n.unsqueeze(1), src=1).to(device)
                cpnt = cpnt.scatter_(1, pnt_n.unsqueeze(1), 1)

                cpnt = cpnt.unsqueeze(1)  # --> [B * 4, 1, 200]
                t += 1


            s_wv = torch.stack(s_wv_list, 1) # [B,
            s_wv = s_wv.squeeze(2) #
            # # Following lines seems to be unnecessary.
            # # Penalty to blank parts
            # for b, l_input1 in enumerate(l_input):
            #     if l_input1 < mL_input:
            #         s_wv[b, :, l_input1:] = -10000000000

        return s_wv


    def EG_forward(self, wenc_s2s, l_input, cls_vec,
                   pnt_start_tok, pnt_end_tok,
                   i_sql_vocab, i_nlu, i_hds, # for EG
                   tokens, nlu, nlu_t, hds, tt_to_t_idx, # for EG
                   tb, engine,
                   beam_size, beam_only=True):

        # Encode
        bS, mL_input, iS = wenc_s2s.shape

        # reshape wenc_s2s to incorperate T later
        wenc_s2s = wenc_s2s.unsqueeze(1)
        # h_0 and c_0 from cls_vec
        # They are not bidirectional.
        h_0 = torch.zeros([self.lS, bS, self.hS]).to(device)
        c_0 = torch.zeros([self.lS, bS, self.hS]).to(device)
        for i_layer in range(self.lS):
            h_st = (2*i_layer)*self.hS
            h_ed = h_st + self.hS

            c_st = (2*i_layer+1)*self.hS
            c_ed = c_st + self.hS

            h_0[i_layer] = cls_vec[:, h_st:h_ed] # [ # of layers, batch, dim]
            c_0[i_layer] = cls_vec[:, c_st:c_ed] # [ # of layers, batch, dim]


        # initial (current) pointer
        pnt_list_beam = []
        cpnt_beam = []
        cpnt_h_beam = []

        for i_beam in range(beam_size):
            pnt_list_beam1 = []
            for b in range(bS):
                pnt_list_beam1.append( [ [pnt_start_tok], 0] )
            pnt_list_beam.append(pnt_list_beam1)
            # initisl cpnt
            # Now, initialize pointer network.
            ipnt = wenc_s2s.new_zeros(bS, 1, mL_input).to(device)  # [B, 1, 200]
            # Distort ipnt by i_bam on purpose to avoid initial duplication of beam-search
            ipnt[:, 0, pnt_start_tok] = 1  # 27 is of start token under current tokenization scheme

            cpnt_beam.append(ipnt)
            cpnt_h_beam.append( (h_0, c_0) )
        t = 0
        while t < self.Tmax:
            # s_wv1_beam = []
            candidates = [ [] for b in range(bS) ]  # [bS]

            # Generate beam
            for i_beam, cpnt in enumerate(cpnt_beam):
                cpnt_h = cpnt_h_beam[i_beam]

                pnt_list_beam1 = pnt_list_beam[i_beam]
                dec_pn, cpnt_h = self.decode_pn(cpnt, cpnt_h)  # lstm
                cpnt_h_beam[i_beam] = cpnt_h

                # [B, 1, 100] -> [B, 1, 1, 100]
                dec_pn = dec_pn.unsqueeze(2)
                # [bS, T, iS]

                # get score
                s_wv1 = self.wv_out(
                    self.W_s2s(wenc_s2s)  # [B, 1,   mL_input, dim]
                    + self.W_pnt(dec_pn)  # [B, T=1,        1, dim]   Now, T=1
                ).squeeze(3)
                # s_wv = [B, 4, 1, mL_n, 1] = [batch, conds, token idx, score]
                # -> [B, 4, mL_n]

                # Masking --
                for b, l_input1 in enumerate(l_input):
                    if l_input1 < mL_input:
                        s_wv1[b, :, l_input1:] = -10000000000


                # Get the candidates only among the input space.
                prob, idxs = F.softmax(s_wv1.view(bS, -1), dim=1).topk(dim=1, k=max(l_input))
                log_prob = torch.log(prob)  # [bS, beam_size]

                for b, log_prob1 in enumerate(log_prob):
                    pnt_list11, score = pnt_list_beam1[b]
                    for i_can, log_prob11 in enumerate(log_prob1):
                        # no update if last token was the end-token
                        previous_pnt = pnt_list11[-1]
                        if previous_pnt== pnt_end_tok:
                            new_seq = pnt_list11
                            new_score = score
                        else:
                            new_seq = pnt_list11 + [idxs[b][i_can].item()]
                            new_score = score + log_prob11.item()
                        _candidate = [new_seq, new_score]

                        candidates[b].append(_candidate)


            # Execution-guided beam filtering
            for b, candidates1 in enumerate(candidates):
                new_pnt_list_batch1 = sorted(candidates1, key=lambda list1: list1[-1], reverse=True)
                cnt = 0
                selected_candidates1 = []
                for new_pnt_list_batch11 in new_pnt_list_batch1:
                    if new_pnt_list_batch11 not in selected_candidates1:
                        if beam_only:
                            selected_candidates1.append(new_pnt_list_batch11)
                            pnt_list_beam[cnt][b] = new_pnt_list_batch11
                            cnt +=1
                        else:
                            # Need to be modified here.
                            executable = False
                            testable = False

                            pr_i_vg_list, pr_i_vg_sub_list = gen_i_vg_from_pnt_idxs([new_pnt_list_batch11[0]], [i_sql_vocab[b]], [i_nlu[b]],
                                                                                    [i_hds[b]])
                            pr_sql_q_s2s, pr_sql_i = gen_sql_q_from_i_vg([tokens[b]], [nlu[b]], [nlu_t[b]], [hds[b]], [tt_to_t_idx[b]],
                                                                         pnt_start_tok, pnt_end_tok,
                                                                         [new_pnt_list_batch11[0]], pr_i_vg_list, pr_i_vg_sub_list)

                            # check testability from select-clause
                            try:
                                # check whether basic elements presents in pr_sql_i
                                # If so, it is testable.

                                idx_agg = pr_sql_i[0]["agg"]
                                idx_sel = pr_sql_i[0]["sel"]
                                testable = True
                            except:
                                testable = False
                                pass

                            # check the presence of conds
                            if testable:
                                try:
                                    conds = pr_sql_i[0]["conds"]
                                except:
                                    conds = []

                                try:
                                    pr_ans1 = engine.execute(tb[b]['id'], idx_sel, idx_agg, conds)
                                    executable = bool(pr_ans1)
                                except:
                                    executable = False

                            #
                            if testable:
                                if executable:
                                    add_candidate = True
                                else:
                                    add_candidate = False
                            else:
                                add_candidate = True


                            if add_candidate:
                                selected_candidates1.append(new_pnt_list_batch11)
                                pnt_list_beam[cnt][b] = new_pnt_list_batch11
                                cnt += 1

                    if cnt == beam_size:
                        break

                if cnt < beam_size:
                    # not executable at all..
                    # add junk sequence.
                    for i_junk in range(cnt, beam_size):
                        pnt_list_beam[i_junk][b] = [[pnt_end_tok],-9999999]

            # generate cpnt
            # formatting pnt_n as a one-hot input.
            for i_beam in range(beam_size):
                cpnt = torch.zeros(bS, mL_input).to(device)
                # cpnt = cpnt.scatter_(dim=1, index=pnt_n.unsqueeze(1), src=1).to(device)
                idx_batch = [seq_score[0][-1] for seq_score in pnt_list_beam[i_beam]]
                pnt_n = torch.tensor(idx_batch).to(device)
                cpnt = cpnt.scatter_(1, pnt_n.unsqueeze(1), 1)
                cpnt = cpnt.unsqueeze(1)  # --> [B, t=1, mL_input]
                cpnt_beam[i_beam] = cpnt
            t += 1

        # Generate best pr_pnt_list, p_tot
        pr_pnt_idxs = []
        p_list = []
        for b in range(bS):
            pnt_list_beam_best = pnt_list_beam[0]
            pr_pnt_idxs.append(pnt_list_beam_best[b][0])
            p_list.append( pnt_list_beam_best[b][1])

        return pr_pnt_idxs, p_list, pnt_list_beam


# =============  Shallow-Layer ===============
class FT_Scalar_1(nn.Module):
    """ Shallow-Layer """
    def __init__(self, iS, hS, lS, dr, n_cond_ops, n_agg_ops, old=False):
        super(FT_Scalar_1, self).__init__()
        self.iS = iS # input_size
        self.hS = hS
        self.ls = lS
        self.dr = dr

        self.n_cond_ops = n_cond_ops
        self.n_agg_ops = n_agg_ops
        self.n_where_num = 4


    def scp(self, wemb_h, l_hs):
        bS, max_header_len, _ = wemb_h.shape
        # s_sc

        s_sc = torch.zeros(bS, max_header_len).to(device)
        s_sc[:, :] = wemb_h[:, :, 0]  # s_sc = [B, max_header length, 1]

        # s_sc[:,:] = F.tanh(wemb_h[:,:,0])  # s_sc = [B, max_header length, 1]
        # s_sc = s_sc.squeeze(2)
        # masking
        # print(f"s_sc {s_sc}")
        for b, l_hs1 in enumerate(l_hs):
            s_sc[b, l_hs1:] = -9999999999.0

        return s_sc

    def sap(self, wemb_h, pr_sc, idx_st, idx_ed):
        bS, max_header_len, _ = wemb_h.shape
        # select of aggregation operator
        s_sa = torch.zeros([bS, self.n_agg_ops]).to(device)
        for b, pr_sc1 in enumerate(pr_sc):
            s_sa[b,:] = wemb_h[b,pr_sc1,idx_st:idx_ed]

        return s_sa

    def wnp(self, cls_vec):
        bS = cls_vec.shape[0]
        # [B,hS] -> [B, n_where_num+1]
        s_wn = torch.zeros(bS, (self.n_where_num + 1)).to(device)
        s_wn[:, :] = cls_vec[:, 0:(self.n_where_num + 1)]

        return s_wn

    def wcp(self, wemb_h, l_hs, idx_st, idx_ed):
        bS, max_header_len, _ = wemb_h.shape

        s_wc = torch.zeros(bS, max_header_len, 1).to(device)
        s_wc[:, :, :] = wemb_h[:, :, idx_st:idx_ed]

        s_wc = s_wc.squeeze(2)  # [B, max_header_length]

        # masking
        for b, l_hs1 in enumerate(l_hs):
            s_wc[b, l_hs1:] = -99999999999.0

        return s_wc

    def wop(self, wemb_h, pr_wc, idx_st, idx_ed):
        bS, max_header_len, _ = wemb_h.shape

        s_wo = torch.zeros([bS, self.n_where_num, self.n_cond_ops]).to(device)
        for b, pr_wc1 in enumerate(pr_wc):
            if len(pr_wc1) > 0:
                s_wo[b, 0:len(pr_wc1), :] = wemb_h[b, pr_wc1, idx_st:idx_ed]
            else:
                pass

        return s_wo

    def wvp(self, wemb_n, l_n, pr_wc):
        bS, _, _ = wemb_n.shape

        s_wv = torch.zeros([bS, self.n_where_num, max(l_n), 2]).to(device)
        for b, pr_wc1 in enumerate(pr_wc):

            if len(pr_wc1) > 0:
                # start logit
                s_wv[b, 0:len(pr_wc1), :, 0] = wemb_n[b, :, pr_wc1].transpose(0, 1)
                # end logit
                s_wv[b, 0:len(pr_wc1), :, 1] = wemb_n[b, :, [pr_wc11 + 100 for pr_wc11 in pr_wc1]].transpose(0, 1)
            else:
                pass

        # masking
        # penalty for spurious tokens
        for b, l_n1 in enumerate(l_n):
            if l_n1 < max(l_n):
                s_wv[b, :, l_n1:, :] = -1e+11
        return s_wv

    def forward(self, wemb_n, l_n, wemb_h, l_hs, cls_vec,
                g_sc=None, g_sa=None, g_wn=None, g_wc=None, g_wo=None, g_wvi=None,
                show_p_sc=False, show_p_sa=False,
                show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False):

        # wemb_n = [B, max_nlu_token_length, hS] # here, # of target_layer is fixed to 1.
        # wemb_h = [B, max_header #, hS]

        s_sc = self.scp(wemb_h, l_hs)
        if g_sc:
            pr_sc = g_sc
        else:
            pr_sc = pred_sc(s_sc)

        # s_sa
        idx_st = 1
        idx_ed = 1 + self.n_agg_ops
        s_sa = self.sap(wemb_h, pr_sc, idx_st, idx_ed)

        if g_sa:
            pr_sa = g_sa
        else:
            pr_sa = pred_sa(s_sa)

        # where_number

        s_wn = self.wnp(cls_vec)
        if g_wn:
            pr_wn = g_wn
        else:
            pr_wn = pred_wn(s_wn)

        # wc
        idx_st = idx_ed+1
        idx_ed = idx_st+1
        s_wc = self.wcp(wemb_h, l_hs, idx_st, idx_ed)

        if g_wc:
            pr_wc = g_wc
        else:
            pr_wc = pred_wc(pr_wn, s_wc)

        # wo
        idx_st = idx_ed+1
        idx_ed = idx_st + self.n_cond_ops

        s_wo = self.wop(wemb_h, pr_wc, idx_st, idx_ed)

        if g_wo:
            pr_wo = g_wo
        else:
            pr_wo = pred_wo(pr_wn, s_wo)

        # wv
        # s_wv =  [bS, 4, mL, 2]
        s_wv = self.wvp(wemb_n, l_n, pr_wc)

        # print(s_wv)
        # s_wv = F.tanh(s_wv)
        return s_sc, s_sa, s_wn, s_wc, s_wo, s_wv


    def forward_EG(self, wemb_n, l_n, wemb_h, l_hs, cls_vec, engine, tb,
                   nlu_t, nlu_tt, tt_to_t_idx, nlu,
                   beam_size=4):
        """
        Execution-guided beam decoding.
        Essentially identical with that of NL2SQL Layer.
        """
        # Select-clause
        prob_sca, pr_sc_best, pr_sa_best, \
        p_sc_best, p_sa_best, p_select \
            = self.EG_decoding_select(wemb_h, l_hs, tb, beam_size=beam_size)

        # Where-clause
        prob_w, prob_wn_w, pr_wn_based_on_prob, pr_sql_i, pr_wvi_best, \
        p_where, p_wn_best, p_wc_best, p_wo_best, p_wvi_best \
            = self.EG_decoding_where(wemb_n, l_n, wemb_h, l_hs, cls_vec, engine, tb,
                                     nlu_t, nlu_tt, tt_to_t_idx, nlu,
                                     pr_sc_best, pr_sa_best,
                                     beam_size=4)

        p_tot = cal_prob_tot(p_select, p_where)
        return pr_sc_best, pr_sa_best, pr_wn_based_on_prob, pr_wvi_best, \
               pr_sql_i, p_tot, p_select, p_where, p_sc_best, p_sa_best, \
               p_wn_best, p_wc_best, p_wo_best, p_wvi_best


    def EG_decoding_select(self, wemb_h, l_hs, tb,
                           beam_size=4, show_p_sc=False, show_p_sa=False):

        # sc
        s_sc = self.scp(wemb_h, l_hs)
        prob_sc = F.softmax(s_sc, dim=-1)
        bS, mcL = s_sc.shape

        # minimum_hs_length = min(l_hs)
        # beam_size = minimum_hs_length if beam_size > minimum_hs_length else beam_size

        # sa
        # Construct all possible sc_sa_score
        prob_sc_sa = torch.zeros([bS, beam_size, self.n_agg_ops]).to(device)
        score_sc_sa = torch.zeros([bS, beam_size, self.n_agg_ops]).to(device)

        prob_sca = torch.zeros_like(prob_sc_sa).to(device)

        # get the top-k indices.  pr_sc_beam = [B, beam_size]
        pr_sc_beam = pred_sc_beam(s_sc, beam_size)

        # calculate and predict s_sa.
        idx_st = 1
        idx_ed = 1 + self.n_agg_ops
        for i_beam in range(beam_size):
            pr_sc = list(array(pr_sc_beam)[:, i_beam])
            s_sa = self.sap(wemb_h, pr_sc, idx_st, idx_ed)
            prob_sa = F.softmax(s_sa, dim=-1)
            prob_sc_sa[:, i_beam, :] = prob_sa
            score_sc_sa[:, i_beam, :] = s_sa

            prob_sc_selected = prob_sc[range(bS), pr_sc]  # [B]
            prob_sca[:, i_beam, :] = (prob_sa.t() * prob_sc_selected).t()
            # [mcL, B] * [B] -> [mcL, B] (element-wise multiplication)
            # [mcL, B] -> [B, mcL]

        # Calculate the dimension of tensor
        # tot_dim = len(prob_sca.shape)

        idxs = topk_multi_dim(torch.tensor(prob_sca), n_topk=beam_size, batch_exist=True)
        # Now as sc_idx is already sorted, re-map them properly.
        idxs = remap_sc_idx(idxs, pr_sc_beam)  # [sc_beam_idx, sa_idx] -> [sc_idx, sa_idx]
        idxs_arr = array(idxs)
        # [B, beam_size, remainig dim]
        # idxs[b][0] gives first probable [sc_idx, sa_idx] pairs.
        # idxs[b][1] gives of second.

        # Calculate prob_sca, a joint probability
        beam_idx_sca = [0] * bS
        beam_meet_the_final = [False] * bS
        while True:
            pr_sc = idxs_arr[range(bS), beam_idx_sca, 0]
            pr_sa = idxs_arr[range(bS), beam_idx_sca, 1]

            # map index properly

            check = check_sc_sa_pairs(tb, pr_sc, pr_sa)

            if sum(check) == bS:
                break
            else:
                for b, check1 in enumerate(check):
                    if not check1:  # wrong pair
                        beam_idx_sca[b] += 1
                        if beam_idx_sca[b] >= beam_size:
                            beam_meet_the_final[b] = True
                            beam_idx_sca[b] -= 1
                    else:
                        beam_meet_the_final[b] = True

            if sum(beam_meet_the_final) == bS:
                break

        # Now pr_sc, pr_sa are properly predicted.
        pr_sc_best = list(pr_sc)
        pr_sa_best = list(pr_sa)

        # output for later analysis.
        p_sc_best = cal_prob_sc(s_sc, pr_sc_best)
        p_sa_best = cal_prob_sa(score_sc_sa[range(bS), beam_idx_sca, :].squeeze(1), pr_sa_best)
        p_select = cal_prob_select(p_sc_best, p_sa_best)
        # p_select  = prob_sca[range(bS),beam_idx_sca,pr_sa_best].detach().to('cpu').numpy()

        return prob_sca, pr_sc_best, pr_sa_best, p_sc_best, p_sa_best, p_select

    def EG_decoding_where(self, wemb_n, l_n, wemb_h, l_hs, cls_vec, engine, tb,
                     nlu_t, nlu_wp_t, tt_to_t_idx, nlu,
                          pr_sc_best, pr_sa_best,
                     beam_size=4, show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False):

        bS, max_header_len, _ = wemb_h.shape

        # Now, Where-clause beam search.
        idx_st = 1
        idx_ed = 1 + self.n_agg_ops

        s_wn = self.wnp(cls_vec)
        prob_wn = F.softmax(s_wn, dim=-1).detach().to('cpu').numpy()

        # Found "executable" most likely 4(=max_num_of_conditions) where-clauses.
        # wc
        idx_st = idx_ed + 1
        idx_ed = idx_st + 1

        s_wc = self.wcp(wemb_h, l_hs, idx_st, idx_ed)
        prob_wc = torch.sigmoid(s_wc).detach().to('cpu').numpy()
        # pr_wc_sorted_by_prob = pred_wc_sorted_by_prob(s_wc)

        # get max_wn # of most probable columns & their prob.
        pr_wn_max = [self.n_where_num] * bS
        pr_wc_max = pred_wc(pr_wn_max, s_wc)  # if some column do not have executable where-claouse, omit that column
        prob_wc_max = zeros([bS, self.n_where_num])
        for b, pr_wc_max1 in enumerate(pr_wc_max):
            prob_wc_max[b, :] = prob_wc[b, pr_wc_max1]

        # get most probable n_where_num where-clouses
        # wo
        idx_st = idx_ed + 1
        idx_ed = idx_st + self.n_cond_ops
        s_wo_max = self.wop(wemb_h, pr_wc_max, idx_st, idx_ed)
        prob_wo_max = F.softmax(s_wo_max, dim=-1).detach().to('cpu').numpy()
        # [B, n_where_num, n_cond_op]

        pr_wvi_beam_op_list = []
        prob_wvi_beam_op_list = []
        prob_wvi_beam_st_op_list = []
        prob_wvi_beam_ed_op_list = []

        # To re-use code, repeat the calculation unnecessarily.
        for i_op in range(self.n_cond_ops - 1):
            pr_wo_temp = [[i_op] * self.n_where_num] * bS
            # wv
            s_wv = self.wvp(wemb_n, l_n, pr_wc_max)
            prob_wv = F.softmax(s_wv, dim=-2).detach().to('cpu').numpy()

            # prob_wv
            pr_wvi_beam, prob_wvi_beam, prob_wvi_beam_st, prob_wvi_beam_ed = pred_wvi_se_beam(self.n_where_num, s_wv, beam_size)
            pr_wvi_beam_op_list.append(pr_wvi_beam)

            prob_wvi_beam_op_list.append(prob_wvi_beam)
            prob_wvi_beam_st_op_list.append(prob_wvi_beam_st)
            prob_wvi_beam_ed_op_list.append(prob_wvi_beam_ed)
            # pr_wvi_beam = [B, n_where_num, k_logit**2 [st, ed] paris]

            # pred_wv_beam

        # Calculate joint probability of where-clause
        # prob_w = [batch, wc, wo, wv] = [B, n_where_num, n_cond_op, n_pairs]
        n_wv_beam_pairs = prob_wvi_beam.shape[2]
        prob_w = zeros([bS, self.n_where_num, self.n_cond_ops - 1, n_wv_beam_pairs])
        prob_wc_dupl = zeros([bS, self.n_where_num, self.n_cond_ops - 1, n_wv_beam_pairs])
        prob_wo_dupl = zeros([bS, self.n_where_num, self.n_cond_ops - 1, n_wv_beam_pairs])
        prob_wvi_st_dupl = zeros([bS, self.n_where_num, self.n_cond_ops - 1, n_wv_beam_pairs])
        prob_wvi_ed_dupl = zeros([bS, self.n_where_num, self.n_cond_ops - 1, n_wv_beam_pairs])

        for b in range(bS):
            for i_wn in range(self.n_where_num):
                for i_op in range(self.n_cond_ops - 1):  # do not use final one
                    p_wc = prob_wc_max[b, i_wn]
                    for i_wv_beam in range(n_wv_beam_pairs):
                        # i_wc = pr_wc_max[b][i_wn] # already done
                        p_wo = prob_wo_max[b, i_wn, i_op]
                        p_wv = prob_wvi_beam_op_list[i_op][b, i_wn, i_wv_beam]

                        prob_w[b, i_wn, i_op, i_wv_beam] = p_wc * p_wo * p_wv
                        prob_wc_dupl[b, i_wn, i_op, i_wv_beam] = p_wc
                        prob_wo_dupl[b, i_wn, i_op, i_wv_beam] = p_wo

                        p_wv_st = prob_wvi_beam_st_op_list[i_op][b, i_wn, i_wv_beam]
                        p_wv_ed = prob_wvi_beam_ed_op_list[i_op][b, i_wn, i_wv_beam]

                        prob_wvi_st_dupl[b, i_wn, i_op, i_wv_beam] = p_wv_st
                        prob_wvi_ed_dupl[b, i_wn, i_op, i_wv_beam] = p_wv_ed


        # Perform execution guided decoding
        conds_max = []
        prob_conds_max = []
        # while len(conds_max) < self.n_where_num:
        idxs = topk_multi_dim(torch.tensor(prob_w), n_topk=beam_size, batch_exist=True)
        # idxs = [B, i_wc_beam, i_op, i_wv_pairs]

        # Construct conds1. Collect only executable one. It is descending order of the probability.
        pr_wvi_max = []

        p_wc_max = []
        p_wo_max = []
        p_wvi_max = []
        for b, idxs1 in enumerate(idxs):
            conds_max1 = []
            prob_conds_max1 = []
            pr_wvi1_max = []

            p_wc1_max = []
            p_wo1_max = []
            p_wvi1_max = []

            for i_wn, idxs11 in enumerate(idxs1):
                i_wc = pr_wc_max[b][idxs11[0]]
                i_op = idxs11[1]
                wvi = pr_wvi_beam_op_list[i_op][b][idxs11[0]][idxs11[2]]

                # idx11[0]

                # get wv_str
                temp_pr_wv_str, _ = convert_pr_wvi_to_string([[wvi]], [nlu_t[b]], [nlu_wp_t[b]], [tt_to_t_idx[b]],
                                                             [nlu[b]])
                merged_wv11 = merge_wv_t1_eng(temp_pr_wv_str[0][0], nlu[b])
                conds11 = [i_wc, i_op, merged_wv11]


                prob_conds11 = prob_w[b, idxs11[0], idxs11[1], idxs11[2]]
                p_wc11_max = prob_wc_dupl[b, idxs11[0], idxs11[1], idxs11[2]]
                p_wo11_max = prob_wo_dupl[b, idxs11[0], idxs11[1], idxs11[2]]
                p_wvi11_max = [ prob_wvi_st_dupl[b, idxs11[0], idxs11[1], idxs11[2]],
                                prob_wvi_ed_dupl[b, idxs11[0], idxs11[1], idxs11[2]] ]

                # test execution
                # print(nlu[b])
                # print(tb[b]['id'], tb[b]['types'], pr_sc[b], pr_sa[b], [conds11])
                pr_ans = engine.execute(tb[b]['id'], pr_sc_best[b], pr_sa_best[b], [conds11])
                if bool(pr_ans):
                    # pr_ans is not empty!
                    conds_max1.append(conds11)
                    prob_conds_max1.append(prob_conds11)
                    pr_wvi1_max.append(wvi)

                    p_wc1_max.append(p_wc11_max)
                    p_wo1_max.append(p_wo11_max)
                    p_wvi1_max.append(p_wvi11_max)


            conds_max.append(conds_max1)
            prob_conds_max.append(prob_conds_max1)
            pr_wvi_max.append(pr_wvi1_max)

            p_wc_max.append(p_wc1_max)
            p_wo_max.append(p_wo1_max)
            p_wvi_max.append(p_wvi1_max)

            # May need to do more exhuastive search?
            # i.e. up to.. getting all executable cases.

        # Calculate total probability to decide the number of where-clauses
        pr_sql_i = []
        prob_wn_w = [] # total where-clause probability
        pr_wn_based_on_prob = []
        pr_wvi_best = []

        p_wc = []
        p_wo = []
        p_wvi = []

        for b, prob_wn1 in enumerate(prob_wn):
            max_executable_wn1 = len(conds_max[b])
            prob_wn_w1 = []
            prob_wn_w1.append(prob_wn1[0])  # wn=0 case.
            for i_wn in range(max_executable_wn1):
                prob_wn_w11 = prob_wn1[i_wn + 1] * prob_conds_max[b][i_wn]
                prob_wn_w1.append(prob_wn_w11)
            pr_wn_based_on_prob.append(argmax(prob_wn_w1))
            prob_wn_w.append(prob_wn_w1)

            pr_sql_i1 = {'agg': pr_sa_best[b], 'sel': pr_sc_best[b], 'conds': conds_max[b][:pr_wn_based_on_prob[b]]}
            pr_wvi_best1 = pr_wvi_max[b][:pr_wn_based_on_prob[b]]


            pr_sql_i.append(pr_sql_i1)
            pr_wvi_best.append(pr_wvi_best1)

            p_wc.append( p_wc_max[b][:pr_wn_based_on_prob[b]] )
            p_wo.append( p_wo_max[b][:pr_wn_based_on_prob[b]] )
            p_wvi.append( p_wvi_max[b][:pr_wn_based_on_prob[b]] )




        # s_wv = [B, n_where_num, max_nlu_tokens, 2]

        p_wn = cal_prob_wn(s_wn, pr_wn_based_on_prob)
        p_where = cal_prob_where(p_wn, p_wc, p_wo, p_wvi)

        return prob_w, prob_wn_w, pr_wn_based_on_prob, pr_sql_i, pr_wvi_best, \
               p_where, p_wn, p_wc, p_wo, p_wvi


def Loss_s2s(score, g_pnt_idxs):
    """
    score = [B, T, max_seq_length]
    """
    #         WHERE string part
    loss = 0

    for b, g_pnt_idxs1 in enumerate(g_pnt_idxs):
        ed = len(g_pnt_idxs1) - 1
        score_part = score[b, :ed]
        loss += F.cross_entropy(score_part, torch.tensor(g_pnt_idxs1[1:]).to(device))  # +1 shift.
    return loss
