# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 23:09:42 2019

@author: 63184
"""

class WOP(nn.Module):
    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_cond_ops=3):
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
            real = [wenc_hs[b, col] for col in wc[b]]#[selected col, dim]
            pad = (self.mL_w - wn[b]) * [wenc_hs[b, 0]] # this padding could be wrong. Test with zero padding later. wn[b] is an int to indicate how many cols are selected
            wenc_hs_ob1 = torch.stack(real + pad) # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 4, dim] tensor.
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
                att[b, :, l_n1:] = -10000000000

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

        # [bS, 5-1, dim] -> [bS, 5-1, n_cond_ops]

        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob)], dim=2)
        s_wo = self.wo_out(vec)

        return s_wo
    
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
            nn.Tanh(), nn.Linear(2 * hS, 1)
        )

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wc, penalty=True):
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
                att[b_n, :, l_n1:] = -10000000000

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
                score[b, l_hs1:] = -1e+10

        return score
    
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
                att_h[b, l_hs1:] = -10000000000
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
                att_n[b, l_n1:] = -10000000000
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
        s_wn = self.wn_out(c_n)

        return s_wn
    
def pred_wc(dwn, pr_hrpc, pr_wrpc, pr_nrpc, s_wc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    ! Returned index is sorted!
    """
    # get g_num
    s_wc = torch.sigmoid(s_wc)
    pr_wc = []
    for b, wn1 in enumerate(dwn):
        s_wc1 = s_wc[b]

        pr_wc1 = list(argsort(-s_wc1.data.cpu().numpy())[:wn1])
        
        if pr_hrpc[b]:
            l1 = len(pr_wc1)
            pr_wc1 = [e for e in pr_wc1 if e != pr_wrpc[b]]
            l2 = len(pr_wc1)
            if l1 == l2:
                pr_wc1 = pr_wc[:-1]
            pr_wc1.sort()
            pr_wc1 = [pr_wrpc[b]] * pr_nrpc[b] + pr_wc1
        else:
            pr_wc1.sort()
        pr_wc.append(pr_wc1)
    return pr_wc

def train(train_loader, train_table, model, model_bert, opt, bert_config, tokenizer,
          max_seq_length, num_target_layers, accumulate_gradients=1, check_grad=True,
          st_pos=0, opt_bert=None, path_db=None, dset_name='train'):
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
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
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
        g_sn, g_sc, g_sa, g_wn, g_wr, g_dwn, g_wc, g_wo, g_wv, g_wrcn = get_g(sql_i)#get the where values
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
        g_wvi_corenlp = get_g_wvi_corenlp(t)
        # this function is to get the indices of where values from the question token

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        '''
        print('wemb_n: ', torch.tensor(wemb_n).size())
        print('wemb_h: ', torch.tensor(wemb_h).size())
        print('l_n: ', l_n)
        print('l_hpu: ', l_hpu)
        print('l_hs: ', l_hs)
        print('nlu_tt: ', nlu_tt)
        print('t_to_tt_idx: ', t_to_tt_idx)
        print('tt_to_t_idx: ', tt_to_t_idx)
        print('g_wvi_corenlp', g_wvi_corenlp)
        '''
        # wemb_n: natural language embedding
        # wemb_h: header embedding
        # l_n: token lengths of each question
        # l_hpu: header token lengths
        # l_hs: the number of columns (headers) of the tables.
        try:
            #
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)#if not exist, it will not train not include the length, so the end value is the start index of this word, not the end index of this word, so it need to add sth
            print('g_wvi', g_wvi)
        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            # e.g. train: 32.
            continue
        # score
        s_sn, s_sc, s_sa, s_wn, s_wr, s_hrpc, s_wrpc, s_nrpc, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
                                                   g_sn=g_sn, g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_dwn=g_dwn, g_wr=g_wr, g_wc=g_wc, g_wo=g_wo, g_wvi=g_wvi, g_wrcn=g_wrcn)

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
        print('s_wv: ', s_wv)
        '''
        
        # Calculate loss & step
        loss = Loss_sw_se(s_sn, s_sc, s_sa, s_wn, s_wr, s_hrpc, s_wrpc, s_nrpc, s_wc, s_wo, s_wv, g_sn, g_sc, g_sa, g_wn, g_dwn, g_wr, g_wc, g_wo, g_wvi, g_wrcn)

        #print('loss: ', loss)
        
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
        pr_sn, pr_sc, pr_sa, pr_wn, pr_wr, pr_hrpc, pr_wrpc, pr_nrpc, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sn, s_sc, s_sa, s_wn, s_wr, s_hrpc, s_wrpc, s_nrpc, s_wc, s_wo, s_wv)
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
        
        pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
        '''
        print('pr_wv_str: ', pr_wv_str)
        print('pr_wv_str_wp: ', pr_wv_str_wp)
        '''
        # Sort pr_wc:
        #   Sort pr_wc when training the model as pr_wo and pr_wvi are predicted using ground-truth where-column (g_wc)
        #   In case of 'dev' or 'test', it is not necessary as the ground-truth is not used during inference.
        pr_wc_sorted = sort_pr_wc(pr_wc, g_wc)
        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wr, pr_wc_sorted, pr_wo, pr_wv_str, nlu)
        
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
        break

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
    '''
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

    aux_out = 1
    '''
    return acc, aux_out