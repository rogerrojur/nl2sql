# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:55:08 2019

@author: 63184
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import utils as nn_utils
'''
batch_size = 2
max_length = 4
hidden_size = 2
n_layers =1
 
tensor_in = torch.FloatTensor([[1, 2, 3, 4], [1, 0, 0, 0]]).resize_(2,4,1)
tensor_in = Variable( tensor_in ) #[batch, seq, feature], [2, 4, 1]
seq_lengths = [4,1] # list of integers holding information about the batch size at each sequence step
 
# pack it
pack = nn_utils.rnn.pack_padded_sequence(tensor_in, seq_lengths, batch_first=True)
 
# initialize
rnn = nn.RNN(1, hidden_size, n_layers, batch_first=True)
h0 = Variable(torch.randn(n_layers, batch_size, hidden_size))
 
#forward
out, _ = rnn(pack, h0)
 
# unpack
unpacked = nn_utils.rnn.pad_packed_sequence(out)
print('111',unpacked)
'''
lstm = nn.LSTM(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
c0 = Variable(torch.randn(2, 3, 20))
output, hn = lstm(input, None)
print(output)