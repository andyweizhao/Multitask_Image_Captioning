# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

class CCGAttModel(nn.Module):
    def __init__(self, opt):
        super(CCGAttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.ccg_embedding_dim = opt.ccg_embedding_dim
        self.ccg_vocab_size = opt.ccg_vocab_size
        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        self.ccg_embed = nn.Sequential(nn.Embedding(self.ccg_vocab_size + 1, self.ccg_embedding_dim), 
                                nn.ReLU(), 
                                nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.logit_ccg = nn.Linear(self.rnn_size, self.ccg_vocab_size + 1)
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))

    def forward(self, fc_feats, att_feats, seq, ccg_seq):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs_word = []
        outputs_ccg=[]

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(*(att_feats.size()[:-1] + (self.rnn_size,)))

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.rnn_size))
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))
        
        seq_size = min (seq.size(1) - 1,ccg_seq.size(1) - 1)
        for i in range(seq_size):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0: # disable schedule sampling
                    it = seq[:, i].clone()
                    it_ccg = ccg_seq[:, i].clone()
                else: # enable schedule sampling 
                    sample_ind = sample_mask.nonzero().view(-1)                                     
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))                    
                    prob_prev = torch.exp(outputs_word[-1].data) # fetch prev distribution: shape Nx(M+1)
                    it = seq[:, i].data.clone()  
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)                    
                    
                    # disable shedule sampling for CCG.
#                    prob_prev = torch.exp(outputs_ccg[-1].data) # fetch prev distribution: shape Nx(M+1)
                    it_ccg = ccg_seq[:, i].data.clone()
#                    it_ccg.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it_ccg = Variable(it_ccg, requires_grad=False)
            else:
                it = seq[:, i].clone()
                it_ccg = ccg_seq[:, i].clone()
#                it = Variable(it, requires_grad=False)
#                it_cross = Variable(it_cross, requires_grad=False)

            # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.embed(it)
            xt_ccg = self.ccg_embed(it_ccg)
            output, state = self.core(xt,xt_ccg,  fc_feats, att_feats, p_att_feats, state)                                                                           
            
            output_word = F.log_softmax(self.logit(output))            
            output_ccg = F.log_softmax(self.logit_ccg(output))
            
            outputs_word.append(output_word)
            outputs_ccg.append(output_ccg)

        return torch.cat([_.unsqueeze(1) for _ in outputs_word], 1) , torch.cat([_.unsqueeze(1) for _ in outputs_ccg], 1)

    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(*(att_feats.size()[:-1] + (self.rnn_size,)))

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.rnn_size))
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))

        seq = []
        seq_ccg=[]
        seqLogprobs = []
        seqLogprobs_ccg = []

        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
                it_ccg= fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
                
                sampleLogprobs_ccg, it_ccg = torch.max(logprobs_ccg.data, 1)
                it_ccg = it_ccg.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                    prob_prev_ccg = torch.exp(logprobs_ccg.data).cpu() 
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                    prob_prev_ccg = torch.exp(torch.div(logprobs_ccg.data, temperature)).cpu()
                    
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False)) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing
                
                sampleLogprobs_ccg, it_ccg = torch.max(logprobs_ccg.data, 1)
                it_ccg = it_ccg.view(-1).long()                
#                it_ccg = torch.multinomial(prob_prev_ccg, 1).cuda()                                
#                sampleLogprobs_ccg = logprobs_ccg.gather(1, Variable(it_ccg, requires_grad=False))                
#                it_ccg = it_ccg.view(-1).long()
                
            xt = self.embed(Variable(it, requires_grad=False))
            xt_ccg = self.ccg_embed(Variable(it_ccg, requires_grad=False))

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it) #seq[t] the input of t+2 time step
                
                it_ccg = it_ccg * unfinished.type_as(it_ccg)                
                seq_ccg.append(it_ccg)
                
                seqLogprobs.append(sampleLogprobs.view(-1))
                seqLogprobs_ccg.append(sampleLogprobs_ccg.view(-1))

            output, state = self.core(xt,xt_ccg, fc_feats, att_feats, p_att_feats, state)
            
            logprobs = F.log_softmax(self.logit(output))
            logprobs_ccg = F.log_softmax(self.logit_ccg(output))

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1), \
            torch.cat([_.unsqueeze(1) for _ in seq_ccg], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs_ccg], 1)

class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False,ccg_embedding_dim=None):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + ccg_embedding_dim + opt.rnn_size * 2, 
                                        opt.rnn_size) # we, ccg, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)
        #self.ccg_lstm =  nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)

    def forward(self, xt, xt_ccg, fc_feats, att_feats, p_att_feats, state):
        prev_h = state[0][-1] # h_lang
        att_lstm_input = torch.cat([prev_h, fc_feats, xt, xt_ccg], 1) # h_{t-1}^2, v_{bar}, w_{e}*pi_{t}

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))# h_{t}^1               

        att = self.attention(h_att, att_feats, p_att_feats)# h_{t}^1, conv_feats, p_conv_feats

        lang_lstm_input = torch.cat([att, h_att], 1)# \hat v, h_{t}^1  
        
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), 
                 torch.stack([c_att, c_lang]))

        return output, state
    
class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot)                             # batch * att_size
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res

class CCG_TopDownModel(CCGAttModel):
    def __init__(self, opt):
        print("use new model")
        super(CCG_TopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownCore(opt,ccg_embedding_dim=opt.ccg_embedding_dim)
