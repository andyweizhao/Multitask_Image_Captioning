from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import misc.resnet as resnet
import os
import random

def set_bn_fix(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters(): 
            p.requires_grad=False
            
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        
def build_cnn(opt):
    net = getattr(resnet, opt.cnn_model)()
    if vars(opt).get('start_from', None) is None and vars(opt).get('cnn_weight', '') != '':   
        print(vars(opt).get('start_from')+'/load pretrained reset101')
#        net.load_state_dict(torch.load(opt.cnn_weight))                        
        pretrained_m = torch.load(opt.cnn_weight)
        
    if vars(opt).get('start_from', None) is not None:
        print(vars(opt).get('start_from')+'/model-cnn.pth')        
#        net.load_state_dict(torch.load(os.path.join(opt.start_from, 'model-cnn.pth')))                        
        pretrained_m = torch.load(os.path.join(opt.start_from, 'model-cnn.pth'))
        
    net_dict = net.state_dict()
    pretrained_m = { k:v for k,v in pretrained_m.iteritems() 
                    if k in net_dict and v.size() == net_dict[k].size() }
    net_dict.update(pretrained_m)
    net.load_state_dict(net_dict)
                
#    compact_net = nn.Sequential(\
#        net.conv1,
#        net.bn1,
#        net.relu,
#        net.maxpool,
#        net.layer1,
#        net.layer2,
#        net.layer3,
#        net.layer4)
#    
#    return net,compact_net
    return net

def prepro_images(imgs, data_augment=False):
    # crop the image
    h,w = imgs.shape[2], imgs.shape[3]
    cnn_input_size = 224

    # cropping data augmentation, if needed
    if h > cnn_input_size or w > cnn_input_size:
        if data_augment:
          xoff, yoff = random.randint(0, w-cnn_input_size), random.randint(0, h-cnn_input_size)
        else:
          # sample the center
          xoff, yoff = (w-cnn_input_size)//2, (h-cnn_input_size)//2
    # crop.
    imgs = imgs[:,:, yoff:yoff+cnn_input_size, xoff:xoff+cnn_input_size]

    return imgs

def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    
    N, D = seq.size()
    out = []
   
 
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                
                txt = txt + ix_to_word.get(str(ix),'unknown_token')
            else:
                break
        out.append(txt)
    
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * Variable(mask)
        output = torch.sum(output) / torch.sum(mask)

        return output
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output
class LanguageModel_CCG_Criterion(nn.Module):
    def __init__(self):
        super(LanguageModel_CCG_Criterion, self).__init__()

    def forward(self, word_labels,ccg_labels, word_target,ccg_target,  mask):
        # truncate to the same size
        word_target = word_target[:, :word_labels.size(1)]
        mask =  mask[:, :word_labels.size(1)]
        word_labels = to_contiguous(word_labels).view(-1, word_labels.size(2))
        word_target = to_contiguous(word_target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - word_labels.gather(1, word_target) * mask
        output_word = torch.sum(output) / torch.sum(mask)
        
        
        ccg_target = ccg_target[:, :ccg_labels.size(1)]
        mask =  mask[:, :ccg_labels.size(1)]
        ccg_labels = to_contiguous(ccg_labels).view(-1, ccg_labels.size(2))
        ccg_target = to_contiguous(ccg_target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - ccg_labels.gather(1, ccg_target) * mask
        output_ccg = torch.sum(output) / torch.sum(mask)

        return output_word,output_ccg
def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:       
            if param.grad is not None and param.requires_grad:
                param.grad.data.clamp_(-grad_clip, grad_clip)
