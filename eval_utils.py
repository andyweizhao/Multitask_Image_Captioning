from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import os
import sys
import misc.utils as utils
import torch.nn.functional as F

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def language_eval(dataset, preds, model_id, split):
    sys.path.append("coco-caption")    
    annFile = 'coco-caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    
    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    
    coco = COCO(annFile)    
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(cnn_model, model, crit, loader, eval_kwargs={}, new_features=False):    
    verbose = eval_kwargs.get('verbose', False)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')

    cnn_model.eval()            
    model.eval()
    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        #evaluate loss if we have the labels
        loss = 0        
        torch.cuda.synchronize()
        if new_features:
#            tmp = [data['images'], data.get('labels', np.zeros(1)), data.get('masks', np.zeros(1))]
#            tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
#            images, labels, masks = tmp            
#            att_feats, _ = _att_feats, _ = cnn_model(images)             
#            fc_feats = _fc_feats = att_feats.mean(3).mean(2).squeeze()            
#            att_feats = _att_feats = F.adaptive_avg_pool2d(att_feats,[14,14]).squeeze().permute(0, 2, 3, 1)                        
#            att_feats = att_feats.unsqueeze(1).expand(*((att_feats.size(0), loader.seq_per_img,) + att_feats.size()[1:])).contiguous().view(*((att_feats.size(0) * loader.seq_per_img,) + att_feats.size()[1:]))
#            fc_feats = fc_feats.unsqueeze(1).expand(*((fc_feats.size(0), loader.seq_per_img,) + fc_feats.size()[1:])).contiguous().view(*((fc_feats.size(0) * loader.seq_per_img,) + fc_feats.size()[1:]))            
            tmp = [data.get('labels', np.zeros(1)), data.get('masks', np.zeros(1))]
            tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            labels, masks = tmp            
            images = data['images']            
            _fc_feats = []
            _att_feats = []            
            for i in range(loader.batch_size):
                x = Variable(torch.from_numpy(images[i]), volatile=True).cuda()
                x = x.unsqueeze(0)
                att_feats, _ = cnn_model(x)         
                fc_feats = att_feats.mean(3).mean(2).squeeze()                    
                att_feats = F.adaptive_avg_pool2d(att_feats,[14,14]).squeeze().permute(1, 2, 0)#(0, 2, 3, 1)
                _fc_feats.append(fc_feats)
                _att_feats.append(att_feats)                
            _fc_feats = torch.stack(_fc_feats)
            _att_feats = torch.stack(_att_feats)            
            att_feats = _att_feats.unsqueeze(1).expand(*((_att_feats.size(0), loader.seq_per_img,) + \
                                                           _att_feats.size()[1:])).contiguous().view(*((_att_feats.size(0) * loader.seq_per_img,) + \
                                                           _att_feats.size()[1:]))            
            fc_feats = _fc_feats.unsqueeze(1).expand(*((_fc_feats.size(0), loader.seq_per_img,) + \
                                                          _fc_feats.size()[1:])).contiguous().view(*((_fc_feats.size(0) * loader.seq_per_img,) + \
                                                          _fc_feats.size()[1:]))  
        else:
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
            tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks = tmp
            
        # forward the model to get loss        
        if data.get('labels', None) is not None:                        
            if eval_kwargs.get("ccg",False)==False:
                loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:]).data[0]
            else:
                tmp = [data['ccg']]
                tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
                ccg = tmp
#                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'],data['ccg']]
#                tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
#                fc_feats, att_feats, labels, masks,ccg = tmp
                word_labels, ccg_labels= model(fc_feats, att_feats, labels, ccg)
                loss = crit(word_labels, labels[:,1:], masks[:,1:]).data[0]
     
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample        
        if new_features:
            fc_feats, att_feats = _fc_feats, _att_feats
        else:            
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
            tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            fc_feats, att_feats = tmp                
            
        # forward the model to also get generated samples for each image
        if eval_kwargs.get("ccg",False):
            seq, _,seq_ccg,___ = model.sample(fc_feats, att_feats, eval_kwargs)#model.module.sample(fc_feats, att_feats, eval_kwargs)
        else:
            seq, _ = model.sample(fc_feats.contiguous(), att_feats.contiguous(), eval_kwargs)#model.module.sample(fc_feats, att_feats, eval_kwargs)
        torch.cuda.synchronize()

        sents = utils.decode_sequence(loader.get_vocab(), seq)
        if eval_kwargs.get("ccg",False):
            sents_ccg = utils.decode_sequence(loader.get_vocab_ccg(),seq_ccg)
        for k, sent in enumerate(sents):
            if eval_kwargs.get("ccg",False):
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent,"caption_ccg":sents_ccg[k]}
            else:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            predictions.append(entry)
           
            if verbose and random.random()<0.0001 :
                print('image %s: %s' %(entry['image_id'], entry['caption']))
                if eval_kwargs.get("ccg",False):
                    print('image %s: %s' %(entry['image_id'], entry['caption_ccg']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):            
            predictions.pop()

        if verbose and ix0 % 2500 == 0:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))
            
        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats

def main():
    import opts    
    import misc.utils as utils        
    opt = opts.parse_opt()
    opt.caption_model ='topdown'
    opt.batch_size=10#512#32*4*4
    opt.id ='topdown'
    opt.learning_rate= 5e-4 
    opt.learning_rate_decay_start= 0 
    opt.scheduled_sampling_start=0 
    opt.save_checkpoint_every=5000#450#5000#11500
    opt.val_images_use=5000
    opt.max_epochs=50#30
    opt.start_from='save/rt'#"save" #None
    opt.language_eval = 1
    opt.input_json='data/meta_coco_en.json'
    opt.input_label_h5='data/label_coco_en.h5'
#    opt.input_json='data/coco_ccg.json' #'data/meta_coco_en.json'
#    opt.input_label_h5='data/coco_ccg_label.h5' #'data/label_coco_en.h5'
#    opt.input_fc_dir='/nlp/andyweizhao/self-critical.pytorch-master/data/cocotalk_fc'
#    opt.input_att_dir='/nlp/andyweizhao/self-critical.pytorch-master/data/cocotalk_att'
    opt.finetune_cnn_after = 0
    opt.ccg = False
    opt.input_image_h5 = 'data/coco_image_512.h5'
                         
    opt.use_att = utils.if_use_att(opt.caption_model)
    
    from dataloader import DataLoader # just-in-time generated features
    loader = DataLoader(opt)
    
#    from dataloader_fixcnn import DataLoader # load pre-processed features
#    loader = DataLoader(opt)
    
    opt.vocab_size = loader.vocab_size
    opt.vocab_ccg_size = loader.vocab_ccg_size
    opt.seq_length = loader.seq_length
    
    import models
    model = models.setup(opt)    
    cnn_model = utils.build_cnn(opt)
    cnn_model.cuda()
    model.cuda()
    
    data = loader.get_batch('train')    
    images = data['images']
    
#    _fc_feats_2048 = []
#    _fc_feats_81 = []
#    _att_feats = []            
#    for i in range(loader.batch_size):
#        x = Variable(torch.from_numpy(images[i]), volatile=True).cuda()
#        x = x.unsqueeze(0)
#        att_feats, fc_feats_81 = cnn_model(x)         
#        fc_feats_2048 = att_feats.mean(3).mean(2).squeeze()                    
#        att_feats = F.adaptive_avg_pool2d(att_feats,[14,14]).squeeze().permute(1, 2, 0)#(0, 2, 3, 1)
#        _fc_feats_2048.append(fc_feats_2048)
#        _fc_feats_81.append(fc_feats_81)
#        _att_feats.append(att_feats)                
#    _fc_feats_2048 = torch.stack(_fc_feats_2048)
#    _fc_feats_81 = torch.stack(_fc_feats_81)
#    _att_feats = torch.stack(_att_feats)            
#    att_feats = _att_feats.unsqueeze(1).expand(*((_att_feats.size(0), loader.seq_per_img,) + \
#                                                   _att_feats.size()[1:])).contiguous().view(*((_att_feats.size(0) * loader.seq_per_img,) + \
#                                                   _att_feats.size()[1:]))            
#    fc_feats_2048 = _fc_feats_2048.unsqueeze(1).expand(*((_fc_feats_2048.size(0), loader.seq_per_img,) + \
#                                                  _fc_feats_2048.size()[1:])).contiguous().view(*((_fc_feats_2048.size(0) * loader.seq_per_img,) + \
#                                                  _fc_feats_2048.size()[1:]))   
#    fc_feats_81 = _fc_feats_81        
#                              
#    att_feats = Variable(att_feats, requires_grad=False).cuda()
#    Variable(fc_feats_81)
    
    crit = utils.LanguageModelCriterion()  
    eval_kwargs = {'split': 'val','dataset': opt.input_json,'verbose':True}
    eval_kwargs.update(vars(opt))
    val_loss, predictions, lang_stats = eval_split(cnn_model, model, crit, loader, eval_kwargs, True)
    
#    from models.AttModel import TopDownModel
#    model = TopDownModel(opt)
#
#    import torch.optim as optim
#    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
#    cnn_optimizer = optim.Adam([\
#            {'params': module.parameters()} for module in cnn_model._modules.values()[5:]\
#            ], lr=opt.cnn_learning_rate, weight_decay=opt.cnn_weight_decay)
#    
#    cnn_optimizer.state_dict().keys()
#    import misc.resnet as resnet
#    net = getattr(resnet, opt.cnn_model)() 
##    net.load_state_dict(torch.load('save/'+opt.cnn_weight))
#    net.load_state_dict(torch.load('save/rt/model-cnn.pth'))
##    cnn_model = net
##    net.state_dict().keys()
#    net = nn.Sequential(\
#        net.conv1,
#        net.bn1,
#        net.relu,
#        net.maxpool,
#        net.layer1,
#        net.layer2,
#        net.layer3,
#        net.layer4)
#    
#    net.load_state_dict(torch.load('save/'+opt.cnn_weight))
        
#main()
