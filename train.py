from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
import torch.nn as nn
import eval_utils
import misc.utils as utils
import torch.nn.functional as F

from misc.rewards import get_self_critical_reward


def train(opt):
    opt.use_att = utils.if_use_att(opt.caption_model)
    
    from dataloader import DataLoader 
    loader = DataLoader(opt)
    
    opt.vocab_size = loader.vocab_size
    opt.vocab_ccg_size = loader.vocab_ccg_size
    opt.seq_length = loader.seq_length

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    cnn_model = utils.build_cnn(opt)
    cnn_model.cuda()

    model = models.setup(opt)
    model.cuda()
   # model = DataParallel(model)
   
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    update_lr_flag = True
    model.train()

    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    multilabel_crit = nn.MultiLabelSoftMarginLoss().cuda()
#    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
        print('finetune mode')
        cnn_optimizer = optim.Adam([\
            {'params': module.parameters()} for module in cnn_model._modules.values()[5:]\
            ], lr=opt.cnn_learning_rate, weight_decay=opt.cnn_weight_decay)
    
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):            
        if os.path.isfile(os.path.join(opt.start_from, 'optimizer.pth')):
            optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))
        if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
            if os.path.isfile(os.path.join(opt.start_from, 'optimizer-cnn.pth')):
                cnn_optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer-cnn.pth')))
    
    eval_kwargs = {'split': 'val','dataset': opt.input_json,'verbose':True}
    eval_kwargs.update(vars(opt))
    val_loss, predictions, lang_stats = eval_utils.eval_split(cnn_model, model, crit, 
                                                                          loader, eval_kwargs, True)    
    epoch_start = time.time()
    while True:
        if update_lr_flag:              
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate            
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
                #model.module.ss_prob = opt.ss_prob            
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
            else:
                sc_flag = False
            
            # Update the training stage of cnn
            for p in cnn_model.parameters():
                p.requires_grad = True
            # Fix the first few layers:
            for module in cnn_model._modules.values()[:5]:
                for p in module.parameters():
                    p.requires_grad = False
            cnn_model.train()  
            update_lr_flag = False
            
        cnn_model.apply(utils.set_bn_fix)
        cnn_model.apply(utils.set_bn_eval)    
       
        start = time.time()              
        torch.cuda.synchronize()
        data = loader.get_batch('train')                            
        if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:

            multilabels = [data['detection_infos'][i]['label'] for i in range(len(data['detection_infos']))]
            
            tmp = [data['labels'], data['masks'],np.array(multilabels,dtype=np.int16)]
            tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
            labels, masks, multilabels = tmp             
            images = data['images'] # it cannot be turned into tensor since different sizes.
            _fc_feats_2048 = []
            _fc_feats_81 = []
            _att_feats = []            
            for i in range(loader.batch_size):
                x = Variable(torch.from_numpy(images[i]), requires_grad=False).cuda()
                x = x.unsqueeze(0)
                att_feats, fc_feats_81 = cnn_model(x)         
                fc_feats_2048 = att_feats.mean(3).mean(2).squeeze()                    
                att_feats = F.adaptive_avg_pool2d(att_feats,[14,14]).squeeze().permute(1, 2, 0)#(0, 2, 3, 1)
                _fc_feats_2048.append(fc_feats_2048)
                _fc_feats_81.append(fc_feats_81)
                _att_feats.append(att_feats)                
            _fc_feats_2048 = torch.stack(_fc_feats_2048)
            _fc_feats_81 = torch.stack(_fc_feats_81)
            _att_feats = torch.stack(_att_feats)            
            att_feats = _att_feats.unsqueeze(1).expand(*((_att_feats.size(0), loader.seq_per_img,) + \
                                                           _att_feats.size()[1:])).contiguous().view(*((_att_feats.size(0) * loader.seq_per_img,) + \
                                                           _att_feats.size()[1:]))            
            fc_feats_2048 = _fc_feats_2048.unsqueeze(1).expand(*((_fc_feats_2048.size(0), loader.seq_per_img,) + \
                                                          _fc_feats_2048.size()[1:])).contiguous().view(*((_fc_feats_2048.size(0) * loader.seq_per_img,) + \
                                                          _fc_feats_2048.size()[1:]))   
            fc_feats_81 = _fc_feats_81             
#            
            cnn_optimizer.zero_grad()
        else:    

            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
            tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks = tmp                        
               
        optimizer.zero_grad()
                        
        if not sc_flag:
            loss1 = crit(model(fc_feats_2048, att_feats, labels), labels[:,1:], masks[:,1:])
            loss2 = multilabel_crit(fc_feats_81.double(), multilabels.double())
            loss = 0.8*loss1 + 0.2*loss2.float()
        else:            
            gen_result, sample_logprobs = model.sample(fc_feats_2048, att_feats, {'sample_max':0})
            reward = get_self_critical_reward(model, fc_feats_2048, att_feats, data, gen_result)
            loss1 = rl_crit(sample_logprobs, gen_result, Variable(torch.from_numpy(reward).float().cuda(), requires_grad=False))
            loss2 = multilabel_crit(fc_feats_81.double(), multilabels.double())
            loss3 = crit(model(fc_feats_2048, att_feats, labels), labels[:,1:], masks[:,1:])
            loss = 0.995*loss1 + 0.005*(loss2.float() + loss3)
        loss.backward()
        
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
                
        train_loss = loss.data[0]
        mle_loss = loss1.data[0]
        multilabel_loss = loss2.data[0]
        torch.cuda.synchronize()
        end = time.time()
        if not sc_flag and iteration % 2500==0:
            print("iter {} (epoch {}), mle_loss = {:.3f}, multilabel_loss = {:.3f}, train_loss = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, mle_loss, multilabel_loss, train_loss, end - start))        

        if sc_flag and iteration % 2500==0:
            print("iter {} (epoch {}), avg_reward = {:.3f}, mle_loss = {:.3f}, multilabel_loss = {:.3f}, train_loss = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, np.mean(reward[:,0]), mle_loss, multilabel_loss, train_loss, end - start))
        iteration += 1
        if (iteration % opt.losses_log_every == 0):
            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        if (iteration % opt.save_checkpoint_every == 0):
            eval_kwargs = {'split': 'val','dataset': opt.input_json,'verbose':True}
            eval_kwargs.update(vars(opt))
            
            if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
                val_loss, predictions, lang_stats = eval_utils.eval_split(cnn_model, model, crit, 
                                                                          loader, eval_kwargs, True)
            else:
                val_loss, predictions, lang_stats = eval_utils.eval_split(cnn_model, model, crit, 
                                                                          loader, eval_kwargs, False)                
                        
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}
           
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True:
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')                
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                
                cnn_checkpoint_path = os.path.join(opt.checkpoint_path, 'model-cnn.pth')
                torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
                print("cnn model saved to {}".format(cnn_checkpoint_path))                
                
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
                    cnn_optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer-cnn.pth')
                    torch.save(cnn_optimizer.state_dict(), cnn_optimizer_path)
                
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    
                    cnn_checkpoint_path = os.path.join(opt.checkpoint_path, 'model-cnn-best.pth')                    
                    torch.save(cnn_model.state_dict(), cnn_checkpoint_path)         
                    print("cnn model saved to {}".format(cnn_checkpoint_path))
                    
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)
                        
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True
            print("epoch: "+str(epoch)+ " during: " + str(time.time()-epoch_start))
            epoch_start = time.time()
       
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

def main():
    opt = opts.parse_opt()
    opt.caption_model ='topdown'
    opt.batch_size=10
    opt.id ='topdown'
    opt.learning_rate= 5e-5 
    opt.learning_rate_decay_start= -1 
    opt.scheduled_sampling_start=-1 
    opt.save_checkpoint_every=5000#
    opt.val_images_use=5000
    opt.max_epochs=60
    opt.start_from='save/multitask_pretrain'#"save" #None
    opt.language_eval = 1
    opt.input_json='data/meta_coco_en.json'
    opt.input_label_h5='data/label_coco_en.h5'
    opt.self_critical_after = 25
    opt.finetune_cnn_after = 0
    opt.ccg = False
    opt.input_image_h5 = 'data/coco_image_512.h5'
    opt.checkpoint_path = 'save/multitask_pretrain_rl'
    train(opt)
main()    
