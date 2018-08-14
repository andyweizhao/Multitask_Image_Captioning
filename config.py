# -*- coding: utf-8 -*-
import opts
def get_cn_fixed_opts():
    opt = opts.parse_opt()
    opt.caption_model ='topdown'
    opt.batch_size=10
    #Pretrain
    opt.id ='topdown'
    opt.learning_rate= 5e-4 
    opt.learning_rate_decay_start= 0 
    opt.scheduled_sampling_start=0 
    opt.save_checkpoint_every=1300#11500
    opt.val_images_use=5000
    opt.max_epochs=40
    opt.start_from=None
    opt.input_json='data/meta/aitalk_meta.json'
    opt.input_label_h5='data/dataset/aitalk_label.h5'
    opt.input_fc_dir='/media/andyweizhao/Elements/CVPR/cocotalk_fc'
    opt.input_att_dir='data/cocotalk_att'
    return opt

def get_en_fixed_opts():
    opt = opts.parse_opt()
    opt.caption_model ='topdown'
    opt.batch_size=10
    #Pretrain
    opt.id ='topdown'
    opt.learning_rate= 5e-4 
    opt.learning_rate_decay_start= 0 
    opt.scheduled_sampling_start=0 
    opt.save_checkpoint_every=1300#11500
    opt.val_images_use=5000
    opt.max_epochs=40
    opt.start_from=None
#    opt.input_json='data/meta/aitalk_meta.json'
#    opt.input_label_h5='data/dataset/aitalk_label.h5'
    opt.input_json='/home/andyweizhao/wabywang/010/data/dataset/coco_processed.json'
    opt.input_label_h5='data/dataset/coco_label.h5'
    opt.input_fc_dir='/media/andyweizhao/Elements/CVPR/cocotalk_fc'
    opt.input_att_dir='data/cocotalk_att'
    return opt
def get_cn_opts():
    opt = opts.parse_opt()
    opt.caption_model ='cross_topdown'
    opt.batch_size=10
    #Pretrain
    opt.id ='topdown'
    opt.learning_rate= 5e-4 
    opt.learning_rate_decay_start= 0 
    opt.scheduled_sampling_start=0 
    opt.save_checkpoint_every=1300#11500
    opt.val_images_use=5000
    opt.max_epochs=40
    opt.start_from=None
    opt.input_json='data/meta/aitalk_meta.json'
    opt.input_label_h5='data/dataset/aitalk_label.h5'
    #    opt.input_json='data/dataset/tmp/aitalk_cross.json'
#    opt.input_label_h5='data/dataset/tmp/aitalk_cross_label.h5'
    opt.input_fc_dir='/media/andyweizhao/Elements/CVPR/cocotalk_fc'
    opt.input_att_dir='data/cocotalk_att'
    return opt

def get_en_opts():
    opt = opts.parse_opt()
    opt.caption_model ='cross_topdown'
    opt.batch_size=10
    #Pretrain
    opt.id ='topdown'
    opt.learning_rate= 5e-4 
    opt.learning_rate_decay_start= 0 
    opt.scheduled_sampling_start=0 
    opt.save_checkpoint_every=1300#11500
    opt.val_images_use=5000
    opt.max_epochs=40
    opt.start_from=None
    opt.input_json='/home/andyweizhao/wabywang/010/data/dataset/coco_processed.json'
    opt.input_label_h5='data/dataset/coco_label.h5'
    opt.input_fc_dir='/media/andyweizhao/Elements/CVPR/cocotalk_fc'
    opt.input_att_dir='data/cocotalk_att'
    return opt