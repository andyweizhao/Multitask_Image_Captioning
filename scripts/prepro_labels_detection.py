from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
from torch.autograd import Variable
import skimage.io
import pprint as pp
from shutil import copy2

images_root='/home/dl/dataset/MSCOCO'
karpathy_annFile='data/dataset_coco.json'
train_annFile='/nlp/dataset/MSCOCO/annotations/instances_train2014.json'
val_annFile='/nlp/dataset/MSCOCO/annotations/instances_val2014.json'

karpathy_dataset = json.load(open(karpathy_annFile,'r'))
#train_imgs = json.load(open(train_annFile,'r'))
#train_imgs['type']='instances'
#json.dump(train_imgs,open('data/instances_train2014_new.json',"w"))
#
#val_imgs = json.load(open(val_annFile,'r'))
#val_imgs['type']='instances'
#json.dump(val_imgs,open('data/instances_val2014_new.json',"w"))
  


#Copy images to corresponding folders
'''
for x in xrange(len(karpathy_imgs['images'])):
    img = karpathy_imgs['images'][x]
    file_path = os.path.join(images_root,img['filepath'],img['filename'])
    dest_path = os.path.join(images_root,'karpathy',img['split'])
    copy2(file_path,dest_path)
'''

import torchvision.datasets as datasets
import sys
sys.path.append("coco-caption")    
from pycocotools.coco import COCO
    
trainset = datasets.CocoDetection(root='/nlp/dataset/MSCOCO/train2014',
                                annFile='/nlp/dataset/MSCOCO/annotations/instances_train2014_new.json')

valset = datasets.CocoDetection(root='/nlp/dataset/MSCOCO/val2014',
                                  annFile='/nlp/dataset/MSCOCO/annotations/instances_val2014_new.json')


info = json.load(open('data/meta_coco_en.json'))
ix_to_word = info['ix_to_word']
word_to_ix = {}
for ix,w in ix_to_word.items():
    word_to_ix[w]=ix
        
category_table = train_imgs['categories'] 
category_dict = {}
for i in xrange(len(category_table)):
    cat_id = category_table[i]['id']
    cat_name = category_table[i]['name']
    super_cat_name = category_table[i]['supercategory']    
    label_id = i+1
    category_dict[cat_id] = {'label_id':i+1,'cat_name':cat_name,'super_cat_name':super_cat_name}    
    
pre_train = {}
pre_val = {}
for x in xrange(len(trainset)):
    img,target = trainset[x]
    if len(target) < 1 : continue    
    image_id = target[0]['image_id']
    label_list = [category_dict.get(target[i]['category_id'])['label_id'] 
                    for i in xrange(len(target))]        
    words,w_c = np.unique([word_to_ix.get(category_dict.get(target[i]['category_id'])['cat_name'],-1) 
                    for i in xrange(len(target))], return_counts=True) 
    w_c_dict = dict(zip(words.astype(int), w_c))
    w_c_dict.pop(-1, None)
    words = words.astype(int)
    words = words[words > 0]
    
    super_words,sw_c = np.unique([word_to_ix.get(category_dict.get(target[i]['category_id'])['super_cat_name'],-1) 
                    for i in xrange(len(target))], return_counts=True) 
    sw_c_dict = dict(zip(super_words.astype(int), sw_c))
    sw_c_dict.pop(-1, None)
    super_words = super_words.astype(int)
    super_words = super_words[super_words > 0]
    
    l, c = np.unique(label_list, return_counts=True)
    l_c_dict = dict(zip(l, c))
    l_c_dict[0] = 1
    
    label = np.zeros(80 + 1) # 0 for background    
    label[np.array(label_list,dtype=np.int16)] = 1
    label[0] = 1 # 0 for background    
    l_import = np.array( [l_c_dict.get(i,0)/(len(label_list) + 1) for i in range(len(label))])
    pre_train[image_id] = {'label':label,'words':words,
                         'super_words':super_words,
                         'l_import':l_import,
                         'w_import':w_c_dict,
                         'sw_import':sw_c_dict}

for x in xrange(len(valset)):
    img,target = valset[x]
    if len(target) < 1 : continue    
    image_id = target[0]['image_id']
    label_list = [category_dict.get(target[i]['category_id'])['label_id'] 
                    for i in xrange(len(target))]        
    words,w_c = np.unique([word_to_ix.get(category_dict.get(target[i]['category_id'])['cat_name'],-1) 
                    for i in xrange(len(target))], return_counts=True) 
    w_c_dict = dict(zip(words.astype(int), w_c))
    w_c_dict.pop(-1, None)
    words = words.astype(int)
    words = words[words > 0]
    
    super_words,sw_c = np.unique([word_to_ix.get(category_dict.get(target[i]['category_id'])['super_cat_name'],-1) 
                    for i in xrange(len(target))], return_counts=True) 
    sw_c_dict = dict(zip(super_words.astype(int), sw_c))
    sw_c_dict.pop(-1, None)
    super_words = super_words.astype(int)
    super_words = super_words[super_words > 0]
    
    l, c = np.unique(label_list, return_counts=True)
    l_c_dict = dict(zip(l, c))
    l_c_dict[0] = 1
    
    label = np.zeros(80 + 1) # 0 for background    
    label[np.array(label_list,dtype=np.int16)] = 1
    label[0] = 1 # 0 for background    
    l_import = np.array( [l_c_dict.get(i,0)/(len(label_list) + 1) for i in range(len(label))])
    pre_val[image_id] = {'label':label,'words':words,
                         'super_words':super_words,
                         'l_import':l_import,
                         'w_import':w_c_dict,
                         'sw_import':sw_c_dict}
   
detection_dataset = dict(pre_train.items() + pre_val.items())

from six.moves import cPickle
cPickle.dump(pre_train, open("data/detection_train.json", "w"))
cPickle.dump(pre_val, open("data/detection_val.json", "w"))
cPickle.dump(detection_dataset, open("data/detection_all.json", "w"))

n_label_cnt = 0
for x in xrange(len(karpathy_dataset['images'])):
    img_id =  karpathy_dataset['images'][x]['cocoid']    
    if (detection_dataset.has_key(img_id)):
        label = list(detection_dataset[img_id]['label'].astype(int))
        l_import = list(detection_dataset[img_id]['l_import'])
        super_words = list(detection_dataset[img_id]['super_words'])
        sw_import = detection_dataset[img_id]['sw_import']
        w_import = detection_dataset[img_id]['w_import']
        words = list(detection_dataset[img_id]['words'])
        karpathy_dataset['images'][x]['label'] = label
        karpathy_dataset['images'][x]['l_import'] = l_import
        karpathy_dataset['images'][x]['super_words'] = super_words
        karpathy_dataset['images'][x]['sw_import'] = sw_import
        karpathy_dataset['images'][x]['w_import'] = w_import
        karpathy_dataset['images'][x]['words'] = words
    else:
        label = list(np.zeros(81))
        n_label_cnt += 1        
        karpathy_dataset['images'][x]['label'] = label

    
coco_label_dataset = 'data/dataset_coco_label.json'
with open(coco_label_dataset,'w') as f:
    json.dump(karpathy_dataset['images'],f)

train_img_list = []
val_img_list = []
test_img_list = []
imgs = karpathy_dataset['images']

for x in xrange(len(imgs)):
    if imgs[x]['split'] == 'train':
        filename = imgs[x]['filename']
        path = os.path.join(images_root,'karpathy','train',filename) 
        label = imgs[x]['label']
        item = (path, label)
        train_img_list.append(item)
    if imgs[x]['split'] == 'restval':
        filename = imgs[x]['filename']
        path = os.path.join(images_root,'karpathy','train',filename) 
        label = imgs[x]['label']
        item = (path, label)
        train_img_list.append(item)
    if imgs[x]['split'] == 'val':
        filename = imgs[x]['filename']
        path = os.path.join(images_root,'karpathy','val',filename) 
        label = imgs[x]['label']
        item = (path, label)
        val_img_list.append(item)
    if imgs[x]['split'] == 'test':
        filename = imgs[x]['filename']
        path = os.path.join(images_root,'karpathy','test',filename) 
        label = imgs[x]['label']
        item = (path, label)
        test_img_list.append(item)  

train_json = os.path.join(images_root,'karpathy','train/data.json')
val_json = os.path.join(images_root,'karpathy','val/data.json')
test_json = os.path.join(images_root,'karpathy','test/data.json')

