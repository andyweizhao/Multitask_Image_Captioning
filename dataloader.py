from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import numpy as np
import random
import torch
import cPickle
import skimage.io
from torchvision import transforms as trn
preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 
class DataLoader():    
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = self.opt.seq_per_img

        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']        
        self.ix_to_word_ccg = cPickle.load( open("data/ix_to_ccg.pkl","rb") ) 
        self.detection_dataset = cPickle.load(open("data/detection_all.json", 'rb'))
        
        self.vocab_size = len(self.ix_to_word)
        print('vocab word size is ', self.vocab_size)        
        self.vocab_ccg_size = len(self.ix_to_word_ccg)
        print('vocab ccg size is ', self.vocab_ccg_size)
            
        print('DataLoader loading h5 file: ', opt.input_label_h5, opt.input_image_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5)
        self.h5_image_file = h5py.File(self.opt.input_image_h5)
        self.h5_image_path = np.load('data/image_path.npy')
        
        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir

        # extract image size from dataset
        images_size = self.h5_image_file['images'].shape
        assert len(images_size) == 4, 'images should be a 4D tensor'
        assert images_size[2] == images_size[3], 'width and height must match'
        self.num_images = images_size[0]
        self.num_channels = images_size[1]
        self.max_image_size = images_size[2]
        print('read %d images of size %dx%dx%d' %(self.num_images, 
                    self.num_channels, self.max_image_size, self.max_image_size)) 

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval 
                # I used some of val for train, and that's "restval". So train/restval is train, val is val, test is test
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}
    
    def get_vocab_ccg(self):
        result=dict()
        for k, v in self.ix_to_word_ccg.items():
            result[str(k)]=v        
        return result
    
    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def get_batch(self, split, batch_size=None):
        split_ix = self.split_ix[split]    
        batch_size = batch_size or self.batch_size
        seq_per_img = self.seq_per_img or self.seq_per_img

#        img_batch = np.ndarray([batch_size, 3, 512,512], dtype = 'float32')
        img_batch = []
        label_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype = 'int')
        mask_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype = 'float32')
        ccg_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'int')
        
        max_index = len(split_ix)
        wrapped = False

        infos = []
        gts = []
        detection_infos = []
        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
            
            self.iterators[split] = ri_next
            ix = split_ix[ri]

#            img = self.h5_image_file['images'][ix, :, :, :]
            img_path = self.h5_image_path[ix]
            img_path = img_path.replace('/nlp/dataset/MSCOCO','/data1/zsfx/wabywang/caption/dataset/MSCOCO')
            img = skimage.io.imread(img_path)
            if len(img.shape) == 2:
                img = img[:,:,np.newaxis]
                img = np.concatenate((img,img,img), axis=2)
            img = img.transpose(2,0,1)                                        
            
#            img_batch[i] = preprocess(torch.from_numpy(img.astype('float32')/255.0)).numpy() 
            img_batch.append(preprocess(torch.from_numpy(img.astype('float32')/255.0)).numpy())
            # fetch the sequence labels
            ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
            ix2 = self.label_end_ix[ix] - 1
            ncap = ix2 - ix1 + 1 # number of captions available for this image
            assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

            if ncap < self.seq_per_img:                
                seq = np.zeros([self.seq_per_img, self.seq_length], dtype = 'int')
                ccg_seq = np.zeros([self.seq_per_img, self.seq_length], dtype = 'int')
                for q in range(self.seq_per_img):    
                    ixl = random.randint(ix1,ix2)
                    seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]  
                    if self.opt.ccg:
                        ccg_seq[q, :] = self.h5_label_file['ccg'][ixl, :self.seq_length]# zero with padding and starts with 1
            else:
                ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)# pick the last 5 captions
                seq = self.h5_label_file['labels'][ixl: ixl + self.seq_per_img, :self.seq_length]                
                if self.opt.ccg:
                    ccg_seq = self.h5_label_file['ccg'][ixl: ixl + self.seq_per_img, :self.seq_length]
            # leave bos and eos to 0
            if self.opt.ccg:
                ccg_batch[i * self.seq_per_img : (i + 1) * self.seq_per_img, 1 : self.seq_length + 1] = ccg_seq
            label_batch[i * self.seq_per_img : (i + 1) * self.seq_per_img, 1 : self.seq_length + 1] = seq
            
#           Used for reward evaluation
            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
        
            # record associated info as well
            info_dict = {}
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)
            
            detection_dict = {}
            if (self.detection_dataset.has_key(info_dict['id'])):
                img_id = info_dict['id']
                detection_dict['label'] = self.detection_dataset[img_id]['label'].astype(int)
                detection_dict['l_import'] = self.detection_dataset[img_id]['l_import']
                detection_dict['super_words'] = self.detection_dataset[img_id]['super_words']
                detection_dict['sw_import'] = self.detection_dataset[img_id]['sw_import']
                detection_dict['w_import'] = self.detection_dataset[img_id]['w_import']
                detection_dict['words'] = self.detection_dataset[img_id]['words']
            else:
                detection_dict['label'] = list(np.zeros(81))
            detection_infos.append(detection_dict)
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, label_batch)))        
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        data = {}
        data['images'] = img_batch
        data['labels'] = label_batch
        if self.opt.ccg:
            data['ccg'] = ccg_batch
        data['gts'] = gts
        data['masks'] = mask_batch 
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(split_ix), 'wrapped': wrapped}
        data['infos'] = infos
        data['detection_infos'] = detection_infos

        return data

    def reset_iterator(self, split):
        self.iterators[split] = 0
      
def main():
    import opts    
    import misc.utils as utils        
    opt = opts.parse_opt()
    opt.caption_model ='topdown'
    opt.batch_size=10
    opt.id ='topdown'
    opt.learning_rate= 5e-4 
    opt.learning_rate_decay_start= 0 
    opt.scheduled_sampling_start=0 
    opt.save_checkpoint_every=25#11500
    opt.val_images_use=5000
    opt.max_epochs=40
    opt.start_from=None
    opt.input_json='data/meta_coco_en.json'
    opt.input_label_h5='data/label_coco_en.h5'
    opt.input_image_h5 = 'data/coco_image_512.h5'    
    opt.use_att = utils.if_use_att(opt.caption_model)
    opt.ccg = False
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length    
    data = loader.get_batch('train')
    
    data = loader.get_batch('val')
