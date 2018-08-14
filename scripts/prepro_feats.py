"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

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
from six.moves import cPickle
import numpy as np
import torch
#import torchvision.models as models
from torch.autograd import Variable
import skimage.io
#from torchvision import transforms as trn
#preprocess = trn.Compose([
#        #trn.ToTensor(),
#        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#])
from misc.resnet_utils import myResnet
import misc.resnet as resnet
from scipy.misc import imread, imresize

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(params):
#  net = getattr(resnet, params['model'])()
#  net.load_state_dict(torch.load(os.path.join(params['model_root'],params['model']+'.pth')))
#  my_resnet = myResnet(net)
#  my_resnet.cuda()
#  my_resnet.eval()

  imgs = json.load(open(params['input_json'], 'r'))
  imgs = imgs['images']
 
  seed(123) # make reproducible

#  dir_fc = params['output_dir']+'_fc'
#  dir_att = params['output_dir']+'_att'
#  if not os.path.isdir(dir_fc):
#    os.mkdir(dir_fc)
#  if not os.path.isdir(dir_att):
#    os.mkdir(dir_att)
  
  # create output h5 file
  N = len(imgs)
  resize = 512 #256,448

  f = h5py.File(params['output_h5'], "w")

  dset = f.create_dataset("images", (N,3,resize,resize), dtype='uint8') # space for resized images
#  img = imgs[1]
  dset = [] 
#  f = h5py.File('image_path2.h5', "w")
#  dset = f.create_dataset("images_path", (N,), dtype='str') # space for resized images  

  for i,img in enumerate(imgs):
    # load the image
#    I = imread(os.path.join(params['images_root'], img['filepath'], img['filename']))    
#    try:
#        Ir = imresize(I, (resize,resize))
#    except:
#        print('failed resizing image %s - see http://git.io/vBIE0' % (img['file_path'],))
#        raise
#    if len(Ir.shape) == 2:
#      Ir = Ir[:,:,np.newaxis]
#      Ir = np.concatenate((Ir,Ir,Ir), axis=2)
#    # and swap order of axes from (256,256,3) to (3,256,256)
#    Ir = Ir.transpose(2,0,1)
#    # write to h5
#    dset[i] = Ir
    
    I_path = os.path.join(params['images_root'], img['filepath'], img['filename'])
    dset.append(I_path)
    
    if i % 1000 == 0:
      print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
#  f.close()
#  print('wrote ', params['output_h5']+'_image.h5')

#np.save('image_path.npy', np.array(dset))

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json', default='/data1/zsfx/wabywang/caption/010/data/dataset_coco.json', help='input json file to process into hdf5')
#  parser.add_argument('--output_dir', default='data/cocotalk', help='output h5 file')
  parser.add_argument('--output_h5', default='coco_image_512.h5', help='output h5 file')

  # options
  parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')  
  parser.add_argument('--images_root', default='/data1/zsfx/wabywang/caption/dataset/MSCOCO', help='root location in which images are stored, to be prepended to file_path in input json')
#  parser.add_argument('--att_size', default=14, type=int, help='14x14 or 7x7')
#  parser.add_argument('--model', default='resnet101', type=str, help='resnet101, resnet152')
#  parser.add_argument('--model_root', default='./data/imagenet_weights', type=str, help='model root')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main(params)

#train = json.load(open('/nlp/dataset/Caption_CN/annotations/captions_train2017.json', 'r'))
#val = json.load(open('/nlp/dataset/Caption_CN/annotations/captions_val2017.json', 'r'))
#
#imgs1 = [{'cocoid':os.path.splitext(o.get('image_id'))[0],'filename':o.get('image_id'),'filepath':'train2017'} for o in train]
#imgs2 = [{'cocoid':os.path.splitext(o.get('image_id'))[0],'filename':o.get('image_id'),'filepath':'val2017'} for o in val]
#imgs = imgs1 + imgs2
#
#files = [ os.path.splitext(f)[0] for f in os.listdir('/nlp/andyweizhao/self-critical.pytorch_CN/data/cocotalk_fc/')]
#files = set(files)
#imgs = [img for img in imgs if img.get('cocoid') not in files]
#for o in b:            
#    img_id = o.get('image_id')    
#    if img_id not in image_list:
#        entry = {'image_id': img_id, 'caption': o.get('caption')}
#        image_list.append(img_id)
#        predictions.append(entry)
#json.dump(predictions, open('vis/val.json', 'w'))    