#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:56:13 2017

@author: nlp
"""
import os
i = 0
spath = '/data1/zsfx/wabywang/caption/010/data/aitalk_fc/'
tpath = '/data1/zsfx/wabywang/caption/010/data/combinedtalk_fc/'
for item in os.listdir(spath):
    i = i+1
    os.symlink(spath + item, 
               tpath + item)
    print(i)
    

    #os.path.join('data/cocotalk_fc', str(self.info['images'][ix]['id']) + '.npy'),
