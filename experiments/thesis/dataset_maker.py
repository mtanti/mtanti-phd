import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import scipy.io
import json
import nltk
import re
import sys
import os

from framework import lib
from framework import config

sys.path.append(config.vgg16_dir)
from vgg16 import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess_sent(sent):
    sent = sent.strip().lower()
    sent = re.sub(r'[0-9]+[.,0-9]*', 'NUM', sent)
    sent = re.sub(r'[^a-zA-Z]+', ' ', sent)
    tokens = nltk.word_tokenize(sent)
    return tokens

class FeatureExtractor:
    def __init__(self, feature_container, vgg, sess):
        self.batch_size = 8
        self.feature_container = feature_container
        self.images = np.empty([self.batch_size, 224, 224, 3])
        self.fnames = [ '' for _ in range(self.batch_size) ]
        self.next_pos = 0
        self.vgg = vgg
        self.sess = sess
        self.feature_layer = self.vgg.fc2
        self.feature_layer_shape = [4096]
    
    def add(self, img, fname):
        self.images[self.next_pos] = img
        self.fnames[self.next_pos] = fname
        self.next_pos += 1
        if self.next_pos == self.batch_size:
            features = self.sess.run(self.feature_layer, feed_dict={self.vgg.imgs: self.images})
            for (feature, fname) in zip(features, self.fnames):
                self.feature_container[fname] = feature.reshape(self.feature_layer_shape)
            self.next_pos = 0
    
    def close(self):
        if self.next_pos > 0:
            features = self.sess.run(self.feature_layer, feed_dict={self.vgg.imgs: self.images[:self.next_pos]})
            for (feature, fname) in zip(features, self.fnames):
                self.feature_container[fname] = feature.reshape(self.feature_layer_shape)
            self.next_pos = 0

lib.create_dir(config.dataset_dir)

#####################################################
# Image caption datasets
#####################################################

with tf.Graph().as_default():
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, config.vgg16_dir+'/vgg16_weights.npz', sess)
    
for dataset_name in [ 'flickr8k', 'flickr30k', 'mscoco' ]:
    print(dataset_name)
    
    features = dict()
    dataset = {
        'train': { 'fnames': list(), 'sents': list() },
        'val': { 'fnames': list(), 'sents': list() },
        'test': { 'fnames': list(), 'sents': list() }
    }
    
    extractor = FeatureExtractor(features, vgg, sess)
    with open(config.data_dir(dataset_name)+'/dataset.json', 'r', encoding='utf-8') as f:
        for caption_data in json.load(f)['images']:
            split = caption_data['split']
            if split == 'restval':
                continue
            
            if config.debug == True and len(dataset[split]) >= 500:
                continue
            
            fname = caption_data['filename']
            
            img = imread(config.img_dir(dataset_name)+'/'+fname, mode='RGB')
            img = imresize(img, [224, 224])
            extractor.add(img, fname)
            
            dataset[split]['fnames'].append(fname)
            dataset[split]['sents'].append([ ' '.join(preprocess_sent(sent['raw'])) for sent in caption_data['sentences'] ])
    extractor.close()
    
    with open(config.dataset_dir+'/'+dataset_name+'.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f)
    for split in dataset.keys():
        np.save(config.dataset_dir+'/'+dataset_name+'_'+split+'.npy', [ features[fname] for fname in dataset[split]['fnames'] ])
    
sess.close()


#####################################################
# Text datasets
#####################################################

print('lm1b')
dataset = {'train': {'sents': list()}, 'val': {'sents': list()}}
for (dir_name, split) in [('training-monolingual.tokenized.shuffled', 'train'), ('heldout-monolingual.tokenized.shuffled', 'val')]:
    fnames = sorted(os.listdir(config.data_dir('lm1b')+'/'+dir_name))
    if split == 'val':
        fnames = fnames[:config.google_val_files_used]
    for fname in fnames:
        print('', fname)
        with open(config.data_dir('lm1b')+'/'+dir_name+'/'+fname, 'r', encoding='utf-8') as f:
            for line in f:
                if config.debug == True and len(dataset[split]['sents']) >= 500:
                    continue
                line = line.strip()
                if line == '':
                    continue
                    
                sent = preprocess_sent(line)
                if 1 < len(sent) <= config.google_max_sent_len:
                    dataset[split]['sents'].append(' '.join(sent))

with open(config.dataset_dir+'/lm1b.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f)
