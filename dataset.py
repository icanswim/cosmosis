from abc import ABC, abstractmethod
import os, re, random, h5py, pickle

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np

from torch.utils.data import Dataset, ConcatDataset
from torch import as_tensor, cat, squeeze

from torchvision import datasets as tvds

from sklearn import datasets as skds

from PIL import ImageFile, Image, ImageStat
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CDataset(Dataset, ABC):
    """An abstract base class for cosmosis datasets
    features/targets = ['data','keys'], True/False (use features listed, all or none)
    embed_lookup = {'label': index}  
    ds_idx = list of indices or keys to be passed to the Sampler and Dataloader
    transform/target_transform = [Transformer_Class()]/False
    features/target_dtype = set with init params or with load_data
    """    
    def __init__ (self, *args, features=True, features_dtype='float32', 
                  targets=True, targets_dtype='float32'
                  embeds=False, embed_lookup={}, 
                  transform=False, target_transform=False, **kwargs):
        self.transform, self.target_transform = transform, target_transform
        self.embeds, self.embed_lookup = embeds, embed_lookup
        self.features, self.targets = features, targets
        self.features_dtype, self.targets_type = features_dtype, targets_dtype
        self.ds = self.load_data(*args, **kwargs)
        print('CDataset created...')
    
    def __getitem__(self, i):
        X, embed_idx, y = [], [], []
        
        if self.features:
            X = self._get_features(self.ds[i]['X'], self.features, dtype=self.features_dtype)
            if self.transform: 
                X = self.transform(X)
        
        if self.embeds:
            embed_idx = self._get_embed_idx(self.ds[i]['embeds'], self.embeds, self.embed_lookup)
        
        if self.targets:
            y = self._get_features(self.ds[i]['targets'], self.targets, dtype=self.targets_dtype)
            if self.target_transform: 
                y = self.target_transform(y)
            
        return X, embed_idx, y
            
    def __iter__(self):
        for i in self.ds_idx:
            yield self.__getitem__(i)
    
    def __len__(self):
        return len(self.ds_idx)
    
    def _get_features(self, datadic, features, dtype):
        out = []
        if features == True: 
            features = datadic.keys()
        for f in features:
            out.append(np.reshape(np.asarray(datadic[f]), -1).astype(dtype))
        return as_tensor(np.concatenate(out))
        
    def _get_embed_idx(self, datadic, embeds, embed_lookup):
        embed_idx = []
        if embeds == True: embeds = datadic.keys()
        for e in embeds:
            embed_idx.append(np.reshape(np.asarray(embed_lookup[datadic[e]]), -1)
                                                                     .astype('int64'))
        return as_tensor(np.concatenate(embed_idx))
    
    @abstractmethod
    def load_data(self):
        """Pass any keywords and return datadic with keys X, targets, embeds or
        load your own self.ds and implement __getitem__
        embed_lookup can be loaded or passed with class __init__ params
        set the self.ds_idx
        set the feature_dtype and target_dtype"""
        
        datadic = {1: {'X': {'feature_a': .01,
                             'feature_b': .02},
                       'embeds': {'feature_c': 'a',
                                  'feature_d': 'b'},
                       'targets': {'feature_e': .3,
                                   'feature_f': .4}},
                   2: {'X': {'feature_a': .03,
                             'feature_b': .04},
                       'embeds': {'feature_c': 'c',
                                  'feature_d': 'd'},
                       'targets': {'feature_e': .7,
                                   'feature_f': .8}}}        

        self.embed_lookup = {'a': 1,'b': 2,'c': 3,'d': 4}
        self.features_dtype = 'float32'
        self.targets_dtype = 'float32'
        self.ds_idx = list(datadic.keys())
        
        return datadic
       
    
class ImStat(ImageStat.Stat):
    """A class for calculating a PIL image mean and std dev"""
    def __add__(self, other):
        return ImStat(list(map(np.add, self.h, other.h)))
    
class ImageDatasetStats():
    """A class for calculating an image datasets mean and std dev"""
    def __init__(self, dataset):
        self.stats = None
        i = 1
        print('images to process: {}'.format(len(dataset.ds_idx)))
        for image in dataset:
            if self.stats == None:
                self.stats = ImStat(image[0])
            else: 
                self.stats += ImStat(image[0])
                i += 1
            if i % 10000 == 0:
                print('images processed: {}'.format(i))
        print('mean: {}, stddev: {}'.format(self.stats.mean, self.stats.stddev))
                
class LoadImage():
    """A transformer for use with image file based datasets
    transforms (loads) an image filename into a PIL image"""
    def __call__(self, filename):
        return Image.open(filename)

class AsTensor():
    """Transforms a numpy array to a torch tensor"""
    def __call__(self, arr):
        return as_tensor(arr)
    
class Transpose():
    """Transforms a numpy array"""
    def __call__(self, arr):
        return np.transpose(arr)

class Squeeze():
    """Transforms a torch array"""
    def __call__(self, arr):
        return squeeze(arr)
    
class DType():
    """Transforms a numpy array"""
    def __init__(self, datatype):
        self.datatype = datatype
        
    def __call__(self, arr):
        return arr.astype(self.datatype)
    
class TVDS(CDataset):
    """A wrapper for torchvision.datasets
    dataset = torchvision datasets class name str ('FakeData')
    tv_params = dict of torchvision.dataset parameters ({'size': 1000})
    """
    def __init__(self, dataset='FakeData', 
                       tv_params={'size': 1000, 'image_size': (3,244,244),
                                  'num_classes': 10, 'transform': None,
                                  'target_transform': None}):
        super().__init__(dataset, tv_params)
        print('TVDS created...')
        
    def __getitem__(self, i):
        X = self.ds[i][0]
        #X = np.reshape(np.asarray(self.ds[i][0]), -1).astype(np.float32)
        y = self.ds[i][1]
        #y = np.squeeze(np.asarray(self.ds[i][1]).astype(np.int64))
        
        return X, [], y

    def load_data(self, dataset, tv_params):
        ds = getattr(tvds, dataset)(**tv_params)
        self.ds_idx = list(range(len(ds)))
        return ds
        
        
class SKDS(CDataset):
    """A wrapper for sklearn.datasets
    https://scikit-learn.org/stable/datasets/sample_generators.html
    make = sklearn datasets method name str ('make_regression')
    sk_params = dict of sklearn.datasets parameters ({'n_samples': 100})
    
    """    
    def __init__(self, *args, make='make_regression', 
                 sk_params={'n_samples': 100, 'n_features': 128}, **kwargs):
        super().__init__(make, sk_params, *args, **kwargs)
        print('SKDS {} created...'.format(make))
        
    def load_data(self, make, sk_params):
        ds = getattr(skds, make)(**sk_params)
        datadic = {}
        for i in range(len(ds[0])):
            datadic[i] = {'X': {'Xs': ds[0][i-1]},
                          'targets': {'ys': ds[1][i-1]},
                          'embeds': None}
        self.ds_idx = list(datadic.keys())
        return datadic

        
