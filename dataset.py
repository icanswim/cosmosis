from abc import ABC, abstractmethod
import os, re, random, h5py, pickle

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np

from torch.utils.data import Dataset, ConcatDataset
from torch import as_tensor, squeeze

from torchvision import datasets as tvds

from sklearn import datasets as skds

from PIL import ImageFile, Image, ImageStat
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CDataset(Dataset, ABC):
    """An abstract base class for cosmosis datasets
    features = ['featurename',...]
    embeds = ['featurename',...]
    targets = ['featurename',...]
    embed_lookup = {'label': index}
    ds_idx = [1,2,3,...]  
        a list of indices or keys to be passed to the Sampler and Dataloader
    transform/target_transform = [Transformer_Class(),...]
    pad = int/False  
        each feature will be padded to this length
    do_not_pad = ['featurename',...]  
        these features will not be padded
    """    
    def __init__ (self, features=[], targets=[], embeds=[], embed_lookup={}, 
                  transform=[], target_transform=[], pad=None, flatten=False, 
                  do_not_pad=[None], as_dict=False, **kwargs):
        self.transform, self.target_transform = transform, target_transform
        self.embeds, self.embed_lookup = embeds, embed_lookup
        self.features, self.targets = features, targets
        self.pad, self.do_not_pad = pad, do_not_pad
        self.flatten, self.as_dict = flatten, as_dict
        self.ds = self.load_data(**kwargs)
        try: 
            self.ds_idx = list(self.ds.keys())
        except: 
            pass
        print('CDataset created...')
    
    def __getitem__(self, i):
        X, embed_idx, y = [], [], []

        if len(self.features) > 0:
            X = self._get_features(self.ds[i], self.features)
            for transform in self.transform:
                X = transform(X)
        
        if len(self.embeds) > 0:
            embed_idx = self._get_embed_idx(self.ds[i], self.embeds, self.embed_lookup)

        
        if len(self.targets) > 0:
            y = self._get_features(self.ds[i], self.targets)
            for transform in self.target_transform:
                y = transform(y)

        if self.as_dict:
            datadict = {}
            datadict['X'] = X
            datadict['embed_idx'] = embed_idx
            datadict['y'] = y
            return datadict
        else:
            return X, embed_idx, y
            
    def __iter__(self):
        for i in self.ds_idx:
            yield self.__getitem__(i)
    
    def __len__(self):
        return len(self.ds_idx)
    
    def _get_features(self, datadic, features):
        data = []
        for f in features:
            out = datadic[f]
            if self.pad is not None:
                if f not in self.do_not_pad:
                    out = np.pad(out, (0, (self.pad - out.shape[0])))
            if self.flatten:
                out = np.reshape(out, -1)  
            data.append(out)
        return np.concatenate(data)
        
    def _get_embed_idx(self, datadic, embeds, embed_lookup):
        """convert a list of 1 or more categorical features to an array of ints which can then
        be fed to an embedding layer
        datadic = {'feature_name': 'feature'}
        embeds = ['feature_name','feature_name']
        embed_lookup = {'feature_name': int, '0': 0} 
            dont forget an embedding for the padding (padding_idx)
        do_not_pad = ['feature_name']
        """
        embed_idx = []
        for e in embeds:
            out = datadic[e]
            
            if self.pad is not None:
                if e not in self.do_not_pad:
                    out = np.pad(out, (0, (self.pad - out.shape[0])))
                    
            idx = []        
            for i in np.reshape(out, -1).tolist():
                idx.append(np.reshape(np.asarray(embed_lookup[i]), -1).astype('int64'))
            embed_idx.append(np.concatenate(idx))
            
        return embed_idx
        
    @abstractmethod
    def load_data(self):
        
        datadic = {1: {'feature_1': np.asarray([.04]),
                       'feature_2': np.asarray([.02]),
                       'feature_3': np.asarray(['b']),
                       'feature_4': np.asarray(['c','c','d']),
                       'feature_5': np.asarray([1.1])},
                   2: {'feature_1': np.asarray([.03]),
                       'feature_2': np.asarray([.01]),
                       'feature_3': np.asarray(['a']),
                       'feature_4': np.asarray(['d','d','d']),
                       'feature_5': np.asarray([1.2])}}
        
        self.embed_lookup = {'a': 1,'b': 2,'c': 3,'d': 4, '0': 0}
        #dont forget an embedding for the padding '0' (padding_idx)
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds_idx = list(range(len(self.ds)))
        print('TVDS created...')
        
    def __getitem__(self, i):
        X = self.ds[i][0]
        #X = np.reshape(np.asarray(self.ds[i][0]), -1).astype(np.float32)
        y = self.ds[i][1]
        #y = np.squeeze(np.asarray(self.ds[i][1]).astype(np.int64))
        return X, [], y

    def load_data(self, dataset, tv_params):
        ds = getattr(tvds, dataset)(**tv_params)
        return ds
        
        
class SKDS(CDataset):
    """A wrapper for sklearn.datasets
    https://scikit-learn.org/stable/datasets/sample_generators.html
    make = sklearn datasets method name str ('make_regression')
    sk_params = dict of sklearn.datasets parameters ({'n_samples': 100})
    """    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('SKDS {} created...'.format(kwargs['make']))
        
    def load_data(self, make, sk_params, features_dtype, targets_dtype):              
        ds = getattr(skds, make)(**sk_params)
        datadic = {}
        for i in range(len(ds[0])):
            datadic[i] = {'X': np.reshape(ds[0][i-1], -1).astype(features_dtype),
                          'y': np.reshape(ds[1][i-1], -1).astype(targets_dtype),
                          'embeds': None}

        return datadic
