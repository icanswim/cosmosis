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
    
    features/targets = ['data','keys'], None
    
    embed=['feature',voc,vec,padding_idx,param.requires_grad]
        'feature' = name/key of feature to be embedded
        voc = vocabulary size (int) (passed to CModel)
        vec = length of the embedding vectors (passed to CModel)
        padding_idx = False/int (passed to CModel)
        param.requires_grad = True/False (passed to CModel)
        
    embed_lookup = {'label': index}
        
    self.ds_idx = list of indices or keys to be passed to the Sampler and Dataloader
    
    """    
    def __init__ (self, features=None, targets=None, 
                  embeds=None, embed_lookup={}, 
                  transform=False, target_transform=False, **kwargs):
        self.transform, self.target_transform = transform, target_transform
        self.embeds, self.embed_lookup = embeds, embed_lookup
        self.features, self.targets = features, targets
        self.data = self.load_data(**kwargs)
        self.ds_idx = list(self.data.keys())
        print('CDataset created...')
    
    def __getitem__(self, i):
        X, y, embed_idx = [], [], []
        
        if self.features:
            X = self.data[i]['X']
            X = self._get_features(X, self.features, dtype='float32')
        if self.transform:
            X = self.transform(X)
               
        if self.targets:
            y = self.data[i]['targets']
            y = self._get_features(y, self.targets, dtype='float64')
        if self.target_transform:
            y = self.target_transform(y)
        
        if self.embeds:
            embeds = self.data[i]['embeds']
            embed_idx = self._get_embed_idx(embeds, self.embed_lookup)
        
        return X, y, embed_idx
    
    def _get_features(self, datadic, features, dtype):
        out = []
        for f in features:
            out.append(np.reshape(np.asarray(datadic[f], -1).astype(dtype)))
        return as_tensor(np.concatenate(out))
        
    def _get_embed_idx(self, datadic, embeds, embed_lookup):
        embed_idx = []
        for e in embeds:
            embed_idx.append(np.asarray(embed_lookup[e]).astype('int64'))
        return as_tensor(np.concatenate(embed_idx))
            
    def __iter__(self):
        for i in self.ds_idx:
            yield self.__getitem__(i)
    
    def __len__(self):
        return len(self.ds_idx)
    
    @abstractmethod
    def load_data(self):
        """datadic with keys X, target, embeds (with or without 'features')"""
        return {'X': {'feature': data}}
   
        
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
    
    subclass amd implement __getitem__ as needed
    """
    def __init__(self, dataset='FakeData', embed=[], 
                 tv_params={'size': 1000, 'image_size': (3,244,244),
                            'num_classes': 10, 'transform': None,
                            'target_transform': None}):
        self.load_data(dataset, tv_params)
        self.ds_idx = list(range(len(self.ds)))
        self.embed = embed
        print('TVDS created...')
        
    def __getitem__(self, i):
        
        X = self.ds[i][0]
        #X = np.reshape(np.asarray(self.ds[i][0]), -1).astype(np.float32)
        
        y = self.ds[i][1]
        #y = np.squeeze(np.asarray(self.ds[i][1]).astype(np.int64))
        
        return X, y, []
    
    def __len__(self):
        return len(self.ds)

    def load_data(self, dataset, tv_params):
        ds = getattr(tvds, dataset)
        self.ds = ds(**tv_params)
        
        
class SKDS(CDataset):
    """A wrapper for sklearn.datasets
    make = sklearn datasets method name str ('make_regression')
    sk_params = dict of sklearn.datasets parameters ({'n_samples': 100})
    embed=['feature',voc,vec,padding_idx,param.requires_grad]
    
    subclass amd implement __getitem__ or pass in tranforms or both as needed
    """    
    def __init__(self, make='make_regression', embed=[], embed_lookup={}, 
                 transform=None, target_transform=None, sk_params={'n_samples': 100, 
                                                                   'n_features': 128}):
        self.embed_lookup = embed_lookup
        self.transform, self.target_transform = transform, target_transform
        self.load_data(make, sk_params)
        self.ds_idx = list(range(self.data[0].shape[0]))
        self.embed = embed
        print('SKDS created...')
        
    def __getitem__(self, i):

        X = np.reshape(self.data[0][i], -1).astype(np.float32)
        if self.transform:
            X = self.transform(X)
            
        y = np.reshape(self.data[1][i], -1).astype(np.float32)
        if self.target_transform:
            y = self.target_transform(y)
        
        embed_idx = []
        for emb in self.embed:
            embed_idx.append(as_tensor(
                self.embed_lookup[self.data[2][i]]).astype(np.int64))
        
        return X, y, embed_idx
    
    def __len__(self):
        return self.data[0].shape[0]

    def load_data(self, make, sk_params):
        ds = getattr(skds, make)
        self.data = ds(**sk_params)
        
