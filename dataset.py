from abc import ABC, abstractmethod
import os, re, random, h5py, pickle

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np

from torch.utils.data import Dataset, ConcatDataset
from torch import as_tensor, squeeze

from PIL import ImageFile, Image, ImageStat
ImageFile.LOAD_TRUNCATED_IMAGES = True

#scikit and torchvision datasets are imported by their wrapper classes SKDS and TVDS


class CDataset(Dataset, ABC):
    """An abstract base class for cosmosis datasets
   input_dict = {'model_input': {'X1': ['feature1', 'feature2', ...],
                                  'X2': ['feature3', 'feature4', ...],
                                  'embeds': ['feature5, 'feature6', ...]},
                 'criterion_input': {'target': ['feature7', ...],
                                      'embeds': ['feature8, ...]}}
                                      
    ds_params = {'train_params': {'input_dict': {'model_input': {},
                                                 'criterion_input': {'target': tensor}}}}

    ds_idx = [1,2,3,...]  
        a list of indices or keys to be passed to the Sampler and Dataloader
    transform/target_transform = [Transformer_Class(),...]
    pad = int/None
        each feature will be padded to this length
    pad_feats = ['feature','feature'...]  
    
    """    
    def __init__ (self, input_dict={}, transform=[], target_transform=[], 
                  pad=None, pad_feats=[], flatten=False, 
                  as_tensor=True, **kwargs):
        self.input_dict = input_dict
        self.transform, self.target_transform = transform, target_transform
        self.pad, self.pad_feats = pad, pad_feats
        self.flatten, self.as_tensor = flatten, as_tensor
        self.ds = self.load_data(**kwargs)        
        if not hasattr(self, 'ds_idx'):
            try:
                self.ds_idx = list(self.ds.keys())
            except:
                NotImplemented("if dataset is not loaded as a dict \
                               load_data() must set self.ds_idx")
                
        print('CDataset created...')
        
    @abstractmethod
    def load_data(self, kwargs):
        """
        dont forget an embedding for the padding '0' (padding_idx)
        self.ds_idx = [1,2,5,17,...] #some subset
        if no ds_idx provided the entire dataset will be used, 
        optionally this could be passed to the Selector/Sampler class in its sample_params
        """
        datadic = {1: {'feature_1': np.asarray([.04]),
                       'feature_2': np.asarray([.02]),
                       'feature_3': np.asarray(['z1']),
                       'feature_4': np.asarray(['c','c','d']),
                       'feature_5': np.asarray([1.1])},
                   2: {'feature_1': np.asarray([.03]),
                       'feature_2': np.asarray([.01]),
                       'feature_3': np.asarray(['x1','z1','y1']),
                       'feature_4': np.asarray(['d','a','d']),
                       'feature_5': np.asarray([1.2])}}
        
        self.embed_lookup = {'feature_4': {'a': 1,'b': 2,'c': 3,'d': 4, '0': 0},
                             'feature_3': {'z1': 1, 'y1': 2, 'x1': 3, '0': 0}}

        return datadic
    
    def __iter__(self):
        for i in self.ds_idx:
            yield self.__getitem__(i)
    
    def __len__(self):
        return len(self.ds_idx)
       
    def __getitem__(self, i):
        """this func feeds the model's forward().  use the input_dict keys 
        to direct flow"""
        datadic = {'model_input': {},
                   'criterion_input': {}}
        
        for model_key in self.input_dict['model_input']:
            if model_key == 'embed':
                embed_idx = self._get_embed_idx(self.ds[i], 
                                                self.input_dict['model_input']['embed'], 
                                                self.embed_lookup)
                datadic['model_input']['embed_idx'] = embed_idx
            else:
                out = self._get_features(self.ds[i], self.input_dict['model_input'][model_key])
                for transform in self.transform:
                    out = transform(out)
                if self.as_tensor: out = as_tensor(out)
                datadic['model_input'][model_key] = out
        
        if 'criterion_input' in self.input_dict:
            for crit_key in self.input_dict['criterion_input']:
                if crit_key == 'embed':
                    embed_idx = self._get_embed_idx(self.ds[i], 
                                                    self.input_dict['criterion_input']['embed'], 
                                                    self.embed_lookup)
                    datadic['criterion_input']['embed_idx'] = embed_idx
                    
                else:
                    out = self._get_features(self.ds[i], self.input_dict['criterion_input'][crit_key])
                    for transform in self.target_transform:
                        out = transform(out)
                    if self.as_tensor: out = as_tensor(out)
                    datadic['criterion_input'][crit_key] = out
            
        return datadic
    
    def __getitem__(self, i): 
        datadic = {}
        for input_key in self.input_dict:
            datadic[input_key] = {}
            for output_key in self.input_dict[input_key]:
                print('input_key: ', input_key)
                print('output_key: ', output_key)
                if output_key == 'embed':
                    out = self._get_embed_idx(self.ds[i], 
                                              self.input_dict[input_key][output_key], 
                                              self.embed_lookup)
                else:
                    out = self._get_features(self.ds[i], 
                                             self.input_dict[input_key][output_key])
                    
                    if input_key == 'criterion_input':
                        for target_transform in self.target_transform:
                            out = target_transform(out)
                    else:
                        for transform in self.transform:
                            out = transform(out)
                        
                if self.as_tensor: out = as_tensor(out)
                datadic[input_key][output_key] = out
                
        return datadic
    
    def _get_features(self, datadic, features):
        """select which features to load"""
        data = []
        for f in features:                
            out = datadic[f]
            if f in self.pad_feats and self.pad != None:
                if len(self.pad) == 1:
                    out = np.pad(out, (0, (self.pad[0] - out.shape[0])))
                elif len(self.pad) == 2:
                    out = np.pad(out, ((0, (self.pad[0] - out.shape[0])), (0,(self.pad[1] - out.shape[1]))))       
            if self.flatten:
                out = np.reshape(out, -1)
            data.append(out)
        return np.concatenate(data)
        
    def _get_embed_idx(self, datadic, embeds, embed_lookup):
        """convert a list of 1 or more categorical features to an array of ints which can then
        be fed to an embedding layer
        datadic = {'feature_name': 'feature'}
        embeds = ['feature_name','feature_name']
        embed_lookup = {'feature_name': {'feature': int, '0': 0}}
            dont forget an embedding for the padding (padding_idx)
        pad_feats = ['feature_name','feature_name']
        """
        embed_idx = []
        for e in embeds:
            out = datadic[e]
            if e in self.pad_feats and self.pad != None:
                out = np.pad(out, (0, (self.pad[0] - out.shape[0])))
            idx = []        
            for i in np.reshape(out, -1).tolist():
                idx.append(np.reshape(np.asarray(embed_lookup[e][i]), -1).astype('int64'))
            idx = np.concatenate(idx)
            if self.as_tensor:  as_tensor(idx)
            embed_idx.append(idx)
            
        return embed_idx
    
    
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
        for data in dataset:
            if self.stats == None:
                self.stats = ImStat(data['model_input']['image'])
            else: 
                self.stats += ImStat(data['model_input']['image'])
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
        print('creating torch vision {} dataset...'.format(kwargs['dataset']))
        
        super().__init__(**kwargs)
        
    def __getitem__(self, i):
        image = self.ds[i][0]
        label = self.ds[i][1]
        return {'model_input': {'image': image},
                'criterion_input': {'target': label}}
        
    def load_data(self, dataset, tv_params):
        from torchvision import datasets as tvds
        ds = getattr(tvds, dataset)(**tv_params)
        self.ds_idx = list(range(len(ds)))
        return ds
        
        
class SKDS(CDataset):
    """A wrapper for sklearn.datasets
    https://scikit-learn.org/stable/datasets/sample_generators.html
    dataset = sklearn datasets method name str ('make_regression')
    sk_params = dict of sklearn.datasets parameters ({'n_samples': 100})
    """    
    def __init__(self, **kwargs):
        print('creating scikit learn {} dataset...'.format(kwargs['dataset']))
        super().__init__(**kwargs)
        
    def load_data(self, dataset, sk_params, features_dtype, targets_dtype):
        from sklearn import datasets as skds              
        ds = getattr(skds, dataset)(**sk_params)
        datadic = {}
        for i in range(len(ds[0])):
            datadic[i] = {'X': np.reshape(ds[0][i-1], -1).astype(features_dtype),
                          'y': np.reshape(ds[1][i-1], -1).astype(targets_dtype)}
        return datadic

    

    
    




