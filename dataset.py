from abc import ABC, abstractmethod
import os, re, random, h5py, pickle

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np

from torch.utils.data import Dataset, IterableDataset, ConcatDataset
from torch import as_tensor, cat

from torchvision import datasets as tvds
from torchvision import transforms as tvt

from sklearn import datasets as skds


class CDataset(Dataset, ABC):
    """An abstract base class for cosmosis datasets
    
    embed=['feature',voc,vec,padding_idx,param.requires_grad]
        'feature' = name/key of feature to be embedded
        voc = vocabulary size (int)
        vec = length of the embedding vectors
        padding_idx = False/int
        param.requires_grad = True/False
    
    """
    #embed_lookup = {'category_a': 1,
    #                'category_b': 2}
    
    @abstractmethod
    def __init__ (self, embed=None, in_file='./data/dataset/datafile'):
        self.embed, self.in_file = embed, in_file
        
        self.load_data()
        self.ds_idx = []
    
    @abstractmethod
    def __getitem__(self, i):
        """
        X = numpy float32 continuous values
        embed = numpy int 64 embedding indices
        y = numpy float64 continuous or discreet
        """
        X = self.data[0][i]
        embed = CDataset.embed_lookup[self.data[1][i]]
        y = self.data[2][i]
        
        return as_tensor(X), as_tensor(y) ,[as_tensor(embed),...] 
    
    @abstractmethod
    def __len__(self):
        return len(self.ds_idx)
    
    @abstractmethod
    def load_data(self):
        return data
    
class TVDS(CDataset):
    """A wrapper for torchvision.datasets
    dataset = torchvision datasets class name str ('FakeData')
    tv_params = dict of torchvision.dataset parameters ({'size': 1000})
    embed=['feature',voc,vec,padding_idx,param.requires_grad]
    
    subclass amd implement __getitem__ as needed
    """
    def __init__(self, dataset='FakeData', embed=None, 
                 tv_params={'size': 1000, 'image_size': (3,244,244),
                            'num_classes': 10, 'transform': None,
                            'target_transform': None}):
        self.load_data(dataset, tv_params)
        self.ds_idx = list(range(len(self.ds)))
        self.embed = embed
        
    def __getitem__(self, i):
        
        X = self.ds[i][0]
        y = self.ds[i][1]

        return as_tensor(X), as_tensor(y), []
    
    def __len__(self):
        return len(self.ds)

    def load_data(self, dataset, tv_params):
        ds = getattr(tvds, dataset)
        if tv_params['transform']:
            transforms = tvt.Compose([tvt.ToTensor()])
            tv_params['transform'] = transforms
        if tv_params['target_transform']:
            target_transforms = tvt.Compose([tvt.ToTensor()])
            tv_params['target_transform'] = target_transforms
        self.ds = ds(**tv_params)
        
        
class SKDS(CDataset):
    """A wrapper for sklearn.datasets
    make = sklearn datasets method name str ('make_regression')
    sk_params = dict of sklearn.datasets parameters ({'n_samples': 100})
    embed=['feature',voc,vec,padding_idx,param.requires_grad]
    
    subclass amd implement __getitem__ as needed
    """
    def __init__(self, make='make_regression', embed=None, 
                 sk_params={'n_samples': 100, 'n_features': 128}):
        self.load_data(make, sk_params)
        self.ds_idx = list(range(self.data[0].shape[0]))
        self.embed = embed
        
    def __getitem__(self, i):
        X = self.data[0][i].astype(np.float32)
        y = np.reshape(self.data[1][i], -1).astype(np.float32)
        
        return as_tensor(X), as_tensor(y), []     
    
    def __len__(self):
        return self.data[0].shape[0]

    def load_data(self, make, sk_params):
        ds = getattr(skds, make)
        self.data = ds(**sk_params)
        
