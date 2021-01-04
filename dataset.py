from abc import ABC, abstractmethod
import os, re, random, h5py, pickle

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np

from torch.utils.data import Dataset, IterableDataset, ConcatDataset
from torch import as_tensor, cat

from sklearn import datasets


class CDataset(Dataset, ABC):
    """An abstract base class for cosmosis datasets
    embed = [(n_vocab, len_vec, param.requires_grad),...]
        The dataset reports if it has any categorical values it needs
        to encode and whether or not to train the embedding or fix it as a onehot
        and then serves up the values to be encoded as the x_cat component
        of the __getitem__ methods output.
    """
    @abstractmethod
    def __init__(self, in_file='./data/datafile'):
        self.load_data(in_file)
        self.embed = []  # [(n_vocab, len_vec, param.requires_grad),...]
        self.ds_idx = []  # list of the dataset's indices
    
    @abstractmethod
    def __getitem__(self, i):
        """set X and y and do preprocessing here
        Return continuous, categorical, target.  empty list if none.
        """
        return as_tensor(x_con[i]), as_tensor(x_cat[i]), as_tensor(target[i])  
    
    @abstractmethod
    def __len__(self):
        return len(self.ds_idx)
    
    @abstractmethod
    def load_data(self):
        return data
    
    def pad_data(self):
        """TODO: pad in multiple dimensions and trim"""
        pass

    
class Dummy(CDataset):
    """make = sklearn datasets method name str ('make_regression')
    make_params = dict of parameters ({'n_samples': 100})
    """
    def __init__(self, make='make_regression', make_params={'n_samples': 100,
                                                            'n_features': 128}):
        self.load_data(make, make_params)
        self.ds_idx = list(range(self.y.shape[0]))
        self.embed = []
        
    def __getitem__(self, i):
        return as_tensor(np.reshape(self.X[i], -1).astype(np.float32)), [], \
                        as_tensor(np.reshape(self.y[i], -1).astype(np.float32))
    
    def __len__(self):
        return self.y.shape[0]

    def load_data(self, make, make_params):
        ds = getattr(datasets, make)
        self.X, self.y = ds(**make_params)
        
        
class SuperSet(CDataset):
    
    def __init__(self, PrimaryDS, SecondaryDS, p_params, s_params):
        self.pds = PrimaryDS(**p_params)
        self.sds = SecondaryDS(**s_params)
        
        self.embed = self.pds.embed + self.sds.embed
        self.ds_idx = self.pds.ds_idx 
        
    def __getitem__(self, i):
        # lookup the molecule name used by the primary ds and use it to select data from 
        # the secondary ds and then concatenate both outputs and return it
        x_con1, x_cat1, y1 = self.pds[i]
        x_con2, x_cat2, y2 = self.sds[self.pds.lookup.iloc[i]]  # TODO H5 ds uses numpy indexing
       
        def concat(in1, in2, dim=0):
            try:
                return cat([in1, in2], dim=dim)
            except:
                if len(in1) != 0: return in1
                elif len(in2) != 0: return in2
                else: return []
                
        x_con = concat(x_con1, x_con2)
        x_cat = concat(x_cat1, x_cat2)
        return x_con, x_cat, y1
        
    def __len__(self):
        return len(self.ds_idx)
    
    def load_data(self):
        pass