from abc import ABC, abstractmethod
import os, re, random, h5py, pickle

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np

from torch.utils.data import Dataset, ConcatDataset
from torch import as_tensor, squeeze, is_tensor, cat

from PIL import ImageFile, Image, ImageStat
ImageFile.LOAD_TRUNCATED_IMAGES = True

#scikit and torchvision datasets are imported by their wrapper classes SKDS and TVDS


class CDataset(Dataset, ABC):
    """An abstract base class for cosmosis datasets
    
    embed_param = {'feature': (voc,vec,padding_idx,trainable),
                   'feature_3': (4,16,0,True),
                   'feature_4': (5,16,0,True),
                   'some_param': True}
    
    lookup_feature_3 = ExampleDataset.embed_lookup['feature_3']
    lookup_feature_4 = ExampleDataset.embed_lookup['feature_4']
    lookup_feature_6 = ExampleDataset.embed_lookup['feature_6']
    
    ds_param = {'train_param': {'input_dict': {'model_input': {'X': ['feature_1','feature_2']},
                                                               'embedding_input': {'feature_3': ['feature_3'],
                                                                                   'feature_4': ['feature_4']},
                                                 'criterion_input': {'y': ['feature_5']}},
                                  'transforms': {'feature_1': [ExampleTransform(10), AsTensor()],
                                                 'feature_2': [Reshape(-1), AsTensor()],
                                                 'feature_3': [Pad1d(5), EmbedLookup(lookup_feature_3), AsTensor()],
                                                 'feature_4': [Pad1d(5), EmbedLookup(lookup_feature_4), AsTensor()],
                                                 'feature_5': [AsTensor()],
                                                 'feature_6': [EmbedLookup(lookup_feature_6), AsTensor()]},
                                  'boom': 'bang'}}
                                      
            structure of the input_dict determines the structure of the output data_dict
            keywords: 'criterion_input','model_input','embed','target'
        
    ds_idx = [1,2,3,...]  
        a list of indices or keys (ints or strings) to be passed to the Sampler and Dataloader
        
    transforms = {'feature_1': [Pad1d(5), Flatten()]}
        keys are the feature name or index, values are a list of transforms in order of operation
        
    output should be a single object (dict, Data, tensor) which is parsed in Learn() and then again
    in the CModel()
    """    
    def __init__ (self, input_dict=None, transforms={}, **kwargs):
        self.input_dict = input_dict
        self.transforms = transforms
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
        self.ds_idx = [1,2,5,17,...] #some subset
            if no ds_idx provided the entire dataset will be used, 
            optionally this could be passed to the Selector/Sampler class in its sample_param
        """
        #zero is the lookup for the padding index
        self.embed_lookup = {'feature_4': {'a': 1,'b': 2,'c': 3,'d': 4, '0': 0},
                             'feature_3': {'z1': 1, 'y1': 2, 'x1': 3, '0': 0},
                             'feature_6': {'e': 1, 'f': 2, 'g': 3, '0': 0}}
        
        datadic = {1: {'feature_1': np.asarray([.04]),
                       'feature_2': np.asarray([[.02,.03],[.04,.05]]),
                       'feature_3': np.asarray(['z1']),
                       'feature_4': np.asarray(['c','c','d']),
                       'feature_5': np.asarray([1.1]),
                       'feature_6': np.asarray(['e','f','g'])},
                   2: {'feature_1': np.asarray([.03]),
                       'feature_2': np.asarray([[.1,.2],[.3,.4]]),
                       'feature_3': np.asarray(['x1','z1','y1']),
                       'feature_4': np.asarray(['d','a','d']),
                       'feature_5': np.asarray([1.2]),
                       'feature_6': np.asarray(['f','f','g'])}}
        
        return datadic
    
    def __iter__(self):
        for i in self.ds_idx:
            yield self.__getitem__(i)
    
    def __len__(self):
        return len(self.ds_idx)
    
    def __getitem__(self, i):         
        """if no input_dict is give then the dataset's native __getitem__ is used"""
        if self.input_dict == None:
            return self.ds[i]

        datadic = {}
        for input_key, features in self.input_dict.items():
            datadic[input_key] = self._get_features(self.ds[i], features)
        return datadic
    
    def _get_features(self, data, features):
        """load, transform then concatenate selected features"""
        output = []
        for f in features:
            if type(data) == dict: 
                out = data[f]
            else:
                out = getattr(data, f)

            if f in self.transforms:
                transforms = self.transforms[f] #get the list of transforms for this feature
                for T in transforms:
                    out = T(out)
                
            output.append(out)
            
        if len(output) == 1: return output[0] 
        elif is_tensor(output[0]): return cat(output)
        else: return np.concatenate(output)


class ExampleDataset(CDataset):
    #zero is the lookup for the padding index
    embed_lookup = {'feature_4': {'a': 1,'b': 2,'c': 3,'d': 4, '0': 0},
                    'feature_3': {'z1': 1, 'y1': 2, 'x1': 3, '0': 0},
                    'feature_6': {'e': 1, 'f': 2, 'g': 3, '0': 0}}
    
    def load_data(self, boom='bust'):
        
        datadic = {1: {'feature_1': np.asarray([.04]),
                       'feature_2': np.asarray([[.02,.03],[.04,.05]]),
                       'feature_3': np.asarray(['z1']),
                       'feature_4': np.asarray(['c','c','d']),
                       'feature_5': np.asarray([1.1]),
                       'feature_6': np.asarray(['e','f','g'])},
                   2: {'feature_1': np.asarray([.03]),
                       'feature_2': np.asarray([[.1,.2],[.3,.4]]),
                       'feature_3': np.asarray(['x1','z1','y1']),
                       'feature_4': np.asarray(['d','a','d']),
                       'feature_5': np.asarray([1.2]),
                       'feature_6': np.asarray(['f','f','g'])}}
        
        print(boom)
        return datadic
    
class EmbedLookup():
    """A transform which converts a list of categorical features to an array of ints which
    can then be fed to an embedding layer
    
    arr = numpy array or list of categorical values
    embed_lookup = {'feature': int, '0': 0} # 0 is padding value
    """
    def __init__(self, embed_lookup={}):
        self.embed_lookup = embed_lookup

    def __call__(self, arr):
        idx = []
        for i in np.reshape(arr, -1).tolist():
            idx.append(np.reshape(np.asarray(self.embed_lookup[i]), -1).astype('int64'))
        return np.hstack(idx)
    
class Pad1d():
    """Transforms a numpy array"""
    def __init__(self, pad):
        self.pad = pad
    def __call__(self, arr):
        return np.pad(arr, (0, (self.pad - arr.shape[0])))
        
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
                self.stats = ImStat(data['image'])
            else: 
                self.stats += ImStat(data['image'])
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
        if type(arr) == list:
            return [as_tensor(arr[0])] #embedding indices
        else:
            return as_tensor(arr)
        
class AsSparse():
    """Transforms a numpy array to a torch sparse tensor"""
    def __call__(self, arr):
        return as_tensor(arr).to_sparse()
        
class Reshape():
    """Transforms a numpy array"""
    def __init__(self, shape):
        self.shape = shape
        
    def __call__(self, arr):
        return np.reshape(arr, self.shape)
    
class Flatten():
    """Transforms a numpy array"""
    def __call__(self, arr):
        return np.reshape(arr, -1)
    
class Concat():
    """Transforms a list of numpy arrays"""
    def __call__(self, data):
        return np.concatenate(data)
    
class Transpose():
    """Transforms a numpy array"""
    def __call__(self, arr):
        return np.transpose(arr)

class SqueezeT():
    """Transforms a torch array"""
    def __call__(self, arr):
        return squeeze(arr)
    
class SqueezeN():
    """Transforms a numpy array"""
    def __call__(self, arr):
        return np.squeeze(arr)
    
class DType():
    """Transforms a numpy array"""
    def __init__(self, datatype):
        self.datatype = datatype
        
    def __call__(self, arr):
        return arr.astype(self.datatype)
    
class Index():
    """Transforms a numpy array"""
    def __init__(self, i):
        self.i = i
    def __call__(self, arr):
        return np.reshape(arr[:,self.i], -1)

class ExpandN():
    """Transforms a numpy array"""
    def __init__(self, axis=0):
        self.axis = axis
    def __call__(self ,arr):
        return np.expand_dims(arr, axis=self.axis)
    
class TVDS(CDataset):
    """A wrapper for torchvision.datasets
    dataset = torchvision datasets class name str ('FakeData')
    tv_param = dict of torchvision.dataset parameters ({'size': 1000})
    """
    def __init__(self, **kwargs):
        print('creating torch vision {} dataset...'.format(kwargs['dataset']))
        super().__init__(**kwargs)
        
    def __getitem__(self, i):
        image = self.ds[i][0]
        label = self.ds[i][1]
        return {'image': image,
                'y': label}
        
    def load_data(self, dataset, tv_param):
        from torchvision import datasets as tvds
        ds = getattr(tvds, dataset)(**tv_param)
        self.ds_idx = list(range(len(ds)))
        return ds
        
        
class SKDS(CDataset):
    """A wrapper for sklearn.datasets
    https://scikit-learn.org/stable/datasets/sample_generators.html
    dataset = sklearn datasets method name str ('make_regression')
    sk_param = dict of sklearn.datasets parameters ({'n_samples': 100})
    """    
    def __init__(self, **kwargs):
        print('creating scikit learn {} dataset...'.format(kwargs['dataset']))
        super().__init__(**kwargs)
        
    def load_data(self, dataset, sk_param, features_dtype, targets_dtype):
        from sklearn import datasets as skds              
        ds = getattr(skds, dataset)(**sk_param)
        datadic = {}
        for i in range(len(ds[0])):
            datadic[i] = {'X': np.reshape(ds[0][i-1], -1).astype(features_dtype),
                          'y': np.reshape(ds[1][i-1], -1).astype(targets_dtype)}
        return datadic

    

    
    




