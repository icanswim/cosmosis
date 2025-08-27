from abc import ABC, abstractmethod
import os, re, random, h5py, pickle

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np

from torch.utils.data import Dataset, ConcatDataset
from torch import as_tensor, squeeze, is_tensor, cat, float32

from PIL import ImageFile, Image, ImageStat
ImageFile.LOAD_TRUNCATED_IMAGES = True

#scikit and torchvision datasets are imported by their wrapper classes SKDS and TVDS


class CDataset(Dataset, ABC):
    """Cosmosis Dataset
    
    An abstract base class for cosmosis datasets
    
    embed_param = {'feature': (voc,vec,padding_idx,trainable),
                   'feature_3': (4,16,0,True),
                   'feature_4': (5,16,0,True),
                   'some_param': True}
    
    vocab_feature_3 = ExampleDataset.vocab['feature_3']
    vocab_feature_4 = ExampleDataset.vocab['feature_4']
    vocab_feature_6 = ExampleDataset.vocab['feature_6']
    
    ds_param = {'train_param': {'input_dict': {
                                               'X2': ['feature_1','feature_2'], 
                                               'X3': ['feature_2'],
                                               'embed_3': ['feature_3'],
                                               'embed_4': ['feature_4'],
                                               'y': ['feature_5'],
                                                },
                                'transforms': {'feature_1': [ExampleTransform(10), AsTensor()],
                                               'feature_2': [Reshape(-1), AsTensor()],
                                               'feature_3': [Pad1d(5), Encode(vocab_feature_3), AsTensor()],
                                               'feature_4': [Pad1d(5), Encode(vocab_feature_4), AsTensor()],
                                               'feature_5': [AsTensor()],
                                               'feature_6': [Pad1d(5), Encode(vocab_feature_6), AsTensor()]},
                                  'boom': 'bang'}}           
    
    keywords: 'input_dict'
        
    ds_idx = [1,2,3,...]  
        a list of indices or keys (ints or strings) to be passed to the Sampler and Dataloader
        
    transforms = {'feature_1': [Pad1d(5), Flatten()]}
        keys are the feature name or index, values are a list of transforms in order of operation

    Returns {'X2': Tensor, 'X3': Tensor, 'embed_3': Tensor, 'embed_4': Tensor, 'y': Tensor}
    
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
        #zero is the vocab for the padding index
        self.vocab = {'feature_4': {'a': 1,'b': 2,'c': 3,'d': 4, '0': 0},
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
        """if no input_dict is given then the dataset's native __getitem__ is used"""
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
        elif is_tensor(output[0]): return cat(output, dim=-1)
        else: return np.concatenate(output, axis=-1)


class ExampleDataset(CDataset):
    #zero is the vocab for the padding index
    vocab = {'feature_4': {'a': 1,'b': 2,'c': 3,'d': 4, '0': 0},
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

class TDataset(CDataset):
    """Transfomer Dataset
    """
    def __getitem__(self, i):
        
        X = self.ds[i:i+self.d_seq].astype(np.int64)
        y = self.ds[i+1:i+1+self.d_seq].astype(np.int64)
        pos = np.arange(0, self.d_seq, dtype=np.int64) 
        
        _data = {'tokens': X, 'y': y, 'position': pos}
        data = {}
        
        for feature, Transforms in self.transforms.items():
            out = _data[feature]
            for T in Transforms:
                out = T(out)
            data[feature] = out
            
        del _data
        return data

    def prompt(self, prompt):
        # tokenize the prompt
        tokens = self.encoding.encode_ordinary(prompt)
        ds = np.array(tokens, dtype=np.uint16)
        self.d_seq = ds.shape[-1]
        self.ds_idx = [0]
        return ds

    @abstractmethod
    def load_data(self, d_seq=1, n=1, ds_name='', prompt=None, tokenizer=None):
        
        self.encoding = tokenizer
        self.d_seq, self.n = d_seq, n

        if prompt == None:
            ds = load_some_dataset()
            ds_idx = list(range(ds.shape[-1]-self.d_seq))
        
            if n != len(ds_idx): #subset
                ds_idx = list(np.random.choice(ds_idx, size=n, replace=False))
            self.ds_idx = ds_idx
        else:
            ds = self.prompt(prompt)
            self.ds_idx = [0]

        print('len(self.ds_idx): ', len(self.ds_idx))
        print('data.nbytes: ', ds.nbytes)
        return ds


class Encode():
    """A transform which converts a list of categorical features to an array of ints 
    (which can then be fed to an embedding layer)
    
    arr = numpy array or list of categorical values
    vocab = {'feature': int, 'pad_token': 0} # 0 is padding value
    """
    def __init__(self, vocab={}, pad_token='0', pad_value=0):
        self.vocab = vocab
        self.vocab[pad_token] = pad_value
        self.rev_vocab = {v: k for k, v in self.vocab.items()}

    def __call__(self, tokens):
        idx = []
        for i in np.reshape(tokens, -1):
            idx.append(np.reshape(np.asarray(self.vocab[i]), -1).astype('int64'))
            
        return np.hstack(idx)

    def decode(self, idx):
        tokens = []
        for i in np.reshape(idx, -1):
            tokens.append(self.rev_vocab[i])

        return tokens
    
    
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
    
class FlattenN():
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
    def __getitem__(self, i):
        image = self.ds[i][0]
        label = self.ds[i][1]
        return {'image': image,
                'y': label}
        
    def load_data(self, dataset, tv_param):
        print('creating torch vision {} dataset...'.format(dataset))
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
    def load_data(self, dataset, sk_param, features_dtype, targets_dtype):
        print('creating scikit learn {} dataset...'.format(dataset))
        from sklearn import datasets as skds              
        ds = getattr(skds, dataset)(**sk_param)
        datadic = {}
        for i in range(len(ds[0])):
            datadic[i] = {'X': np.reshape(ds[0][i-1], -1).astype(features_dtype),
                          'y': np.reshape(ds[1][i-1], -1).astype(targets_dtype)}
        return datadic

    

    
    




