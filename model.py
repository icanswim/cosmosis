from abc import ABC, abstractmethod

from math import sqrt

from torch import nn, cat, squeeze, softmax, Tensor, flatten
from torch.nn import functional as F

import torchvision.models as torchvisionmodels


def tv_model(model_name='resnet18', embed=[], tv_params={}, **kwargs):

    launcher = getattr(torchvisionmodels, model_name)
    model = launcher(**tv_params)
    
    if model_name in ['resnet18']:
        model.conv1 = nn.Conv2d(in_channels=kwargs['in_channels'], out_channels=64, 
                                kernel_size=7, stride=2, padding=3, bias=False)
    return model

class CModel(nn.Module):
    """A base class for cosmosis models
    embed = [('feature',n_vocab,len_vec,padding_idx,param.requires_grad),...]
    """
    def __init__(self, embed=[], **kwargs):
        super().__init__()
        #self.embeddings = self.embedding_layer(embed)
        #self.layers = nn.ModuleList()
        
    def embedding_layer(self, embed):
        if len(embed) == 0:
            return None
        else:
            embeddings = [nn.Embedding(voc, vec, padding_idx).to('cuda:0') \
                          for _, voc, vec, padding_idx, _ in embed]
            for i, e in enumerate(embed):
                param = embeddings[i].weight
                param.requires_grad = e[4]
            return embeddings

    def forward(self, X=None, embed=[]):
        """check for categorical and/or continuous inputs, get the embeddings and  
        concat as appropriate, feed to model. 
        
        embed = a list of torch.cuda tensor int64 indices to be fed to the embedding layer
            ex: [[1,2,1][5]] (2 different embeded features, 3 instances and 1 instance respectively)
        X = torch tensor of concatenated continuous feature vectors"""
        if len(embed) > 0:
            embedded = []
            for e, emb in enumerate(embed):
                out = self.embeddings[e](emb)
                embedded.append(flatten(out, start_dim=1))
            if len(embedded) > 1:
                embedded = cat(emb, dim=1)    
            if X is not None:
                X = cat([X, *embedded], dim=1)
            else:  
                X = embedded 
        
        for l in self.layers:
            X = l(X)
        return X
    
    def adapt(self, shape):
        """for adapting a dataset shape[0] to a saved model shape[1]
        shape = (data_shape, model_shape)"""
        # freeze the layers
        for param in self.parameters(): 
            param.requires_grad = False
        # prepend a trainable adaptor layer    
        for l in self.ffunit(shape[0], shape[1], 0.2)[::-1]:
            self.layers.insert(0, l)
            
    def ffunit(self, D_in, D_out, drop, activation=nn.SELU):
        ffu = []
        ffu.append(nn.BatchNorm1d(D_in))
        ffu.append(nn.Linear(D_in, D_out))
        ffu.append(activation())
        ffu.append(nn.Dropout(drop))
        return ffu
    
class FFNet(CModel):
    
    model_config = {}
    model_config['simple'] = {'shape': [('D_in',1),(1,1),(1,1/2),(1/2,'D_out')], 
                              'dropout': [.2, .3, .1]}
    model_config['funnel'] = {'shape': [('D_in',1),(1,1/2),(1/2,1/2),(1/2,1/4),(1/4,1/4),(1/4,'D_out')], 
                              'dropout': [.1, .2, .3, .2, .1]}

    def __init__(self, model_name='funnel', D_in=0, H=0, D_out=0, embed=[]):
        super().__init__()
        
        config = FFNet.model_config[model_name]
        layers = []
        layers.append(self.ffunit(D_in, int(config['shape'][0][1]*H), config['dropout'][0]))
        for i, s in enumerate(config['shape'][1:-1]):
            layers.append(self.ffunit(int(s[0]*H), int(s[1]*H), config['dropout'][i]))
        layers.append([nn.Linear(int(config['shape'][-1][0]*H), D_out)])
        self.layers = [l for ffu in layers for l in ffu] # flatten
        self.layers = nn.ModuleList(self.layers)  
    
        self.embeddings = self.embedding_layer(embed)
        
        

        
        

        