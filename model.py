from abc import ABC, abstractmethod

from math import sqrt

from torch import nn, cat, squeeze, softmax, Tensor, flatten
from torch.nn import functional as F

import torchvision.models as torchvisionmodels


def tv_model(model_name='resnet18', embed=[], tv_params={}, **kwargs):

    launcher = getattr(torchvisionmodels, model_name)
    model = launcher(**tv_params)
    
    if model_name in ['resnet18','resnet34','resnet50','wide_resnet50_2','resnext50_32x4d']:
        model.conv1 = nn.Conv2d(in_channels=kwargs['in_channels'], out_channels=64, 
                                kernel_size=7, stride=2, padding=3, bias=False)
    print('TorchVision model {} loaded...'.format(model_name))
    return model

class CModel(nn.Module):
    """A base class for cosmosis models
    embed = [('feature',n_vocab,len_vec,padding_idx,param.requires_grad),...]
    """
    def __init__(self, embed=[], **kwargs):
        super().__init__()
        print('CModel loaded...')
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
            
    def ff_unit(self, D_in, D_out, drop, activation=nn.SELU):
        ffu = []
        ffu.append(nn.BatchNorm1d(D_in))
        ffu.append(nn.Linear(D_in, D_out))
        ffu.append(activation())
        ffu.append(nn.Dropout(drop))
        return nn.Sequential(*ffu)
    
    def attention(self):
        pass
    
    def conv_unit(self, in_channels, out_channels, kernel_size=3, 
                         stride=1, padding=1, dilation=1, bias=False):
        conv = []
        conv.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             stride=stride, padding=padding, dilation=dilation, bias=bias))
        conv.append(nn.BatchNorm2d(out_channels))
        conv.append(nn.ReLU())
        conv.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                             stride=stride, padding=padding, dilation=dilation, bias=bias))
        conv.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*conv)
    
    def bottleneck(self):
        pass
    
    def res_connect(self, in_channels, out_channels, kernel_size=1, stride=1)
        res = []
        res.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                                                     stride=stride))
        res.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*res)

class ResNet(CModel):
    
    def __init__(self, n_classes, in_channels, residuals=False, embed=[]):
        super().__init__()
        self.residuals = residuals
        layers = []
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, 
                                           padding=3, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.unit1 = self.conv_unit(64, 64, stride=1)
        
        self.unit2 = self.conv_unit(64, 128, stride=1)
        self.res2 = self.res_connect(64, 128, kernel_size=1, stride=1)
        self.unit3 = self.conv_unit(128, 256, stride=1)
        self.res3 = self.res_connect(128, 256, kernel_size=1, stride=1)
        self.unit4 = self.conv_unit(256, 512)
        self.res4 = self.res_connect(256, 512, kernel_size=1, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, n_classes)
        print('ResNet model loaded...')
        
    def forward(self, X):
        res = 0
        
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.activation(X)
        X = self.maxpool(X)
        
        if self.residuals:
            res = X.clone()
        X = self.unit1(X)
        X += res
        X = self.activation(X)
        
        if self.residuals:
            clone = X.clone()
            res = self.res2(clone)
        X = self.unit2(X)
        X += res
        X = self.activation(X)
        
        if self.residuals:
            clone = X.clone()
            res = self.res3(clone)
        X = self.unit3(X)
        X += res
        X = self.activation(X)
        
        if self.residuals:
            clone = X.clone()
            res = self.res4(clone)
        X = self.unit4(X)
        X += res
        X = self.activation(X)
        
        X = self.avgpool(X)
        X = flatten(X, 1)
        X = self.fc(X)
            
        return X

    
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
        layers.append(self.ff_unit(D_in, int(config['shape'][0][1]*H), config['dropout'][0]))
        for i, s in enumerate(config['shape'][1:-1]):
            layers.append(self.ff_unit(int(s[0]*H), int(s[1]*H), config['dropout'][i]))
        layers.append([nn.Linear(int(config['shape'][-1][0]*H), D_out)])
        self.layers = [l for ffu in layers for l in ffu] # flatten
        self.layers = nn.ModuleList(self.layers)  
    
        self.embeddings = self.embedding_layer(embed)
        print('FFNet model loaded...')
        
        