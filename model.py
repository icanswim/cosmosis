from math import sqrt

from torch import nn, cat, squeeze, softmax, Tensor, flatten, sigmoid, max, mean
from torch.nn import functional as F

# torchvision models are imported by its launcher tv_models()


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

     
class CModel(nn.Module):
    """A base class for cosmosis pytorch models
    model_param = {}
    device = 'cpu'/'cuda:0' 
    embed_param = {'feature': (voc, vec, padding_idx, trainable)}
        'feature' = name/key of feature to be embedded
        voc = vocabulary size (int) 
        vec = length of the embedding vectors (int)
        padding_idx = None/int 
        param.requires_grad = True/False
        
    datadict keywords: model_input, embedding_input
      
    """
    def __init__(self, model_param):
        super().__init__()
        
        self.device = 'cuda:0'
        if 'device' in model_param:  
            self.device = model_param['device']
        
        self.softmax = None
        self.X = 'X'
        self.y = 'y'

        if 'embed_param' in model_param:
            self.embed_param = model_param['embed_param']
            self.embedding_layer = self.create_embedding_layer()
            if 'flatten' not in self.embed_param:
                self.embed_param['flatten'] = False
        else:
            self.embed_param = None
            
        self.build(**model_param)    
        if hasattr(self, 'layers'):
            self.layers = nn.ModuleList(self.layers) 
        
        self.weight_init()
        
        print('CModel loaded...')
                            
    def build(self, **kwargs):
        self.layers = []
        raise NotImplementedError('subclass and implement build()...')
        
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                #nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                   
    def create_embedding_layer(self):
        """
        creates the embedding layer per the embed_param
        
        'feature' = name/key of feature to be embedded
        voc = int (vocabulary size)
        vec = int (embedding dimension length)
        padding_idx = 0 (the token used for padding)
        trainable = True/False (feed gradient back to the embedding)

        embed_param = {'feature':(voc,vec,padding_idx,trainable),
                       'feature_3':(4,16,0,True),
                       'some_param': True}

        returns embedded = {'feature': nn.Embedding(voc,vec,padding_idx,trainable),
                            'feature2': nn.Embedding(10,32,0,True)}
        """
        embedding_layer = {}
        
        for feature, param in self.embed_param.items():                
            if type(param) == tuple and len(param) == 4:
                voc, vec, padding_idx, trainable = param
                embedding_layer[feature] = nn.Embedding(voc, vec, padding_idx).to(self.device)
                embedding_layer[feature].weight.requires_grad = trainable
            else:
                continue
                
        return embedding_layer

    def embed_feature(self, data):
        """
        passes the tokens to the embedding layer
        
        embed_param = {'feature':(voc,vec,padding_idx,trainable),
                       'feature_3':(4,16,0,True),
                       'some_param': True}
        
        returns embedded = {'feature': torch.tensor,
                            'feature2': torch.tensor}
        """
        embedded = {}
        for feature, param in self.embed_param.items():
            if type(param) == tuple and len(param) == 4:
                if type(data) == dict:
                    embed = self.embedding_layer[feature](data[feature])
                elif hasattr(data, feature):
                    embed = self.embedding_layer[feature](data.feature)
                else:
                    embed = self.embedding_layer[feature](data) 
                embedded[feature] = embed

        return embedded

    def forward(self, data):

        if self.embed_param:
            embedded_dict = self.embed_feature(data)

            embedded = []
            for feature, embed in embedded_dict.items(): # mechanism for multiple embedding output
                if self.embed_param['flatten']:
                    embed = flatten(embed)
                feature, embedded.append(embed)
            embedded = cat(embedded)
            
            if type(data) == dict and len(data) != 0:
                X = []
                for k, v in data.items(): # mechanism for multiple model inputs
                    if k not in list(embedded_dict.keys()): # filter out already embedded features
                        X.append(v)
                X.append(embedded)
                X = cat(X, dim=0)      
            elif hasattr(data, self.X): 
                attr = self.X
                X = data.attr
                X = cat([X, embedded])
            else:
                X = embedded
        else:
            if type(data) == dict and len(data) != 0:
                X = []
                for k, v in data.items(): # mechanism for multiple model inputs
                    if k not in list(embedded_dict.keys()): # filter out already embedded features
                        X.append(v)
                X = cat(X, dim=0)   
            elif hasattr(data, self.X): 
                attr = self.X
                X = data.attr
            else:
                X = data
           
        for l in self.layers:
            X = l(X)
            
        if self.softmax is not None:
            X = getattr(F, self.softmax)(X, dim=1)

        return X
    
    def adapt(self, in_channels, out_channels, dropout):
        """prepends a trainable feedforward layer"""
        for l in self.ff_unit(in_channels, out_channels, activation=None, dropout=dropout)[::-1]:
            self.layers.insert(0, l)
            
    def ff_unit(self, in_channels, out_channels, activation=nn.ReLU, 
                                batch_norm=True, dropout=.2):
        ffu = []
        ffu.append(nn.Linear(in_channels, out_channels))
        if batch_norm: ffu.append(nn.BatchNorm1d(out_channels))
        if activation is not None: ffu.append(activation())
        if dropout is not None:  ffu.append(nn.Dropout(dropout))
        
        return nn.Sequential(*ffu)
    
    def conv_unit(self, in_channels, out_channels, kernel_size=3, 
                  stride=1, padding=1, dilation=1, groups=1, bias=False, 
                  activation=None, dropout=None, pool=(5,2,1)):
        conv = []
        conv.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias))
        conv.append(nn.BatchNorm2d(out_channels))
        if activation is not None: conv.append(activation())
        if pool is not None: conv.append(nn.MaxPool2d(kernel_size=pool[0], stride=pool[1], padding=pool[2]))
        conv.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, bias=bias))
        if dropout is not None: conv.append(nn.Dropout(p=dropout))
                        
        return nn.Sequential(*conv)

    
def tv_model(model_param):
    """A torchvision model launcher"""
    from torchvision import models as torchvisionmodels
    
    launcher = getattr(torchvisionmodels, model_param['model_name'])
    model = launcher(**model_param['tv_param'])

    def forward(data): # monkey patch
        return model._forward_impl(data['model_input']['image'])
    
    model.forward = forward
    print('torchvision model {} loaded...'.format(model_param['model_name'])) 
    return model
    
    
class SModel(CModel):
    """TODO:  A base class wrapper for cosmosis sklearn models
    """
    def build(self):
        pass

      
class FFNet(CModel):
    model_config = {}
    model_config['simple'] = {'shape': [('in_channels',1),(1,1),(1,1),(1,'out_channels')], 
                              'dropout': [.1, .2, .3],
                              'activation': nn.ReLU}
    model_config['funnel'] = {'shape': [('in_channels',1),(1,1),(1,1),
                                        (1,1/2),(1/2,1/2),(1/2,'out_channels')], 
                              'dropout': [.1, .2, .3, .1, .2],
                              'activation': nn.ReLU}

    def build(self, model_name='funnel', in_channels=0, hidden=0, out_channels=0, 
                      softmax=None, **kwargs):
        
        config = FFNet.model_config[model_name]
        self.layers = []
        
        self.layers.append(self.ff_unit(in_channels, int(config['shape'][0][1]*hidden),
                                        dropout=config['dropout'][0],
                                        batch_norm=True,
                                        activation=config['activation']))
        
        for i, s in enumerate(config['shape'][1:-1]):
            self.layers.append(self.ff_unit(int(s[0]*hidden), int(s[1]*hidden), 
                                            dropout=config['dropout'][i+1],
                                            batch_norm=True,
                                            activation=config['activation']))
            
        self.layers.append(self.ff_unit(int(config['shape'][-1][0]*hidden), out_channels,
                                        dropout=None,
                                        batch_norm=False,
                                        activation=None))
        
        self.softmax = softmax
        
        print('FFNet model loaded...')
        

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)


class GPT(CModel):

    def build(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=128, **kwargs):
        self.layers = []
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.layers.append(nn.TransformerDecoder(decoder_layer, num_layers))

    def forward(self, data):

        embedded_dict = self.embed_features(data)

        for l in self.layers:
            X = l(embedded_dict['X'], embedded_dict['X1'])
            
        if self.softmax is not None:
            X = getattr(F, self.softmax)(X, dim=1)

        return X


class IdentityModel(CModel):

    def build(self, *args, **kwargs):
        self.layers = []
        self.layers.append(nn.Identity())
        

