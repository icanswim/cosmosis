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
    device = 'cpu'/'cuda:0' 
    embed_params = [('feature', voc, vec, padding_idx, param.requires_grad),...]
        'feature' = name/key of feature to be embedded
        voc = vocabulary size (int) 
        vec = length of the embedding vectors (int)
        padding_idx = None/int 
        param.requires_grad = True/False
        
    datadict keywords: 'X','embed'
        
    """
    def __init__(self, model_params):
        super().__init__()
        
        self.device = 'cuda:0'
        if 'device' in model_params:  
            self.device = model_params['device']
            
        self.build(**model_params)
        
        if 'embed_params' in model_params:
            self.embeddings = self.embedding_layer(model_params['embed_params'], self.device)
        
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
        
    def embedding_layer(self, embed_params, device):
        embeddings = [nn.Embedding(voc, vec, padding_idx).to(device) \
                      for _, voc, vec, padding_idx, _ in embed_params]
        for i, e in enumerate(embed_params):
            param = embeddings[i].weight
            param.requires_grad = e[4]
        return embeddings

    def forward(self, data):
        """data['X'] = torch tensor of concatenated continuous feature vectors
        embed = a list of lists (one for each feature) of torch.cuda tensor int64 
            indices (keys) to be fed to the embedding layer
        """
        if 'X' in data: 
            X = data['X']

        if 'embed' in data:
            embedded = []
            for e, idx in enumerate(data['embed']):
                out = self.embeddings[e](idx)
                embedded.append(flatten(out, start_dim=1))

            if len(embedded) > 1:
                embedded = cat(embedded, dim=1)
            else:
                embedded = embedded[0]

            if 'X' in data:
                X = cat([X, embedded], dim=1)
            else:  
                X = embedded
   
        for l in self.layers:
            X = l(X)
            
        return X
    
    def adapt(self, in_channels, out_channels, dropout):
        """prepends a trainable feedforward layer"""
        for l in self.ff_unit(in_channels, out_channels, activation=None, dropout=dropout)[::-1]:
            self.layers.insert(0, l)
            
    def ff_unit(self, in_channels, out_channels, activation=nn.ReLU, 
                                batch_norm=nn.BatchNorm1d, dropout=.2):
        ffu = []
        ffu.append(nn.Linear(in_channels, out_channels))
        if activation is not None: ffu.append(activation())
        if batch_norm is not None: ffu.append(batch_norm(out_channels))
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

    
def tv_model(model_params):
    """A torchvision model launcher"""
    from torchvision import models as torchvisionmodels
    
    launcher = getattr(torchvisionmodels, model_params['model_name'])
    model = launcher(**model_params['tv_params'])

    def forward(data): # monkey patch
        return model._forward_impl(data['image'])
    
    model.forward = forward
    print('torchvision model {} loaded...'.format(model_params['model_name'])) 
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
                              'batch_norm': nn.BatchNorm1d,
                              'activation': nn.ReLU}
    model_config['funnel'] = {'shape': [('in_channels',1),(1,1),(1,1),
                                        (1,1/2),(1/2,1/2),(1/2,'out_channels')], 
                              'dropout': [.1, .2, .3, .1, .2],
                              'batch_norm': nn.BatchNorm1d,
                              'activation': nn.ReLU}

    def build(self, model_name='funnel', in_channels=0, hidden=0, out_channels=0, **kwargs):
        config = FFNet.model_config[model_name]
        self.layers = []
        self.layers.append(self.ff_unit(in_channels, int(config['shape'][0][1]*hidden),                                                                   dropout=config['dropout'][0],
                                        batch_norm=config['batch_norm'],
                                        activation=config['activation']))
        for i, s in enumerate(config['shape'][1:-1]):
            self.layers.append(self.ff_unit(int(s[0]*hidden), int(s[1]*hidden), 
                                            dropout=config['dropout'][i+1],
                                            batch_norm=config['batch_norm'],
                                            activation=config['activation']))
        self.layers.append(self.ff_unit(int(config['shape'][-1][0]*hidden), out_channels,
                                        dropout=None,
                                        batch_norm=None,
                                        activation=None))
        
        print('FFNet model loaded...')
        

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


