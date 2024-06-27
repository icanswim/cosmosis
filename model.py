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
        
        self.softmax = None
        self.X = 'X'
        self.target = 'y'
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
        """creates the embedding layer per the embedding params specs
        
        voc = int (vocabulary size)
        vec = int (embedding dimension length)
        padding_idx = 0 (the token used for padding)
        trainable = True/False (feed gradient back to the embedding)
        flatten = True/False flatten the embedding outputs

        embed_params = {'output_key': [('feature_key',voc,vec,padding_idx,trainable,flatten)]}

        embed_params =  {'embed': [('feature_3',3,16,0,True,False),('feature_4',4,16,0,True,False)],
                         'embed2': [('feature_6',3,8,0,True,False)]}

        returns embedded = {'feature_3': nn.Embedding(3,16,0,True),
                            'feature_4': nn.Embedding(4,16,0,True),
                            'feature_6': nn.Embedding(3,8,0,True)}
        """
        self.embed_params = embed_params
        embeddings = {}
        
        for k, params in embed_params.items():                
            if type(params) == list and type(params[0]) == tuple:
                for v in params:
                    feature, voc, vec, padding_idx, trainable, flatten = v
                    embeddings[feature] = nn.Embedding(voc, vec, padding_idx).to(device)
                    embeddings[feature].weight.requires_grad = trainable
            else:
                continue
                
        return embeddings

    def embed_features(self, data):
        """
        embeds, flattens and then concatenates keys per self.embed_params
        returns embedded = {'output_key1': torch.tensor,
                            'output_key2': torch.tensor}
        """
        embedded = {}
        # [(voc,vec,padding_idx,trainable,flatten),(...)]
        for output_key, embed_p in self.embed_params.items(): 
            output = []
            if type(embed_p) == list and type(embed_p[0]) == tuple: 
                for i, p in enumerate(embed_p):
                    feature = p[0]
                    if type(data) == dict:
                        out = self.embeddings[feature](data['embedding_input'][feature])
                    elif hasattr(data, feature):
                        out = self.embeddings[feature](data.feature)
                    else:
                        out = self.embeddings[feature](data)

                    if p[5] == True: 
                        out = flatten(out)
                    output.append(out)
                
                if len(output) > 1:
                    output = cat(output, dim=0)
                else:
                    output = output[0]
                    
                embedded[output_key] = output
        return embedded
        

    def forward(self, data):
        """data['X'] = torch tensor of concatenated continuous feature vectors
        
        embed_params: {'output_key1': [('feature_3',4,16,0,True,False),('feature_4',5,16,0,True,False)],
                       'ouput_key2': [('feature_6',4,8,0,True,False)]}
        
        lookup_feature_3 = ExampleDataset.embed_lookup['feature_3']
        lookup_feature_4 = ExampleDataset.embed_lookup['feature_4']
        lookup_feature_6 = ExampleDataset.embed_lookup['feature_6']

        ds_params = {'train_params': {'input_dict': {'model_input': {'X': ['feature_1','feature_5'],
                                                                     'X2': ['feature_2']},
                                                     'embedding_imput': {'feature_3': ['feature_3'],
                                                                         'feature_4': ['feature_4'],
                                                                         'feature_6': ['feature_6']},
                                                     'criterion_input': {'target': ['feature_5']}},
                                      'transforms': {'feature_1': [ExampleTransform(10), AsTensor()],
                                                     'feature_2': [Reshape(-1), Pad1d(10), AsTensor()],
                                                     'feature_3': [Pad1d(5), EmbedLookup(lookup_feature_3),
                                                                     AsTensor()],
                                                     'feature_4': [Pad1d(5), EmbedLookup(lookup_feature_4),
                                                                     AsTensor()],
                                                     'feature_5': [AsTensor()],
                                                     'feature_6': [EmbedLookup(lookup_feature_6), AsTensor()]},
                                      'boom': 'bang'}}
        
        """
        if self.embed_params:
            embedded_dict = self.embed_features(data)

            for embed_key, embed in embedded_dict.items(): # mechanism for multiple embedding output
                embed_key, embed # single embedding output and key

            if type(data) == dict and 'model_input' in data and len(data['model_input']) != 0:
                for model_key, X in data['model_input'].items(): # mechanism for multiple model inputs
                    model_key, X # single model input and key
                    X = cat([X, embed], dim=0)      
            elif hasattr(data, self.X): 
                attr = self.X
                X = data.attr
                X = cat([X, embed])
            else:
                X = embed
        else:
            if type(data) == dict and 'model_input' in data:
                for model_key, X in data['model_input'].items():
                    X = cat([X, embed], dim=0)
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
                              'activation': nn.ReLU}
    model_config['funnel'] = {'shape': [('in_channels',1),(1,1),(1,1),
                                        (1,1/2),(1/2,1/2),(1/2,'out_channels')], 
                              'dropout': [.1, .2, .3, .1, .2],
                              'activation': nn.ReLU}

    def build(self, model_name='funnel', in_channels=0, hidden=0, out_channels=0, 
                      softmax=None, **kwargs):
        
        config = FFNet.model_config[model_name]
        self.layers = []
        
        self.layers.append(self.ff_unit(in_channels, int(config['shape'][0][1]*hidden),                                                                   dropout=config['dropout'][0],
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
        return x.view(x.size(0), -1)


class GPT(CModel):

    def build(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=128, **kwargs):
        self.layers = []
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.layers.append(nn.TransformerDecoder(decoder_layer, num_layers))

    def forward(self, data):
        if self.embed_params:
            embedded_dict = self.embed_features(data)

            for embed_key, embed in embedded_dict.items(): # mechanism for multiple embedding output
                embed_key, embed # single embedding output and key

            if type(data) == dict and 'model_input' in data and len(data['model_input']) != 0:
                for model_key, X in data['model_input'].items(): # mechanism for multiple model inputs
                    model_key, X # single model input and key
                    X = cat([X, embed], dim=0)      
            elif hasattr(data, self.X): 
                attr = self.X
                X = data.attr
                X = cat([X, embed])
            else:
                X = embed
        else:
            if type(data) == dict and 'model_input' in data:
                for model_key, X in data['model_input'].items():
                    X = cat([X, embed], dim=0)
            elif hasattr(data, self.X): 
                attr = self.X
                X = data.attr
            else:
                X = data
           
        for l in self.layers:
            X = l(X, X)
            
        if self.softmax is not None:
            X = getattr(F, self.softmax)(X, dim=1)

        return X


class IdentityModel(CModel):

    def build(self, *args, **kwargs):
        self.layers = []
        self.layers.append(nn.Identity())
        

