from math import sqrt

from torch import nn, cat, squeeze, Tensor, flatten, sigmoid, arange, topk
from torch import max, mean, multinomial, transpose, tril, ones, long, no_grad
from torch.nn import functional as F

# torchvision models are imported by its launcher tv_models()
  
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

    init_weights = True/False
    
        
    datadict keywords:
      
    """
    def __init__(self, model_param):
        super().__init__()

        self.device = 'cuda:0'
        if 'device' in model_param:  
            self.device = model_param['device']

        self.data_keys = None
        if 'data_keys' in model_param:
            self.data_keys = model_param['data_keys']

        self.y = 'y'
        if 'y' in model_param:
            self.y = model_param['y']

        self.embed_param = None
        if 'embed_param' in model_param:
            self.embed_param = model_param['embed_param']
            self.embedding_layer = self.create_embedding_layer()
            if 'flatten' not in self.embed_param:
                self.embed_param['flatten'] = False
            
        self.build(**model_param)    
        if hasattr(self, 'layers'):
            self.layers = nn.ModuleList(self.layers) 

        self.init_weights()
        print('{} model loaded...'.format(self.__class__.__name__))
                            
    def build(self, **kwargs):
        self.layers = []
        raise NotImplementedError('subclass and implement build()...')

    def init_weights(self):
        """
        define _init_weights for custom weight initializations

        def _init_weights(self, module):
            if isinstance(module, torch.nn.Layer):
                torch.nn.init.func_(module.weight, **kwargs)
                if module.bias is not None:
                    torch.nn.init.func_(module.bias)
        """
        
        if hasattr(self, '_init_weights'):
            print('applying _init_weights...')
            self.apply(self._init_weights)
        else:
            print('default weight initialization...')
     
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params  
        
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

    def embed_features(self, data):
        """
        passes the tokens to the embedding layer

        categorical features or tokens can be passed to the embedding layer in any of 3 ways:
            data['feature'] = torch.int64 
            data.feature = torch.int64
            torch.int64
        
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
        """
        data can have the form:
            data['feature'] = array
            data.feature = array
            array

        array can be type numpy or torch
        """
        X = []
        filter_keys = [] # keys not to be included
        if self.y is not None: filter_keys.append(self.y)

        # if any features are to be embedded, embed them, add keys to filter
        if self.embed_param is not None:
            embedded = []
            embedded_dict = self.embed_features(data)
            for e, embed in embedded_dict.items():
                if self.embed_param['flatten']:
                    embed = flatten(embed, start_dim=0)
                embedded.append(embed)
                filter_keys.append(e)
            embedded = cat(embedded, dim=-1) 
        # data can be passed as a dict 
        if type(data) == dict:
            for k in data.keys(): 
                if k not in filter_keys:
                    X.append(data[k])
            if len(X) != 0: X = cat(X, dim=-1) 
        # or as a data object
        elif self.data_keys is not None and all(hasattr(data, dk) for dk in self.data_keys): 
            for k in self.data_keys:
                if k not in filter_keys:
                    X.append(data.k)
            X = cat(X, dim=-1)
        # or as an array
        else:
            X = data
        # cat any features with any embedded features    
        if self.embed_param is not None:
            if len(X) == 0:
                X = embedded
            else:
                X = cat([X, embedded], dim=-1)
        # pass the prepared features to the model 
        for l in self.layers:
            X = l(X)
            
        return X
    
    def adapt(self, in_channels, out_channels, dropout):
        """prepends a trainable feedforward layer"""
        for l in self.ff_unit(in_channels, out_channels, activation=None, dropout=dropout)[::-1]:
            self.layers.insert(0, l)
            
    def ff_unit(self, in_channels, out_channels, 
                    activation=nn.ReLU, norm=True, dropout=.2):
        ffu = []
        ffu.append(nn.Linear(in_channels, out_channels))
        if norm: ffu.append(nn.BatchNorm1d(out_channels))
        if activation is not None: ffu.append(activation())
        if dropout is not None:  ffu.append(nn.Dropout(dropout))
            
        return nn.Sequential(*ffu)
    
    def conv_unit(self, in_channels, out_channels, kernel_size=3, 
                  stride=1, padding=1, dilation=1, groups=1, bias=False, 
                  activation=nn.ReLU, dropout=None, pool=(5,2,1)):
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

    def create_mask(self, size):
        mask = tril(ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # convert ones to 0
        return mask
    

def tv_model(model_param):
    """A torchvision model launcher"""
    from torchvision import models as torchvisionmodels
    
    launcher = getattr(torchvisionmodels, model_param['model_name'])
    model = launcher(**model_param['tv_param'])

    def forward(data): # patch
        return model._forward_impl(data['image'])
    
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
                              'activation': nn.GELU}
    model_config['funnel'] = {'shape': [('in_channels',1),(1,1),(1,1),
                                        (1,1/2),(1/2,1/2),(1/2,'out_channels')], 
                              'dropout': [.1, .2, .3, .1, .2],
                              'activation': nn.GELU}

    def build(self, model_name='funnel', in_channels=0, hidden=0, out_channels=0, **kwargs):
        
        config = FFNet.model_config[model_name]
        self.layers = []
        
        self.layers.append(self.ff_unit(in_channels, int(config['shape'][0][1]*hidden),
                                        dropout=config['dropout'][0],
                                        norm=True,
                                        activation=config['activation']))
        
        for i, s in enumerate(config['shape'][1:-1]):
            self.layers.append(self.ff_unit(int(s[0]*hidden), int(s[1]*hidden), 
                                            dropout=config['dropout'][i+1],
                                            norm=True,
                                            activation=config['activation']))
            
        self.layers.append(self.ff_unit(int(config['shape'][-1][0]*hidden), out_channels,
                                        dropout=None,
                                        norm=False,
                                        activation=None))
        
        print('FFNet model loaded...')

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)


class IdentityModel(CModel):
    def build(self, *args, **kwargs):
        self.layers = []
        self.layers.append(nn.Identity())


class Attention(CModel):
    """
    d_vec = dimension embedding vector
    n_head = number of attention heads
    """

    def build(self, d_vec=0, n_head=0, **model_param):

        self.d_vec = d_vec
        self.n_head = n_head
        assert d_vec % n_head == 0
        
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(d_vec, 3 * d_vec, bias=False)
        self.proj = nn.Linear(d_vec, d_vec, bias=False)

        self.attn_dropout = nn.Dropout(p=.1)
        self.proj_dropout = nn.Dropout(p=.1)

    def forward(self, x):
        batch, d_seq, d_vec = x.size() # d_seq = dimesion sequence (time, sentence length)
        assert d_vec == self.d_vec
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.attn(x).split(self.d_vec, dim=2)
        k = k.view(batch, d_seq, self.n_head, d_vec // self.n_head).transpose(1, 2) # (batch, n_head, d_seq, hs)
        q = q.view(batch, d_seq, self.n_head, d_vec // self.n_head).transpose(1, 2) # (batch, n_head, d_seq, hs)
        v = v.view(batch, d_seq, self.n_head, d_vec // self.n_head).transpose(1, 2) # (batch, n_head, d_seq, hs)
        # self-attend: (batch, n_head, d_seq, hs) x (batch, n_head, hs, d_seq) -> (B, nh, T, T)
        y = F.scaled_dot_product_attention(q, k, v, 
                attn_mask=None, dropout_p=.1 if self.training else 0, is_causal=True)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(batch, d_seq, d_vec) 
        y = self.proj_dropout(self.proj(y))
        
        return y


class Block(CModel):

    def build(self, d_vec=0, **model_param):
        self.ln_1 = nn.LayerNorm(d_vec, bias=False)
        self.attn = Attention(**model_param)
        self.ln_2 = nn.LayerNorm(d_vec, bias=False)
        self.ffnet = FFNet(model_name='funnel', in_channels=d_vec, hidden=2*d_vec, 
                                                   out_channels=d_vec, **model_param)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffnet(self.ln_2(x))
        return x


class GPT(CModel):
    """https://github.com/karpathy/nanoGPT"""

    def build(self, n_layer=0, d_vec=0, d_vocab=0, d_seq=0, temperature=1, **model_param):
        self.d_seq = d_seq
        self.temperature = temperature
        self.dropout = nn.Dropout(p=.1)
        self.layers = [Block(**model_param) for _ in range(n_layer)]
        self.layer_norm = nn.LayerNorm(d_vec, bias=False)
        self.lm_head = nn.Linear(d_vec, d_vocab, bias=False)
        # weight tying
        self.embedding_layer['tokens'].weight = self.lm_head.weight 
  
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=.02)

    def forward(self, data):
        embedded_dict = self.embed_features(data)
        x = self.dropout(embedded_dict['tokens'] + embedded_dict['position'])
        for block in self.layers:
            x = block(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        return logits

    def generate(self, logits):
        """
        (n_batch, d_seq, d_vocab)
        """
        logits = logits[:,-1:,:]
        for _ in range(self.d_seq):
            last = logits[:,-1:,:]
            probs = F.softmax(last, dim=-1)
            tokens = probs.argmax(dim=-1)
            data = {'tokens': tokens,
                    'position': arange(0, tokens.shape[1], dtype=long).to('cuda:0')}
            next = self(data)
            next = next / self.temperature
            logits = cat((logits, next), dim=1)

        return logits


