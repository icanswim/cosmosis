from math import sqrt
from dataclasses import dataclass

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
            
        print('CModel loaded...')
                            
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
        """data type can be a 
            dict: {'key': array}, 
            object: data.key, 
            array: numpy or torch
        """
        X = []
        filter_keys = [] #keys not to be included
        if self.y is not None: filter_keys.append(self.y)

        #if any features are to be embedded, embed them, add keys to filter
        if self.embed_param is not None:
            embedded = []
            embedded_dict = self.embed_features(data)
            for e, embed in embedded_dict.items():
                if self.embed_param['flatten']:
                    embed = flatten(embed, start_dim=0)
                embedded.append(embed)
                filter_keys.append(e)
            embedded = cat(embedded, dim=-1) 
        #data can be passed as a dict 
        if type(data) == dict:
            for k in data.keys(): 
                if k not in filter_keys:
                    X.append(data[k])
            if len(X) != 0: X = cat(X, dim=-1) 
        #or as a data object
        elif self.data_keys is not None and all(hasattr(data, dk) for dk in self.data_keys): 
            for k in self.data_keys:
                if k not in filter_keys:
                    X.append(data.k)
            X = cat(X, dim=-1)
        #or as an array
        else:
            X = data
        #cat any features with any embedded features    
        if self.embed_param is not None:
            if len(X) == 0:
                X = embedded
            else:
                X = cat([X, embedded], dim=-1)
        #pass the prepared features to the model 
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
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
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


class Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                dropout_p=self.dropout if self.training else 0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 10
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 16
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):
    """https://github.com/karpathy/nanoGPT"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight 
        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = arange(0, t, dtype=long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = cat((idx, idx_next), dim=1)

        return idx


