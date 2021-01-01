from abc import ABC, abstractmethod
from torch import nn, cat, squeeze, softmax, Tensor, flatten
from torch.nn import functional as F
from math import sqrt


class QModel(nn.Module):
    """A base class for Fastchem models
    embed = [(n_vocab, len_vec, param.requires_grad),...]
        The QDataset reports any categorical values it has to encode and whether 
        or not to train the embedding or fix it as a onehot
        and then serves up the values to be encoded as the x_cat component
        of the __getitem__ method.
    
    self.embeddings = embedding_layer() method checks the QDatasets embed 
    requirements and creates a list of embedding layers as appropriate"""
    def __init__(self, embed=[]):
        super().__init__()
        #self.embeddings = self.embedding_layer(embed)
        #self.layers = nn.ModuleList()
        
    def embedding_layer(self, embed):
        if len(embed) == 0:
            return None
        else:
            embeddings = [nn.Embedding(voc, vec, padding_idx=None).to('cuda:0') for voc, vec, _ in embed]
            for i, e in enumerate(embed):
                param = embeddings[i].weight
                param.requires_grad = e[2]
            return embeddings

    def forward(self, x_con, x_cat):
        """check for categorical and/or continuous inputs, get the embeddings and  
        concat as appropriate, feed to model.  
        x_cat = list of torch cuda tensors which are the embedding indices
        x_con = torch cuda tensor of concatenated continous feature vectors"""
        if len(x_cat) != 0:
            emb = []
            for i in range(len(x_cat)):
                out = self.embeddings[i](x_cat[i])
                emb.append(flatten(out, start_dim=1))
            emb = cat(emb, dim=1)
            if x_con.shape[1] != 0:
                x = cat([x_con, emb], dim=1)
            else:  
                x = emb    
        else:
            x = x_con
        
        for l in self.layers:
            x = l(x)
        return x
        
    def adapt(self, shape):
        """for adapting a dataset shape[0] to a saved model shape[1]"""
        # freeze the layers
        for param in self.parameters(): 
            param.requires_grad = False
        # prepend a trainable adaptor layer    
        for l in self.ffunit(shape[0], shape[1], 0.2)[::-1]:
            self.layers.insert(0, l)
            
    def ffunit(self, D_in, D_out, drop):
        ffu = []
        ffu.append(nn.BatchNorm1d(D_in))
        ffu.append(nn.Linear(D_in, D_out))
        ffu.append(nn.SELU())
        ffu.append(nn.Dropout(drop))
        return ffu
    
class FFNet(QModel):
    
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