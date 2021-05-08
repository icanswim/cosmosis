from abc import ABC, abstractmethod

from math import sqrt

from torch import nn, cat, squeeze, softmax, Tensor, flatten, sigmoid, max, mean
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

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, 
                             relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, 
                                                          groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FFUnit(nn.Module):
    def __init__(self, D_in, D_out, dropout=False, activation=nn.SELU):
        ffu = []
        ffu.append(nn.Linear(D_in, D_out))
        ffu.append(activation())
        ffu.append(nn.BatchNorm1d(D_out))
        if bam: ffu.append(BAM(D_out)) 
        if dropout: ffu.append(nn.Dropout(drop))
        
        self.layers = nn.Sequential(*ffu)
        
    def forward(self, x):
        return self.layers(x)
    
    
class ConvUnit(nn.Module):
    def __init__(self,  in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1, dilation=1, groups=1, bias=False, 
                 activation=None, cbam=False):
        conv = []
        conv.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, bias=bias))
        conv.append(nn.BatchNorm2d(out_channels))
        if activation: conv.append(activation())
        if pool: 
            conv.append(nn.MaxPool2d(2, stride=2, padding=1))
            out_channels = out_channels/2
        conv.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, bias=bias))
        if cbam: conv.append(CBAM(out_channels))
        if dropout: conv.append(nn.Dropout(p=dropout))
                      
        self.layers = nn.Sequential(*conv)
                 
    def forward(self, x):
        return self.layers(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    
class ChannelGateB(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super().__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module('gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]))
            self.gate_c.add_module('gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]))
            self.gate_c.add_module('gate_c_relu_%d'%(i+1), nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))
        
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d(in_tensor, in_tensor.size(2), stride=in_tensor.size(2))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)
    
class ChannelGateC(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super().__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
    

class ChannelPool(nn.Module):
    def forward(self, x):
        return cat((max(x,1)[0].unsqueeze(1), mean(x,1).unsqueeze(1)), dim=1)
    
    
class SpatialGateC(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = sigmoid(x_out) # broadcasting
        return x * scale
    
    
class SpatialGateB(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super().__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0', nn.Conv2d(
                               gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(
                                                            gate_channel//reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0',nn.ReLU())
        for i in range( dilation_conv_num ):
            self.gate_s.add_module('gate_s_conv_di_%d'%i, nn.Conv2d(
                    gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3,\
                                                    padding=dilation_val, dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d'%i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 
                                                                            1, kernel_size=1))
    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)

    
class BAM(nn.Module):
    def __init__(self, gate_channel):
        super().__init__()
        self.channel_att = ChannelGateB(gate_channel)
        self.spatial_att = SpatialGateB(gate_channel)
        
    def forward(self,in_tensor):
        att = 1 + sigmoid(self.channel_att(in_tensor) * self.spatial_att(in_tensor))
        return att * in_tensor   
    
    
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super().__init__()
        self.ChannelGate = ChannelGateC(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGateC()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

    
class CModel(nn.Module):
    """A base class for cosmosis models
    embed = [('feature',n_vocab,len_vec,padding_idx,param.requires_grad),...]
    """
    def __init__(self, embed=[], **kwargs):
        super().__init__()
        print('CModel loaded...')
        #self.embeddings = self.embedding_layer(embed)
        #self.layers = nn.ModuleList()
        #self.weight_init()
    
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
            
    def ff_unit(self, D_in, D_out, dropout=False, activation=nn.SELU):
        ffu = []
        ffu.append(nn.Linear(D_in, D_out))
        if activation: ffu.append(activation())
        ffu.append(nn.BatchNorm1d(D_out))
        if dropout:  ffu.append(nn.Dropout(dropout))
        
        return nn.Sequential(*ffu)
    
    def conv_unit(self, in_channels, out_channels, kernel_size=3, 
                  stride=1, padding=1, dilation=1, groups=1, bias=False, 
                  activation=None, cbam=False, dropout=False, pool=False):
        conv = []
        conv.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias))
        conv.append(nn.BatchNorm2d(out_channels))
        if activation: conv.append(activation())
        if pool: conv.append(nn.MaxPool2d(kernel_size=5, stride=2, padding=1))
        conv.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, bias=bias))
        if cbam: conv.append(CBAM(out_channels))
        if dropout: conv.append(nn.Dropout(p=dropout))
                        
        return nn.Sequential(*conv)
    
    def res_connect(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1):
        res = []
        res.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                                    stride=stride, dilation=dilation))
        res.append(nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*res)

class ResBam(CModel):
    """ConvNet with options for residual connections and attention units and NeXt groupings

    ResNet https://arxiv.org/abs/1512.03385
    ResNeXt https://arxiv.org/abs/1611.05431
    CBAM https://arxiv.org/abs/1807.06521v2
    """
    def __init__(self, n_classes, in_channels, groups=1, residual=False, bam=False, 
                 dropout=[False,False,False,False,False], embed=[], act=nn.LeakyReLU):
        super().__init__()
        self.residual = residual
        self.bam = bam
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, stride=2, 
                                           padding=5, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=2, padding=1)
        self.activation = nn.SELU()
        
        self.unit1 = self.conv_unit(64, 128, kernel_size=3, stride=1, groups=groups, 
                                    activation=act, cbam=bam, dropout=dropout[0])
        if residual: self.res1 = self.res_connect(64, 128, kernel_size=1, stride=1, dilation=1)
        if bam: self.bam1 = BAM(128)
        
        self.unit2 = self.conv_unit(128, 256, kernel_size=3, stride=2, groups=groups,
                                    activation=act, cbam=bam, dropout=dropout[1])
        if residual: self.res2 = self.res_connect(128, 256, kernel_size=1, stride=4, dilation=1)
        if bam: self.bam2 = BAM(256)
        
        self.unit3 = self.conv_unit(256, 512, kernel_size=3, stride=2, groups=groups,
                                    activation=act, cbam=bam, dropout=dropout[2])
        if residual: self.res3 = self.res_connect(256, 512, kernel_size=1, stride=4, dilation=1)
        if bam: self.bam3 = BAM(512)
        
        self.unit4 = self.conv_unit(512, 1024, kernel_size=3, stride=2, groups=groups,
                                    activation=None, cbam=False, dropout=dropout[3])
        if residual: self.res4 = self.res_connect(512, 1024, kernel_size=1, stride=4, dilation=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = self.ff_unit(1024, n_classes, dropout=dropout[4], activation=None)
        
        self.weight_init()

        print('ResBam model loaded...')
        
    def forward(self, X):
        
        X = self.conv1(X)
        X = self.bn(X)
        X = self.activation(X)
        X = self.maxpool(X)
        
        if self.residual: res = self.res1(X) 
        X = self.unit1(X)
        if self.residual: X += res
        X = self.activation(X)
        if self.bam: X = self.bam1(X)
            
        if self.residual: res = self.res2(X)    
        X = self.unit2(X)
        if self.residual: X += res
        X = self.activation(X)
        if self.bam: X = self.bam2(X)
        
        if self.residual: res = self.res3(X)
        X = self.unit3(X)
        if self.residual: X += res
        X = self.activation(X)
        if self.bam: X = self.bam3(X)
            
        if self.residual: res = self.res4(X)
        X = self.unit4(X)
        if self.residual: X += res
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
        layers.append(self.ff_unit(D_in, int(config['shape'][0][1]*H), dropout=config['dropout'][0]))
        for i, s in enumerate(config['shape'][1:-1]):
            layers.append(self.ff_unit(int(s[0]*H), int(s[1]*H), dropout=config['dropout'][i]))
        layers.append([nn.Linear(int(config['shape'][-1][0]*H), D_out)])
        self.layers = [l for ffu in layers for l in ffu] # flatten
        self.layers = nn.ModuleList(self.layers)  
        self.embeddings = self.embedding_layer(embed)
        self.weight_init()

        print('FFNet model loaded...')
        
        
