from datetime import datetime
import logging
import random
import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch import no_grad, save, load, from_numpy, squeeze
from torch.utils.data import Sampler, DataLoader
from torch.nn.functional import softmax

from sklearn import metrics


class Metrics():
    
    def __init__(self, report_interval=10, sk_metric_name=None, sk_params={}):
        
        self.start = datetime.now()
        self.report_time = self.start
        self.report_interval = report_interval
        
        self.epoch, self.e_loss, self.predictions = 0, [], []
        self.train_loss, self.val_loss, self.lr_log = [], [], []
        
        self.sk_metric_name, self.sk_params = sk_metric_name, sk_params
        self.skm, self.sk_y, self.sk_pred, self.sk_log = None, [], [], []
        if self.sk_metric_name is not None:
            self.skm = getattr(metrics, self.sk_metric_name)
            
        logging.basicConfig(filename='./logs/cosmosis.log', level=20)
        self.log('\nNew Experiment: {}'.format(self.start))
    
    def __call__(self, epoch):
        self.epoch = epoch
    
    def infer(self):
        self.predictions = np.concatenate(self.predictions, axis=0)
        self.predictions = np.reshape(self.predictions, (-1, 2))
        self.predictions = pd.DataFrame(self.predictions, columns=['id','predictions'])
        self.predictions['id'] = self.predictions['id'].astype('int64')
        print('self.predictions.iloc[:10]', self.predictions.shape, self.predictions.iloc[:10])
        self.predictions.to_csv('cosmosis_inference.csv', 
                                header=['id','predictions'], 
                                index=False)
        print('inference complete and saved to csv...')

    def sk_metric(self):
        if self.skm is not None:
            def softmax(x):
                return np.exp(x)/sum(np.exp(x))
            y = np.reshape(np.vstack(np.asarray(self.sk_y)), -1)
            y_pred = np.vstack(np.asarray(self.sk_pred))
            if self.sk_metric_name == 'roc_auc_score':
                y_pred = np.apply_along_axis(softmax, 1, y_pred)
            self.sk_log.append(self.skm(y, y_pred, **self.sk_params))
            self.sk_y, self.sk_pred = [], []
        else: 
            self.sk_log.append(0)
        
    def loss(self, flag, loss):
        if flag == 'train':
            self.train_loss.append(loss)
        if flag == 'val':
            self.val_loss.append(loss)
        if flag == 'test':
            self.log('test loss: {}'.format(loss))
            print('test loss: {}'.format(loss))
          
    def log(self, message):
        logging.info(message)
        
    def status_report(self):
        elapsed = datetime.now() - self.report_time
        if elapsed.total_seconds() > self.report_interval or self.epoch % 10 == 0:
            print('learning time: {}'.format(datetime.now()-self.start))
            print('epoch: {}, lr: {}'.format(self.epoch, self.lr_log[-1]))
            print('train loss: {}, val loss: {}'.format(self.train_loss[-1], self.val_loss[-1]))
            print('sk_metric: \n{}'.format(self.sk_log[-1]))
            self.report_time = datetime.now()
        
    def report(self):
        elapsed = datetime.now() - self.start            
        self.log('learning time: {} \n'.format(elapsed))
        print('learning time: {}'.format(elapsed))
        self.log('sklearn metric: \n{} \n'.format(self.sk_log[-1]))
        print('sklean metric: \n{} \n'.format(self.sk_log[-1]))
        pd.DataFrame(zip(self.train_loss, self.val_loss, self.lr_log, self.sk_log),
                     columns=['train_loss','val_loss','lr','sk_metric']).to_csv(
                                            './logs/'+self.start.strftime("%Y%m%d_%H%M"))
        self.view_log('./logs/'+self.start.strftime('%Y%m%d_%H%M'))
        
    @classmethod    
    def view_log(cls, log_file):
        log = pd.read_csv(log_file)
        log.iloc[:,1:5].plot(logy=True)
        plt.show() 


class Selector(Sampler):
    """splits = (.8,) or (.7,.15) or None
    
    single ds use splits with 2 values to make train/val/test sets
    double ds use splits with 1 value to make train/val set and second ds for test
    triple ds use splits None 
    """
   
    def __init__(self, dataset_idx=None, train_idx=None, val_idx=None, test_idx=None,
                 splits=(.7,.15), set_seed=False):
        self.set_seed = set_seed
        
        if dataset_idx == None:  
            self.dataset_idx = train_idx
        else:
            self.dataset_idx = dataset_idx
            
        self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx
        
        if set_seed: 
            random.seed(set_seed)
            
        random.shuffle(self.dataset_idx)                
        if len(splits) == 1:  
            cut1 = int(len(self.dataset_idx)*splits[0])
            self.train_idx = self.dataset_idx[:cut1]
            self.val_idx = self.dataset_idx[cut1:]
        if len(splits) == 2:
            cut1 = int(len(self.dataset_idx)*splits[0])
            cut2 = int(len(self.dataset_idx)*splits[1])
            self.train_idx = self.dataset_idx[:cut1]
            self.val_idx = self.dataset_idx[cut1:cut1+cut2]
            self.test_idx = self.dataset_idx[cut1+cut2:]
        
        random.seed()
        
    def __iter__(self):
        if self.flag == 'train':
            return iter(self.train_idx)
        if self.flag == 'val':
            return iter(self.val_idx)
        if self.flag == 'test':
            return iter(self.test_idx)
        if self.flag == 'infer':
            return iter(self.dataset_idx)

    def __len__(self):
        if self.flag == 'train':
            return len(self.train_idx)
        if self.flag == 'val':
            return len(self.val_idx)
        if self.flag == 'test':
            return len(self.test_idx) 
        if self.flag == 'infer':
            return len(self.dataset_idx)
        
    def __call__(self, flag):
        self.flag = flag
        return self
    
    def shuffle_train_val_idx(self):
        if self.set_seed:
            random.seed(self.set_seed)
        random.shuffle(self.val_idx)
        random.shuffle(self.train_idx)
        random.seed()
        
        
class Learn():
    """
    Datasets = [TrainDS, ValDS, TestDS]
        if 1 DS is given it is split into train/val/test using splits param
        if 2 DS are given first one is train/val second is test
        if 3 DS are given first is train second is val third is test
        
    Criterion = None implies inference mode
    
    TODO: accuracy
          early stopping/checkpoints
          inference abstraction
          
    load_model = False/'saved_model.pth'/'saved_model.pk'
    """
    def __init__(self, Datasets, Model, Sampler=Selector, Metrics=Metrics,
                 Optimizer=None, Scheduler=None, Criterion=None, 
                 ds_params={}, model_params={}, sample_params={},
                 opt_params={}, sched_params={}, crit_params={}, metrics_params={}, 
                 adapt=False, load_model=False, load_embed=False, save_model=False,
                 batch_size=10, epochs=1):
        
        self.bs = batch_size
        self.ds_params = ds_params
        self.dataset_manager(Datasets, Sampler, ds_params, sample_params)
        
        self.metrics = Metrics(**metrics_params)
        self.metrics.log('model: {}\n{}'.format(Model, model_params))
        self.metrics.log('dataset: {}\n{}'.format(Datasets, ds_params))
        self.metrics.log('sampler: {}\n{}'.format(Sampler, sample_params))
        self.metrics.log('epochs: {}, batch_size: {}, save_model: {}, load_model: {}'.format(
                                                    epochs, batch_size, save_model, load_model))
        
        if load_model:
            try: #uses the same embed params for all datasets (train/val/test)
                model = Model(embed=self.ds_params['train_params']['embed'], 
                                                              **model_params)
                model.load_state_dict(load('./models/'+load_model))
                print('model loaded from state_dict...')
            except:
                model = load('./models/'+load_model)
                print('model loaded from pickle...')                                                      
        else:
            model = Model(embed=self.ds_params['train_params']['embed'], 
                                                          **model_params)
        
        if load_embed:
            for i, embedding in enumerate(model.embeddings):
                try:
                    weight = np.load('./models/{}_{}_embedding_weight.npy'.format(
                                                                load_embed, i))
                    embedding.from_pretrained(from_numpy(weight), 
                                              freeze=self.ds_params['train_ds']['embed'][i][4])
                    print('loading embedding weights...')
                except:
                    print('no embedding weights found.  initializing... ')
                    
        if adapt: 
            model.adapt(adapt)
        self.model = model.to('cuda:0')
        self.metrics.log(self.model.children)
        
        if Criterion:
            self.criterion = Criterion(**crit_params).to('cuda:0')
            self.metrics.log('criterion: {}\n{}'.format(self.criterion, crit_params))
            self.opt = Optimizer(self.model.parameters(), **opt_params)
            self.metrics.log('optimizer: {}\n{}'.format(self.opt, opt_params))
            self.scheduler = Scheduler(self.opt, **sched_params)
            self.metrics.log('scheduler: {}\n{}'.format(self.scheduler, sched_params))
            
            for e in range(epochs):
                self.metrics(e)
                self.sampler.shuffle_train_val_idx()
                self.run('train')
                with no_grad():
                    self.run('val')  
                
            with no_grad():
                self.run('test')
                
        else: # no Criterion implies inference mode
            with no_grad():
                self.run('infer')
        
        if save_model:
            if adapt: save(self.model, './models/{}.pth'.format(
                                                            start.strftime("%Y%m%d_%H%M")))
            if not adapt: save(self.model.state_dict(), './models/{}.pth'.format(
                                                            start.strftime("%Y%m%d_%H%M")))
            if hasattr(self.model, 'embeddings'):
                for i, embedding in enumerate(self.model.embeddings):
                    weight = embedding.weight.detach().cpu().numpy()
                    np.save('./models/{}_{}_embedding_weight.npy'.format(
                                             start.strftime("%Y%m%d_%H%M"), i), weight)
        self.metrics.report()
        
    def run(self, flag): 
        e_loss, e_sk, i = 0, 0, 0
        
        if flag == 'train': 
            self.model.training = True
            dataset = self.train_ds
            sampler = self.sampler(flag=flag)
            drop_last = True
            
        if flag == 'val':
            self.model.training = False
            dataset = self.val_ds
            sampler = self.sampler(flag=flag)
            drop_last = True

        if flag == 'test':
            self.model.training = False
            dataset = self.test_ds
            sampler = self.sampler(flag=flag)
            drop_last = False
            
        if flag == 'infer':
            self.model.training = False
            dataset = self.test_ds
            sampler = self.sampler(flag=flag)
            drop_last = False
            
        dataloader = DataLoader(dataset, batch_size=self.bs, 
                                sampler=self.sampler(flag=flag), 
                                num_workers=8, pin_memory=True, 
                                            drop_last=drop_last)

        def to_cuda(data):
            if len(data) == 0: return None
            else: return data.to('cuda:0', non_blocking=True)

        for  X, y, embed in dataloader:
            i += self.bs
            X = to_cuda(X)
            if len(embed) > 0:
                embed = [to_cuda(emb) for emb in embed]
                y_pred = self.model(X, embed)
            else:
                y_pred = self.model(X)
                
            if flag == 'infer':
                self.metrics.predictions.append(np.concatenate((y_pred, y), axis=1))
            else:
                y = to_cuda(y)
                self.opt.zero_grad()
                b_loss = self.criterion(y_pred, y)
                e_loss += b_loss.item()
                if self.metrics.skm is not None:
                    self.metrics.sk_y.append(y.detach().cpu().numpy())
                    self.metrics.sk_pred.append(y_pred.detach().cpu().numpy())
                if flag == 'train':
                    b_loss.backward()
                    self.opt.step()
                    
        if flag == 'infer':
            self.metrics.infer()
        else:
            self.metrics.loss(flag, e_loss/i)
            self.metrics.sk_metric()
            
        if flag == 'val': 
            self.scheduler.step(e_loss/i)
            self.metrics.lr_log.append(self.opt.param_groups[0]['lr'])
            self.metrics.status_report()
        
            
    def dataset_manager(self, Datasets, Sampler, ds_params, sample_params):
    
        if len(Datasets) == 1:
            self.train_ds = Datasets[0](**ds_params['train_params'])
            self.val_ds = self.test_ds = self.train_ds
            self.sampler = Sampler(dataset_idx=self.train_ds.ds_idx, 
                                   **sample_params)
        if len(Datasets) == 2:
            self.train_ds = Datasets[0](**ds_params['train_params'])
            self.val_ds = self.train_ds
            self.test_ds = Datasets[1](**ds_params['test_params'])
            self.sampler = Sampler(train_idx=self.train_ds.ds_idx, 
                                   test_idx=self.test_ds.ds_idx,
                                   **sample_params)
        if len(Datasets) == 3:
            self.train_ds = Datasets[0](**ds_params['train_params'])
            self.val_ds = Datasets[1](**ds_params['val_params'])
            self.test_ds = Datasets[2](**ds_params['test_params'])
            self.sampler = Sampler(train_idx=self.train_ds.ds_idx, 
                                   val_idx=self.val_ds.ds_idx, 
                                   test_idx=self.test_ds.ds_idx,
                                   **sample_params)


        
        
