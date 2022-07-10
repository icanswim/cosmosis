from datetime import datetime
import logging
import random
import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch import no_grad, save, load, from_numpy, as_tensor, squeeze
from torch.utils.data import Sampler, DataLoader
from torch.nn.functional import softmax

from sklearn import metrics


class Metrics():
    #TODO checkpointing and early stopping
    def __init__(self, report_interval=10, sk_metric_name=None, sk_params={}):
        
        self.start = datetime.now()
        self.report_time = self.start
        self.report_interval = report_interval
        
        self.epoch, self.e_loss, self.predictions = 0, [], []
        self.train_loss, self.val_loss, self.lr_log = [], [], []
        
        self.sk_metric_name, self.sk_params = sk_metric_name, sk_params
        self.skm, self.sk_train_log, self.sk_val_log = None, [], []
        self.sk_y, self.sk_pred = [], []
        if self.sk_metric_name is not None:
            self.skm = getattr(metrics, self.sk_metric_name)
            
        logging.basicConfig(filename='./logs/cosmosis.log', level=20)
        self.log('\nNew Experiment: {}'.format(self.start))
    
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

    def sk_metric(self, flag):
        if self.skm is not None:
            def softmax(x):
                return np.exp(x)/sum(np.exp(x))
            y = np.reshape(np.vstack(np.asarray(self.sk_y, 'float64')), -1)
            y_pred = np.vstack(np.asarray(self.sk_pred, 'float64'))
            
            if self.sk_metric_name == 'roc_auc_score':
                y_pred = np.apply_along_axis(softmax, 1, y_pred)
                
            score = self.skm(y, y_pred, **self.sk_params)
            
            if flag == 'train':
                self.sk_train_log.append(score)
            else:
                self.sk_val_log.append(score)

            self.sk_y, self.sk_pred = [], []
        else: 
            self.sk_train_log.append(0)
            self.sk_val_log.append(0)
        
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
            print('sklearn train metric: {}, sklearn validation metric: {}'.format(
                                                    self.sk_train_log[-1], self.sk_val_log[-1]))
            self.report_time = datetime.now()
        
    def report(self):
        elapsed = datetime.now() - self.start            
        self.log('learning time: {} \n'.format(elapsed))
        print('learning time: {}'.format(elapsed))
        self.log('sklearn test metric: \n{} \n'.format(self.sk_val_log[-1]))
        print('sklearn test metric: \n{} \n'.format(self.sk_val_log[-1]))
        pd.DataFrame(zip(self.train_loss, self.val_loss, self.lr_log, self.sk_val_log),
                     columns=['train_loss','val_loss','lr','sk_metric']).to_csv(
                                            './logs/'+self.start.strftime("%Y%m%d_%H%M"))
        self.view_log('./logs/'+self.start.strftime('%Y%m%d_%H%M'))
        
    @classmethod    
    def view_log(cls, log_file):
        log = pd.read_csv(log_file)
        log.iloc[:,1:5].plot(logy=True)
        plt.show() 


class Selector(Sampler):
    """splits = (train_split,) remainder is val_split or 
                (train_split,val_split) remainder is test_split or 
                None
    """
    def __init__(self, dataset_idx=None, train_idx=None, val_idx=None, test_idx=None,
                 splits=(.7,.15), set_seed=False, subset=False):
        self.set_seed = set_seed
        
        if dataset_idx == None:  
            self.dataset_idx = train_idx
        else:
            self.dataset_idx = dataset_idx
            
        self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx
        
        if set_seed: 
            random.seed(set_seed)
            
        random.shuffle(self.dataset_idx) 
        if subset:
            sub = int(len(self.dataset_idx)*subset)
            self.dataset_idx = self.dataset_idx[:sub]
            
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
    TODO: early stopping/checkpoints
    load_model = None/'saved_model.pth'/'saved_model.pk'
    load_embed = None/'model_name'
    squeeze_y = True/False (torch.squeeze(y))
    adapt = (D_in, D_out, Activation, dropout rate)
    """
    def __init__(self, Datasets, Model, Sampler=Selector, Metrics=Metrics,
                 Optimizer=None, Scheduler=None, Criterion=None, 
                 ds_params={}, model_params={}, sample_params={},
                 opt_params={}, sched_params={}, crit_params={}, metrics_params={}, 
                 adapt=None, load_model=None, load_embed=None, save_model=False,
                 batch_size=10, epochs=1, squeeze_y=False, gpu=True):
        
        self.gpu = gpu
        self.bs = batch_size
        self.squeeze_y = squeeze_y
        self.ds_params = ds_params
        self.dataset_manager(Datasets, Sampler, ds_params, sample_params)
        
        self.metrics = Metrics(**metrics_params)
        self.metrics.log('model: {}\n{}'.format(Model, model_params))
        self.metrics.log('dataset: {}\n{}'.format(Datasets, ds_params))
        self.metrics.log('sampler: {}\n{}'.format(Sampler, sample_params))
        self.metrics.log('epochs: {}, batch_size: {}, save_model: {}, load_model: {}'.format(
                                                    epochs, batch_size, save_model, load_model))
        
        if load_model is not None:
            try: #uses the same embed params for all datasets (train/val/test)
                model = Model(model_params)
                model.load_state_dict(load('./models/'+load_model))
                print('model loaded from state_dict...')
            except:
                model = load('./models/'+load_model)
                print('model loaded from pickle...')                                                      
        else:
            model = Model(model_params)
        
        if load_embed is not None:
            for i, embedding in enumerate(model.embeddings):
                try:
                    weight = np.load('./models/{}_{}_embedding_weight.npy'.format(
                                                                            load_embed, i))
                    embedding.from_pretrained(from_numpy(weight), 
                                              freeze=model_params['embeds'][i][4])
                    print('loading embedding weights...')
                except:
                    print('no embedding weights found.  initializing... ')
                    
        if adapt is not None: 
            model.adapt(adapt)
        
        if self.gpu:
            try:
                self.model = model.to('cuda:0')
                print('running model on gpu...')
            except:
                print('gpu not available.  running model on cpu...')
                self.model = model
                self.gpu = False
        else:
            print('running model on cpu...')
            self.model = model
            
        self.metrics.log(self.model.children)
        
        if Criterion:
            self.criterion = Criterion(**crit_params)
            if self.gpu: self.criterion.to('cuda:0')
            self.metrics.log('criterion: {}\n{}'.format(self.criterion, crit_params))
            self.opt = Optimizer(self.model.parameters(), **opt_params)
            self.metrics.log('optimizer: {}\n{}'.format(self.opt, opt_params))
            self.scheduler = Scheduler(self.opt, **sched_params)
            self.metrics.log('scheduler: {}\n{}'.format(self.scheduler, sched_params))
            
            for e in range(epochs):
                self.metrics.epoch = e
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
                            self.metrics.start.strftime("%Y%m%d_%H%M")))
            if not adapt: save(self.model.state_dict(), './models/{}.pth'.format(
                                         self.metrics.start.strftime("%Y%m%d_%H%M")))
            if hasattr(self.model, 'embeddings'):
                for i, embedding in enumerate(self.model.embeddings):
                    weight = embedding.weight.detach().cpu().numpy()
                    np.save('./models/{}_{}_embedding_weight.npy'.format(
                        self.metrics.start.strftime("%Y%m%d_%H%M"), i), weight)
        self.metrics.report()
        
    def run(self, flag): 
        e_loss, e_sk, i = 0, 0, 0
        
        if flag == 'train': 
            self.model.training = True
            dataset = self.train_ds
            drop_last = True
            
        if flag == 'val':
            self.model.training = False
            dataset = self.val_ds
            drop_last = True

        if flag == 'test':
            self.model.training = False
            dataset = self.test_ds
            drop_last = True
            
        if flag == 'infer':
            self.model.training = False
            dataset = self.test_ds
            drop_last = False
            
        #dataset = dataset.to('cuda:0', non_blocking=True)
        dataloader = DataLoader(dataset, batch_size=self.bs, 
                                sampler=self.sampler(flag=flag), 
                                num_workers=8, pin_memory=True, 
                                            drop_last=drop_last)
        
        def to_cuda(data):
            if len(data) < 1: return None
            elif not self.gpu: return data
            else: return data.to('cuda:0', non_blocking=True) 
        
        for X, embeds, y in dataloader:
            i += self.bs
            X = to_cuda(as_tensor(X))
            if len(embeds) > 0:
                embeds = [to_cuda(as_tensor(emb)) for emb in embeds]
                y_pred = self.model(X, embeds)
            else:
                y_pred = self.model(X)
                
            if flag == 'infer':
                self.metrics.predictions.append(np.concatenate((y_pred, y), axis=1))
            else:
                y = to_cuda(as_tensor(y))
                if self.squeeze_y:
                    y = squeeze(y)
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
            self.metrics.sk_metric(flag)
            
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


        
        
