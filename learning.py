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


class Learn():
    """
    Datasets = [TrainDS, ValDS, TestDS]
        if 1 DS is given it is split into train/val/test using splits param
        if 2 DS are given first one is train/val second is test
        if 3 DS are given first is train second is val third is test
        
    Criterion = None implies inference mode
    """
    def __init__(self, Datasets, Model, Sampler, 
                 Optimizer=None, Scheduler=None, Criterion=None, 
                 ds_params={}, model_params={}, sample_params={},
                 opt_params={}, sched_params={}, crit_params={}, 
                 adapt=False, load_model=False, load_embed=False, save_model=False,
                 batch_size=10, epochs=1):
        
        logging.basicConfig(filename='./logs/cosmosis.log', level=20)
        start = datetime.now()
        logging.info('New experiment...\n\n model: {}, start time: {}'.format(
                                        Model, start.strftime('%Y%m%d_%H%M')))
        self.bs = batch_size
        self.ds_params = ds_params
        self.dataset_manager(Datasets, Sampler, ds_params, sample_params)
        
        if load_model:
            try:
                model = Model(**model_params)
                model.load_state_dict(load('./models/'+load_model))
                print('model loaded from state_dict...')
            except:
                model = load('./models/'+load_model)
                print('model loaded from pickle...')                                                      
        else:
            model = Model(**model_params)
        
        if load_embed:
            for i, embedding in enumerate(model.embeddings):
                try:
                    weight = np.load('./models/{}_{}_embedding_weight.npy'.format(
                                                                load_embed, i))
                    embedding.from_pretrained(from_numpy(weight), 
                                              freeze=self.ds_params['embed'][i][2])
                    print('loading embedding weights...')
                except:
                    print('no embedding weights found.  initializing... ')
                    
        if adapt: 
            model.adapt(adapt)
        self.model = model.to('cuda:0')
        
        logging.info(self.model.children)
        logging.info('dataset: {}\n{}'.format(Datasets, self.ds_params))
        logging.info('sampler: {}\n{}'.format(type(self.sampler), sample_params))
        logging.info('epochs: {}, batch_size: {}, save_model: {}, load_model: {}'.format(
                                    epochs, batch_size, save_model, load_model))
        
        if Criterion:
            self.criterion = Criterion(**crit_params).to('cuda:0')
            self.opt = Optimizer(self.model.parameters(), **opt_params)
            self.scheduler = Scheduler(self.opt, **sched_params)
            logging.info('criterion: {}\n{}'.format(type(self.criterion), crit_params))
            logging.info('optimizer: {}\n{}'.format(type(self.opt), opt_params))

            self.train_log, self.val_log = [], []
            for e in range(epochs):
                self.sampler.shuffle_train_val_idx()
                self.run('train')
                with no_grad():
                    self.run('val')
                if epochs > 10:
                    if e % int(epochs/10) == 0:
                        print('epoch: {} of {}, train loss: {}, val loss: {}, lr: {}'.\
                                format(e, epochs, self.train_log[-1], self.val_log[-1], 
                                                       self.opt.param_groups[0]['lr']))
                else:
                    print('epoch: {} of {}, train loss: {}, val loss: {}, lr: {}'.\
                            format(e, epochs, self.train_log[-1], self.val_log[-1], 
                                                   self.opt.param_groups[0]['lr']))
            with no_grad():
                self.run('test')
                
            pd.DataFrame(zip(self.train_log, self.val_log)).to_csv(
                                        './logs/'+start.strftime("%Y%m%d_%H%M"))
            self.view_log('./logs/'+start.strftime('%Y%m%d_%H%M'))
        else: # no Criterion implies inference mode
            with no_grad():
                self.run('infer')
        
        elapsed = datetime.now() - start
        
        if save_model:
            if adapt: save(self.model, './models/{}.pth'.format(
                                                            start.strftime("%Y%m%d_%H%M")))
            if not adapt: save(self.model.state_dict(), './models/{}.pth'.format(
                                                            start.strftime("%Y%m%d_%H%M")))
            if self.model.embeddings:
                for i, embedding in enumerate(self.model.embeddings):
                    weight = embedding.weight.detach().cpu().numpy()
                    np.save('./models/{}_{}_embedding_weight.npy'.format(
                                             start.strftime("%Y%m%d_%H%M"), i), weight)
                    
        logging.info('learning time: {} \n'.format(elapsed))
        print('learning time: {}'.format(elapsed))
        
    def run(self, flag): 
        e_loss, i, predictions = 0, 0, []
        
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
                y = np.reshape(y, (-1, 1)) # y = 'id'
                y_pred = np.reshape(y_pred.data.to('cpu').numpy(), (-1, 1))
                predictions.append(np.concatenate((y, y_pred), axis=1)) 
            else:
                y = to_cuda(y)
                self.opt.zero_grad()
                b_loss = self.criterion(y_pred, y)
                e_loss += b_loss.item()
                if flag == 'train':
                    b_loss.backward()
                    self.opt.step()

        if flag == 'train': 
            self.train_log.append(e_loss/i)
        if flag == 'val': 
            self.val_log.append(e_loss/i)
            self.scheduler.step(e_loss)
        if flag == 'test':  
            logging.info('test loss: {}'.format(e_loss/i))
            print('test loss: {}'.format(e_loss/i))
            print('y_pred:\n{}\n y:\n{}'.format(y_pred[:10].data, y[:10].data))
        if flag == 'infer': 
            # TODO abstraction
            logging.info('inference complete')
            predictions = np.concatenate(predictions, axis=0)
            predictions = np.reshape(predictions, (-1, 2))
            self.predictions = pd.DataFrame(predictions, columns=['id','scalar_coupling_constant'])
            self.predictions['id'] = self.predictions['id'].astype('int64')
            print('self.predictions.iloc[:10]', self.predictions.shape, self.predictions.iloc[:10])
            self.predictions.to_csv('quantum_inference.csv', 
                                    header=['id','scalar_coupling_constant'], 
                                    index=False)
            print('inference complete and saved to csv...')
    
    def dataset_manager(self, Datasets, Sampler, ds_params, sample_params):
    
        if len(Datasets) == 1:
            self.train_ds = Datasets[0](**ds_params['ds_params'])
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

    @classmethod    
    def view_log(cls, log_file):
        log = pd.read_csv(log_file)
        log.iloc[:,1:3].plot(logy=True)
        plt.show()

class Selector(Sampler):
    """splits = (train/val/test) splits the dataset_idx, does not overwrite 
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
                        
        if splits:  
            random.shuffle(self.dataset_idx)
            cut1 = int(len(self.dataset_idx)*splits[0])
            cut2 = int(len(self.dataset_idx)*splits[1])
            self.train_idx = self.dataset_idx[:cut1]
            if self.val_idx == None:
                self.val_idx = self.dataset_idx[cut1:cut1+cut2]
            if self.test_idx == None:
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