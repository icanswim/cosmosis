from datetime import datetime
import logging
import random
import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch import no_grad, save, load, from_numpy
from torch.utils.data import Sampler, DataLoader


class Learn():
    """
    save_model = True/False
    load_model = False/'./models/savedmodel.pth'
    Criterion = None implies inference mode.
    adapt = False/(dataset input shape, model input shape)
    TODO early stopping
    """
    def __init__(self, Dataset, Model, Sampler, Optimizer=None, Scheduler=None, Criterion=None, 
                 ds_params={}, model_params={}, sample_params={},
                 opt_params={}, sched_params={}, crit_params={}, 
                 adapt=False, load_model=False, load_embed=False, save_model=False,
                 batch_size=10, epochs=1):
        
        logging.basicConfig(filename='./logs/cosmosis.log', level=20)
        start = datetime.now()
        logging.info('New experiment...\n\n model: {}, start time: {}'.format(
                                        Model, start.strftime('%Y%m%d_%H%M')))
        self.bs = batch_size
        self.ds = Dataset(**ds_params)
        logging.info('dataset: {}\n{}'.format(type(self.ds), ds_params))
        logging.info('epochs: {}, batch_size: {}, save_model: {}, load_model: {}'.format(
                                    epochs, batch_size, save_model, load_model))
        print('{} dataset created...'.format(type(self.ds)))
        
        if load_model:
            try:
                model = Model(embed=self.ds.embed, **model_params)
                model.load_state_dict(load('./models/'+load_model))
                print('model loaded from state_dict...')
            except:
                model = load('./models/'+load_model)
                print('model loaded from pickle...')                                                   
                
        else:
            model = Model(embed=self.ds.embed, **model_params)
        
        if load_embed:
            for i, embedding in enumerate(model.embeddings):
                try:
                    weight = np.load('./models/{}_{}_embedding_weight.npy'.format(
                                                                load_embed, i))
                    embedding.from_pretrained(from_numpy(weight), 
                                              freeze=self.ds.embed[i][2])
                    print('loading embedding weights...')
                except:
                    print('no embedding weights found.  initializing... ')
                    
        if adapt: 
            model.adapt(adapt)
            
        self.model = model.to('cuda:0')
        logging.info(self.model.children)
        
        self.sampler = Sampler(self.ds.ds_idx, **sample_params)
        logging.info('sampler: {}\n{}'.format(type(self.sampler), sample_params))
        
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
        else: 
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
            drop_last = True
        if flag == 'val':
            self.model.training = False
            drop_last = True
        if flag == 'test':
            self.model.training = False
            drop_last = True
        if flag == 'infer':
            self.model.training = False
            drop_last = False
            
        dataloader = DataLoader(self.ds, batch_size=self.bs, shuffle=False, 
                                sampler=self.sampler(flag=flag), batch_sampler=None, 
                                num_workers=8, collate_fn=None, pin_memory=True, 
                                drop_last=drop_last, timeout=0, worker_init_fn=None)
  
        def to_cuda(data):
            if len(data) == 0: return None
            else: return data.to('cuda:0', non_blocking=True)

        for  X, y, embed in dataloader:
            i += self.bs
            X = to_cuda(X)
            embed = [to_cuda(emb) for emb in embed]
            
            y_pred = self.model(X=X, embed=embed)
            
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

    @classmethod    
    def view_log(cls, log_file):
        log = pd.read_csv(log_file)
        log.iloc[:,1:3].plot(logy=True)
        plt.show()

class Selector(Sampler):
    """A base class for subset selection for creating train, validation and test sets.
    Very fast, optimized for large datasets.  It is also possible to do filtering here 
    or at the dataset level.  
    subset = create randomly selected subset of size = set_size*subset
    splits = (test,train) remainer = val set
    set_seed = False/seed for reproducible train/val/test set selection
    TODO memory optimization
    """
   
    def __init__(self, dataset_idx, splits=(.1,.8), subset=False, set_seed=False):
        self.set_seed = set_seed
        self.splits = splits 
        if subset:
            self.dataset_idx = random.sample(dataset_idx, int(len(dataset_idx)*subset))
        else:    
            self.dataset_idx = dataset_idx
        
        if self.set_seed: 
            random.seed(self.set_seed)
        random.shuffle(self.dataset_idx)
        cut1 = int(len(self.dataset_idx)*self.splits[0])
        cut2 = int(len(self.dataset_idx)*self.splits[1])
        self.test_idx = self.dataset_idx[:cut1]
        self.train_idx = self.dataset_idx[cut1:cut1+cut2]
        self.val_idx = self.dataset_idx[cut1+cut2:]
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
        random.shuffle(self.val_idx)
        random.shuffle(self.train_idx)
