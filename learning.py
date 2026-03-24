from datetime import datetime
import logging, random, os, gc, sys

os.environ['NUMEXPR_MAX_THREADS'] = '8'

import numpy as np

from torch import no_grad, save, load, from_numpy, cat
from torch import cuda, is_tensor
from torch.utils.data import Sampler, DataLoader
from torch.nn import functional as F

from torcheval.metrics import functional as t_metric

from sklearn import metrics as sk_metric


class Metric():
    """
    """
    sk_metric = ['accuracy_score','roc_auc_score']
    torch_metric = ['auc','multiclass_accuracy','multiclass_auprc','binary_accuracy']
    
    def __init__(self, report_interval=1, metric_name=None,
                    dir='/app/data/', min_lr=.00125, last_n=1, metric_param={}):

        now = datetime.now()
        self.dir = dir
        self.start = now
        self.report_time = now
        self.report_interval = report_interval
        self.last_n = last_n
        self.min_lr = min_lr
        
        self.epoch, self.e_loss, self.n = 0, 0, 0
        self.train_loss, self.val_loss, self.test_loss = [], [], []
        self.predictions, self.lr_log = [], []
        
        self.metric_name, self.metric_param = metric_name, metric_param
        self.metric_func, self.metric_train, self.metric_val = None, [], []
        self.y, self.y_pred = [], []
        
        if self.metric_name is not None:
            if self.metric_name in ['transformer']:
                self.metric_func = None
            elif self.metric_name in Metric.sk_metric:
                self.metric_func = getattr(sk_metric, self.metric_name)
            elif self.metric_name in Metric.torch_metric:
                self.metric_func = getattr(t_metric, self.metric_name)
            else:
                raise Exception('hey just what you see pal...')
                
        log_file = os.path.join(self.dir, 'cosmosis.log')
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Only add handlers if they aren't already there
        if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
            # File Handler (for your Frontend/Volume)
            file_h = logging.FileHandler(log_file)
            file_h.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            root_logger.addHandler(file_h)

        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            # Stream Handler (for your Terminal/Skaffold)
            stream_h = logging.StreamHandler(sys.stdout)
            stream_h.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            root_logger.addHandler(stream_h)

        self.log('.....................\nnew experiment: {}'.format(self.start))
    
    def infer(self):
        """
        process the predictions and save
        """
        now = datetime.now()
        self.log('inference job: {}'.format(self.start))
        self.log('total inference time: {}'.format(now - self.start))

        if self.metric_name == 'transformer':
            predictions = F.softmax(self.predictions[-1].squeeze(), dim=-1)
            predictions = predictions.argmax(dim=-1)
            predictions = predictions.detach().cpu().numpy().tolist()
            predictions = self.decoder(predictions)
            predictions = np.asarray(predictions).reshape((1,-1))
        else:
            predictions = cat(self.predictions).detach().cpu().numpy()
        self.log('predictions: {}'.format( predictions))
        self.predictions = []
        
    def softmax_overflow(x):
        x_max = x.max(axis=1, keepdims=True)
        normalized = np.exp(x - x_max)
        return normalized / normalized.sum(axis=1, keepdims=True)
        
    def metric(self, flag):
        """
        called at the end of each run() loop
        TODO multiple metric
        flags = train, val, test, infer
        """
        if self.metric_func == None:
            return
        
        y_pred = cat(self.y_pred, dim=0)
        y = cat(self.y, dim=0)

        # preprocess
        if self.metric_name in ['accuracy_score','multiclass_accuracy','multiclass_auprc']:
            y_pred = F.softmax(y_pred, dim=-1)

        if self.metric_name in ['accuracy_score','multiclass_accuracy']:
            y_pred = y_pred.argmax(dim=-1)

        # sklearn metric preprocess
        if self.metric_name in Metric.sk_metric:
            y_pred = y_pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            
        # torch metric
        if self.metric_name in Metric.torch_metric: 
            score = self.metric_func(y_pred, y, **self.metric_param)
        else:
        # sklearn
            score = self.metric_func(y, y_pred, **self.metric_param)
        
        score = score.cpu().item() if is_tensor(score) else score

        if flag == 'train':
            self.metric_train.append(score)
        else:
            self.metric_val.append(score)
        
    def log(self, message):
        logging.info(message)
        for handler in logging.getLogger().handlers:
            handler.flush()
        
    def report(self, y_pred, y, flag):
        """
        called at the end of each run() loop
        """
        if flag == 'train': return 
            
        now = datetime.now()
        tot_elapsed = now - self.start
        self.log('epoch: {}, elapsed time: {}'.format(self.epoch, tot_elapsed))

        if len(self.predictions) > 0: 
            self.log('len(self.predictions): {}'.format(len(self.predictions)))
            return

        if self.epoch % self.report_interval != 0: return
        
        if self.metric_name == 'transformer':
            # get the last instance
            y_pred = y_pred[-1]
            y = y[-1]
            y_pred = F.softmax(y_pred, dim=-1)
            y_pred = y_pred.argmax(dim=0)
            y_pred = y_pred.detach().cpu().numpy().tolist()
            y_pred = self.decoder(y_pred)
            y = y.detach().cpu().numpy().tolist()
            y = self.decoder(y)
            
        self.log('last {} y_pred values: \n{}\nlast {} y values: \n{}'.format(
                    self.last_n, y_pred[-self.last_n:], self.last_n, y[-self.last_n:]))

        self.log('train loss: {}, val loss: {}, lr: {}'.format(
                    self.train_loss[-1], self.val_loss[-1], self.lr_log[-1]))

        if len(self.metric_train) != 0:
            self.log('{} train score: {}, validation score: {}'.format(
                self.metric_name, self.metric_train[-1], self.metric_val[-1]))
    
    def loss(self, flag):
        """
        called at the end of each run() loop
        """
        if flag == 'train':
            self.train_loss.append(self.e_loss/self.n)
        if flag == 'val':
            self.val_loss.append(self.e_loss/self.n)
        if flag == 'test':
            self.test_loss.append(self.e_loss/self.n)
            
    def reset_loop(self):
        """
        called at the end of each run() loop
        """
        self.n, self.e_loss = 0, 0
        self.y, self.y_pred = [], []

    def final(self):
        now = datetime.now()
        self.log('........final........\ntotal learning time: {}'.format(now - self.start))

        if len(self.test_loss) != 0:
            self.log('test loss: {}'.format(self.test_loss))
            
        if len(self.metric_train) != 0:
            self.log('{} test metric: {}'.format(self.metric_name, self.metric_val[-1]))


class Selector(Sampler):
    """splits = (train_split,) remainder is val_split or 
                (train_split,val_split) remainder is test_split or None
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
    
    load_model = None/'model_name.pth'/'model_name.pk'
    
    load_embed = True/False 

    save_model = None/'model_name.pth'
    
    adapt = (D_in, D_out, dropout)
        prepends a trainable linear layer

    weights_only = True/False
        enable un pickling of models = False (only unpickle trusted files)
        
    the dataset output can either be a dictionary utilizing the form 
    data = {'model_input': {},
            'criterion_input': {'target':{}}} 
    or an object with a feature 'target' (data.target)
    the entire data object is passed to the model
    """
    def __init__(self, Datasets, Model, Sampler=Sampler, Metric=Metric,
                 DataLoader=DataLoader,
                 Optimizer=None, Scheduler=None, Criterion=None, 
                 ds_param={}, model_param={}, sample_param={},
                 opt_param={}, sched_param={}, crit_param={}, metric_param={}, 
                 adapt=None, load_model=None, load_embed=False, save_model=False,
                 batch_size=10, epochs=1, dir='/app/data/',
                 gpu=False, weights_only=False, num_workers=0, target='y'):
        
        self.dir = dir
        self.weights_only = weights_only
        self.num_workers = num_workers
        self.save_model = save_model
        self.gpu = gpu
        self.bs = batch_size
        self.epochs = epochs
        self.target = target
        self.ds_param = ds_param
        self.dataset_manager(Datasets, Sampler, ds_param, sample_param)
        self.DataLoader = DataLoader
        self.criterion = Criterion(**crit_param) if Criterion is not None else None
        
        self.metric = Metric(**metric_param)
        self.metric.gpu = gpu
        if hasattr(self.train_ds, 'encoding'): # retain the encodings for later use in decoding
            self.metric.decoder = self.train_ds.encoding.decode
        
        self.metric.log('model: {}\n{}\ndataset: {}\n{}\nsampler: {}\n{}'.format(
                            Model, model_param, Datasets, ds_param, Sampler, sample_param))
        self.metric.log('epochs: {}, batch_size: {}, save_model: {}, load_model: {}'.format(
                                                        epochs, batch_size, save_model, load_model))

        if load_model is not None:
            if os.path.exists(self.dir + load_model):
                try: 
                    model = Model(model_param)
                    model.load_state_dict(load(self.dir + load_model, weights_only=self.weights_only))
                    self.metric.log('model loaded from state_dict...')
                except:
                    model = load(self.dir + load_model, weights_only=self.weights_only)
                    self.metric.log('model loaded from pickle...')                                                   
            else:
                model = Model(model_param)

        if load_embed is True:
            try:
                for feature, embedding in model.embedding_layer.items():
                    weight = np.load(self.dir + '{}_{}_embedding_weight.npy'.format(load_model[:-4], feature))
                    embedding.from_pretrained(from_numpy(weight), freeze=model_param['embed_param'][feature][3])
                self.metric.log('loading embedding weights...')
            except:
                self.metric.log('embedding weights failed to load.  reinitializing...')
                
        if adapt is not None: model.adapt(*adapt)

        if self.gpu == True:
            try:
                model.to('cuda:0')
                model.device = 'cuda:0'
                self.metric.log('running model on gpu...')
            except:
                self.metric.log('gpu not available.  running model on cpu...')
                self.gpu = False
                model.device = 'cpu'
        else:
            self.metric.log('running model on cpu...')
            model.gpu = 'cpu'

        self.model = model
        self.metric.log('\n{}'.format(self.model.children))
    
        if self.criterion is not None:
            self.criterion = Criterion(**crit_param)
            if self.gpu: self.criterion.to('cuda:0')
            self.metric.log('\ncriterion: {}\n{}'.format(self.criterion, crit_param))
            self.opt = Optimizer(self.model.parameters(), **opt_param)
            self.metric.log('\noptimizer: {}\n{}'.format(self.opt, opt_param))
            self.scheduler = Scheduler(self.opt, **sched_param)
            self.metric.log('\nscheduler: {}\n{}'.format(self.scheduler, sched_param))

    # primary loop       
    def run_experiment(self, prompt=None):
        if self.criterion is not None:
            for e in range(self.epochs):
                self.metric.epoch = e
                self.sampler.shuffle_train_val_idx()
                self.run('train')
                with no_grad():
                    self.run('val')
                    if e > 1 and self.metric.lr_log[-1] <= self.metric.min_lr:
                        self.metric.log('early stopping!  learning rate is below the set minimum...')
                        break
            with no_grad():
                self.run('test')
            self.metric.final()
            
        else: # no Criterion implies inference mode
            with no_grad():
                for e in range(self.epochs): 
                    self.run('infer', prompt=prompt)
                    self.metric.infer()
                    
        if self.save_model:
            if type(self.save_model) == str:
                model_name = self.save_model
            else:
                model_name = self.metric.start.strftime("%Y%m%d_%H%M")

            save_path = os.path.join(self.dir, f"{model_name}")    
            try: 
                save(self.model.state_dict(), save_path + '.pth')
                self.metric.log('model state dict saved...')
            except:
                save(self.model, save_path + '.pk')
                self.metric.log('model has been pickled...')
                     
            if hasattr(self.model, 'embedding_layer'):
                for feature, embedding in self.model.embedding_layer.items():
                    weight = embedding.weight.detach().cpu().numpy()
                    np.save(save_path + '_{}_{}_embedding_weight.npy'.format(model_name, feature), weight)
                self.metric.log('model embeddings saved...')

            self.metric.log('model: {} saved...'.format(model_name))

        del self.model
        del self.metric
        gc.collect()
        if self.gpu: cuda.empty_cache()
        print('experiment complete...')

    # secondary loop
    def run(self, flag, prompt=None): 
        
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
            self.model.generate = True
            dataset = dataset.prompt(prompt)
            
        dataloader = self.DataLoader(dataset, batch_size=self.bs, 
                                     sampler=self.sampler(flag=flag), 
                                     num_workers=self.num_workers, 
                                     pin_memory=self.gpu, 
                                     drop_last=drop_last)
        # tertiary loop
        for data in dataloader:
            if self.gpu: # overwrite the datadic with a new copy on the gpu
                if type(data) == dict: 
                    _data = {}
                    for k, v in data.items():
                        _data[k] = data[k].to('cuda:0', non_blocking=True)
                    data = _data
                else: 
                    data = data.to('cuda:0', non_blocking=True)

            y_pred = self.model(data)
            
            if flag == 'infer':
                self.metric.predictions.append(y_pred)
                y = None
            else:
                if type(data) == dict: y = data[self.target]
                else: y = getattr(data, self.target)
                    
                self.opt.zero_grad()
                b_loss = self.criterion(y_pred, y)
                self.metric.e_loss += b_loss.item()
                self.metric.n += self.bs
                
                if self.metric.metric_func is not None: self.metric.y.append(y)
                if self.metric.metric_func is not None: self.metric.y_pred.append(y_pred)
                    
                if flag == 'train':
                    b_loss.backward()
                    self.opt.step()

        if flag == 'val': 
            self.scheduler.step(self.metric.e_loss)
            self.metric.lr_log.append(self.opt.param_groups[0]['lr'])
            
        self.metric.metric(flag)
        self.metric.loss(flag)
        self.metric.report(y_pred, y, flag)
        self.metric.reset_loop()
                
    def dataset_manager(self, Datasets, Sampler, ds_param, sample_param):
        
        if len(Datasets) == 1:
            self.train_ds = Datasets[0](**ds_param['train_param'])
            self.val_ds = self.test_ds = self.train_ds
            self.sampler = Sampler(dataset_idx=self.train_ds.ds_idx, 
                                       **sample_param)

        if len(Datasets) == 2:
            self.train_ds = Datasets[0](**ds_param['train_param'])
            self.val_ds = self.train_ds
            self.test_ds = Datasets[1](**ds_param['test_param'])
            self.sampler = Sampler(train_idx=self.train_ds.ds_idx, 
                                       test_idx=self.test_ds.ds_idx,
                                           **sample_param)
        if len(Datasets) == 3:
            self.train_ds = Datasets[0](**ds_param['train_param'])
            self.val_ds = Datasets[1](**ds_param['val_param'])
            self.test_ds = Datasets[2](**ds_param['test_param'])
            self.sampler = Sampler(train_idx=self.train_ds.ds_idx, 
                                       val_idx=self.val_ds.ds_idx, 
                                           test_idx=self.test_ds.ds_idx,
                                               **sample_param)


        
        

