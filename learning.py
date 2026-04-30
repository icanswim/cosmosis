from datetime import datetime
from pathlib import Path
import logging, random, os, gc, sys

os.environ['NUMEXPR_MAX_THREADS'] = '8'

import numpy as np

from torch import no_grad, save, load, from_numpy, cat
from torch import cuda, is_tensor
from torch.utils.data import Sampler, DataLoader
from torch.nn import functional as F

from torcheval.metrics import functional as t_metric

from sklearn import metrics as sk_metric

logger = logging.getLogger(__name__)

class Metric():
    
    sk_metric = ['accuracy_score','roc_auc_score']
    torch_metric = ['auc','multiclass_accuracy','multiclass_auprc','binary_accuracy']
    
    def __init__(self, report_interval=1, metric_name=None,
                    min_lr=.00125, last_n=1, metric_param={}):

        now = datetime.now()
        self.start = now
        self.report_time = now
        self.report_interval = report_interval
        self.last_n = last_n
        self.min_lr = min_lr
        
        self.epoch, self.e_loss, self.n = 0, 0, 0
        self.train_loss, self.val_loss, self.test_loss = [], [], []
        self.predictions, self.lr = [], []
        
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
                raise Exception('metric function not found...')
                
    @classmethod
    def setup_logging(cls, log_name='cosmosis', log_dir='./data'):

        if log_name is None: log_name = __name__

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")

        try:
            os.makedirs(log_dir, exist_ok=True)
        except PermissionError as e:
            print(f"logging failed for {log_dir}, error: {e}.")
            sys.exit(1)

        class FlushFileHandler(logging.FileHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()
                try:
                    os.fsync(self.stream.fileno())
                except Exception: 
                    pass

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            handlers=[
                FlushFileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )

        logger = logging.getLogger(log_name) 
        logger.info(f'{log_name} logging initialized at: {log_file}')
        return logger

    
    def infer(self):
        """
        process the predictions and save
        """
        if self.metric_name == 'transformer':
            predictions = F.softmax(self.predictions[-1].squeeze(), dim=-1)
            predictions = predictions.argmax(dim=-1)
            predictions = predictions.detach().cpu().numpy().tolist()
            predictions = self.decoder(predictions)
            predictions = [predictions]
        else:
            predictions = cat(self.predictions).detach().cpu().numpy()
        logger.info('learn.infer predictions: {}'.format( predictions))
        self.predictions = []
        return {'learn.infer predictions': predictions}

    def softmax_overflow(x):
        x_max = x.max(axis=1, keepdims=True)
        normalized = np.exp(x - x_max)
        return normalized / normalized.sum(axis=1, keepdims=True)
        
    def metric(self, flag):
        """
        called at the end of each run() loop
        runs and aggregates the metric function over the epoch
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
        
    def loss(self, flag):
        if flag == 'infer':
            return
        avg_loss = self.e_loss / self.n
        if flag == 'train':
            self.train_loss.append(avg_loss)
        elif flag == 'val':
            self.val_loss.append(avg_loss)
        elif flag == 'test':
            self.test_loss.append(avg_loss)
            logger.info(f'metric.loss test loss: {avg_loss}')

    def report(self, y_pred, y, flag):

        if flag == 'train' or flag == 'test': 
            return 

        now = datetime.now()
        tot_elapsed = now - self.start
        
        logger.info(f'learn.report epoch: {self.epoch} elapsed: {tot_elapsed}')

        if len(self.predictions) > 0: 
            logger.info(f'learn.report inference mode: {len(self.predictions)} predictions')
            return

        if self.epoch % self.report_interval != 0: 
            return
        
        if self.metric_name == 'transformer':
            y_pred_val = self.decoder(F.softmax(y_pred[-1], dim=-1).argmax(dim=0).detach().cpu().numpy().tolist())
            y_val = self.decoder(y[-1].detach().cpu().numpy().tolist())
        else:
            y_pred_val = y_pred[-self.last_n:]
            y_val = y[-self.last_n:]

        logger.info(f'metric.report last {self.last_n} predictions: {y_pred_val}')
        logger.info(f'metric.report last {self.last_n} targets:     {y_val}')
        
        logger.info(f'metric.report train loss: {self.train_loss[-1]:.4f} val loss: {self.val_loss[-1]:.4f} lr: {self.lr[-1]}')

        if len(self.metric_train) != 0:
            logger.info(f'metric.report metric: {self.metric_name} | train: {self.metric_train[-1]:.4f} | val: {self.metric_val[-1]:.4f}')

    
    def reset_loop(self):
        """
        called at the end of each run() loop
        """
        self.n, self.e_loss = 0, 0
        self.y, self.y_pred = [], []

    def final(self):
        now = datetime.now()
        logger.info('metric.final...\ntotal learning time: {}'.format(now - self.start))
        final = {'training job': 'completed', 
                 'total_time': now - self.start}
        if len(self.test_loss) != 0:
            logger.info('metric.final test loss: {}'.format(self.test_loss))
            final['test_loss'] = self.test_loss
        if len(self.metric_train) != 0:
            logger.info('metric.final {} test metric: {}'.format(self.metric_name, self.metric_val[-1]))
            final[self.metric_name] = self.metric_val[-1]
        return final

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
    load_model = True/False or 'model_name'
    save_model = True/False or 'model_name' (if True, saved with timestamp)
    load_embed = True/False or 'embedding_name' 
        (if True loads embedding weights with prefix 'model_name')
    """
    def __init__(self, Datasets, Model, Metric, 
                 Sampler=Selector, DataLoader=DataLoader,
                 Optimizer=None, Scheduler=None, Criterion=None,
                 ds_param={}, model_param={}, sample_param={},
                 opt_param={}, sched_param={}, crit_param={}, metric_param={}, 
                 adapt=None, load_model=False, save_model=True,
                 batch_size=10, epoch=1, dir='./data',
                 gpu=False, num_workers=0, target='y'):
        
        self.dir = dir
        self.num_workers = num_workers
        self.save_model = save_model
        self.load_model = load_model
        self.gpu = gpu
        self.bs = batch_size
        self.epoch = epoch
        self.target = target

        try:
            os.makedirs(self.dir, exist_ok=True)
        except PermissionError as e:
            logger.error(f"learn.__init__ data dir creation failed for {self.dir}, error: {e}.")
            sys.exit(1)
        
        self.dataset_manager(Datasets, Sampler, ds_param, sample_param)
        self.DataLoader = DataLoader
        self.metric = Metric(**metric_param)
        self.metric.gpu = gpu
        self.device = 'cuda:0' if self.gpu else 'cpu'
        if hasattr(self.train_ds, 'encoding'):
            self.metric.decoder = self.train_ds.encoding.decode

        model = self.model_loader(Model, model_param, name=load_model)
        
        if adapt is not None: 
            model.adapt(*adapt)

        try:
            model.to(self.device)
            model.device = self.device
            logger.info(f'learn.__init__ running model on {self.device}...')
        except Exception:
            logger.warning('learn.__init__ gpu not available. on cpu...')
            self.gpu = False
            self.device = 'cpu'
            model.to('cpu')
            model.device = 'cpu'

        self.model = model
        
        if Criterion is not None:
            self.criterion = Criterion(**crit_param)
            if self.gpu: self.criterion.to(self.device)
            self.opt = Optimizer(self.model.parameters(), **opt_param)
            self.scheduler = Scheduler(self.opt, **sched_param)
            logger.info(f'learn.__init__ learner ready: {Criterion.__name__}, {Optimizer.__name__}, Scheduler: {Scheduler.__name__}')
        else:
            self.criterion = None
            self.opt, self.scheduler = None, None
            logger.info(f'learn.__init__ inference engine ready...')

    def model_loader(self, Model, model_param, name=None):
        model = Model(model_param)
        self.model_param = model_param
        
        if type(name) != str:
            logger.info("learn.__init__ initializing new model {}...".format(model.__class__.__name__))
            return model

        base_path = Path(self.dir) / name
        pth_path = base_path.with_suffix('.pth')
        
        if pth_path.exists():
            state_dict = load(pth_path, weights_only=True, map_location=self.device)
            model.load_state_dict(state_dict)
            logger.info(f"learn.__init__ model loaded from: {pth_path.name}")
        else:
            logger.warning(f"learn.__init__ no file found at {pth_path}. initialized new model...")

        # load embeddings
        if hasattr(model, 'embedding_layer'):
            try:
                for feat, embedding in model.embedding_layer.items():
                    w_path = Path(self.dir) / f"{name}_{feat}_embedding_weight.npy"
                    freeze = model_param['embed_param'][feat][3]
                    np_weights = np.load(w_path)
                    embedding.from_pretrained(from_numpy(np_weights), freeze=freeze)
                model.to(self.device)
                logger.info("learn.__init__ embedding weights loaded successfully...")
            except Exception as e:
                logger.warning(f"learn.__init__ embedding weights failed to load: {e}")

        logger.info('learn.__init__ model loaded: {}'.format(model.__class__.__name__))
        return model
    
    def reload_model(self, name=None):
        self.model = self.model_loader(type(self.model), self.model_param, name=name)
        logger.info("learn.reload_model model reloaded successfully.")
    
    def model_saver(self):

        if self.save_model is True:
            name = self.metric.start.strftime("%Y%m%d_%H%M")
        elif type(self.save_model) == str: 
            name = self.save_model
        else:
            logger.info("learn.model_saver model saving skipped...")
            return
            
        base_path = Path(self.dir) / name
        pth_path = base_path.with_suffix('.pth')
        save(self.model.state_dict(), pth_path)
        logger.info(f"learn.model_saver saved state_dict: {pth_path.name}")
                
        if hasattr(self.model, 'embedding_layer'):
            try:
                for feat, emb in self.model.embedding_layer.items():
                    w_path = Path(self.dir) / f"{name}_{feat}_embedding_weight.npy"
                    np.save(w_path, emb.weight.detach().cpu().numpy())
                logger.info(f"learn.model_saver embeddings saved with prefix '{name}'")
            except Exception as e:
                logger.warning(f"learn.model_saver failed to save embeddings: {e}")

    def run_experiment(self, prompt=False):
        if not prompt and self.criterion is not None:
            for e in range(self.epoch):
                self.metric.epoch = e + 1
                self.sampler.shuffle_train_val_idx()
                self.run('train')
                with no_grad():
                    self.run('val')
                    if e > 1 and self.metric.lr[-1] <= getattr(self.metric, 'min_lr', 0):
                        logger.info('learn.run_experiment early stopping.  lr below minimum...')
                        break
            with no_grad():
                self.run('test')
            output = self.metric.final()
        else:
            with no_grad():
                self.run('infer', prompt=prompt)
                output = self.metric.infer()
                    
        self.model_saver()
        self.cleanup()
        return output

    def run(self, flag, prompt=False):

        if flag == 'train': 
            dataset = self.train_ds
            self.model.train(True)

        elif flag == 'val':   
            dataset = self.val_ds
            self.model.train(False)

        elif flag == 'test':  
            dataset = self.test_ds
            self.model.train(False)
 
        elif flag == 'infer':
            self.model.train(False)
            if prompt: 
                self.test_ds.ds = self.test_ds.prompt(prompt)
            dataset = self.test_ds
            self.model.generate = True
        logger.info(f'learn.run {flag}')
        dataloader = self.DataLoader(dataset, batch_size=self.bs, 
                                     sampler=self.sampler(flag), 
                                     num_workers=self.num_workers, 
                                     pin_memory=self.gpu, 
                                     drop_last=(flag != 'infer'))

        for data in dataloader:
            if isinstance(data, dict):
                data = {k: v.to(self.device, non_blocking=self.gpu) if hasattr(v, 'to') else v for k, v in data.items()}
                y = data[self.target] if flag != 'infer' else None  
            else:
                data = data.to(self.device, non_blocking=self.gpu)
                y = getattr(data, self.target) if flag != 'infer' else None

            y_pred = self.model(data)
            if flag == 'infer':
                self.metric.predictions.append(y_pred)
                return
                
            if flag == 'train':
                self.opt.zero_grad()
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.opt.step()
            else:
                loss = self.criterion(y_pred, y)
            self.metric.e_loss += loss.item()
            self.metric.n += self.bs
            if self.metric.metric_func:
                self.metric.y.append(y.detach().cpu())
                self.metric.y_pred.append(y_pred.detach().cpu())

        if flag == 'val' and self.scheduler:
            self.scheduler.step(self.metric.e_loss)
            self.metric.lr.append(self.opt.param_groups[0]['lr'])
            
        self.metric.metric(flag)
        self.metric.loss(flag)
        self.metric.report(y_pred, y, flag)
        self.metric.reset_loop()

    def cleanup(self):
        #del self.model
        #del self.metric
        gc.collect()
        if self.gpu: cuda.empty_cache()
        logger.info('learn.cleanup experiment complete...')

                
    def dataset_manager(self, Datasets, Sampler, ds_param, sample_param):

        if len(Datasets) == 1:
            self.train_ds = Datasets[0](**ds_param['train_param'])
            self.val_ds = self.test_ds = self.train_ds
            self.sampler = Sampler(dataset_idx=self.train_ds.ds_idx, 
                                       **sample_param)

        elif len(Datasets) == 2:
            self.train_ds = Datasets[0](**ds_param['train_param'])
            self.val_ds = self.train_ds
            self.test_ds = Datasets[1](**ds_param['test_param'])
            self.sampler = Sampler(train_idx=self.train_ds.ds_idx, 
                                       test_idx=self.test_ds.ds_idx,
                                           **sample_param)
        elif len(Datasets) == 3:
            self.train_ds = Datasets[0](**ds_param['train_param'])
            self.val_ds = Datasets[1](**ds_param['val_param'])
            self.test_ds = Datasets[2](**ds_param['test_param'])
            self.sampler = Sampler(train_idx=self.train_ds.ds_idx, 
                                       val_idx=self.val_ds.ds_idx, 
                                           test_idx=self.test_ds.ds_idx,
                                               **sample_param)
        else:
            raise ValueError('learn.dataset_manager check datasets...')
        
        

