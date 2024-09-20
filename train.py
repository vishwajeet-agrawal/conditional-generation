
import torch
import pandas as pd
from torch.nn import functional as F
from functools import cached_property
import glob
import os
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
def get_attn_mask(S, device):
    N = S.size(0)
    n = S.size(1)
    attn_mask = torch.ones(N, n, n, device = device)
    for i in range(N):
        attn_mask[i, S[i], S[i]] = 0
    attn_mask = attn_mask.to(torch.bool)
    attn_mask = torch.zeros_like(attn_mask, dtype=torch.float).masked_fill_(attn_mask, float("-inf"))
    return attn_mask

def np_to_torch(X, device, dtype =  torch.float32):
    device = torch.device(device)
    return torch.from_numpy(X).to(device).type(dtype)


class Scheduler:
    def __init__(self, optimizer, lr_schedule):
        # Step function must be a function that takes an epoch number and returns a scaling factor for LR
        self.lr_schedule = lr_schedule
        self.index = 0
        self.optimizer = optimizer
    
    def step(self, step):
        # Use the custom step function to determine the scaling factor for LR
        if self.index < len(self.lr_schedule):
            if (self.lr_schedule[self.index][0] <= step):
                self.index += 1
                if (self.index < len(self.lr_schedule)):
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr_schedule[self.index][1]
                


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
            
def print_current_lr(optimizer, epoch):
    for i, param_group in enumerate(optimizer.param_groups): 
        print(f"Epoch:{epoch}, LR: {param_group['lr']}")
        break

class Trainer:
    def __init__(self, model, data_generator, config, args):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr = config.lr_schedule[0][1])
        self.config = config
        self.data_generator = data_generator
        self.args = args
        self.scheduler = Scheduler(self.optimizer, config.lr_schedule)
        self.path = self.set_path()
        self.config.path = self.path

    def set_path(self):
        subfiles = glob.glob(self.config.save_dir + '/*')
        for i in range(10000):
            if f'{i}' not in [sf.split('/')[-1] for sf in subfiles]:
                os.makedirs(self.config.save_dir + f'/{i}')
                return self.config.save_dir + f'/{i}'
        
    def train_step(self, X, Xi, I, S):
        self.optimizer.zero_grad()
        logits = self.model(X, I, S)
        loss = F.cross_entropy(logits, Xi)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def to_torch(self, X, Xi, I, S):
        X = np_to_torch(X, self.config.device, dtype=torch.int)
        Xi = np_to_torch(Xi, self.config.device, dtype=torch.long)
        I = np_to_torch(I, self.config.device, dtype=torch.int)
        S = np_to_torch(S, self.config.device, dtype=torch.int)
        return X, Xi, I, S
    
    def get_batch(self, n_samples):
        X, Xi, I, S = self.data_generator.sample_conditional(n_samples, seed = 135)
        return self.to_torch(X, Xi, I, S)
    
    def get_mini_batch(self, X, Xi, I, S):
        N = X.size(0)
        batch_num = N // self.config.batch_size
        for i in range(batch_num):
            start = i * self.config.batch_size
            end = (i + 1) * self.config.batch_size
            yield X[start:end], Xi[start:end], I[start:end], S[start:end]

    def train_epoch(self):
        X, Xi, I, S = self.get_batch(self.config.batches_per_epoch * self.config.batch_size)
        
        loss = 0
        for Xb, Xib, Ib, Sb in self.get_mini_batch(X, Xi, I, S):
            loss += self.train_step(Xb, Xib, Ib, Sb) 
        return loss/self.config.batches_per_epoch
    
    @property
    def trueprob_estimate(self):
        return torch.from_numpy(self.data_generator.estimate_true_prob).to(self.config.device)
    
    @property
    def test_data(self):
        X, Xi, I, S, P = self.data_generator.test_data
        return self.to_torch(X, Xi, I, S)

    def estimate_modelprob(self):
        X, Xi, I, S = self.test_data
        return self.model.estimate_prob(X, Xi, I, S, self.config.batch_size)
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.path + '/model.pth')
        config = om.merge({'data':self.data_generator.config}, {'model':self.model.config}, {'train':self.config})
        config = om.structured(config)
        config.train.data_path = self.data_generator.path
        config.model.num_params = self.model.parameter_count
        config.model.num_params_total = sum(self.model.parameter_count.values())
        config.data = om.create({'path':self.data_generator.path, 
                                'type': self.data_generator.config.type,
                                'source':self.data_generator.config.source.type,
                                'n_features': self.model.config.n_features, 
                                'mask': config.data.mask,
                                'eval_size': self.data_generator.config.eval_size})
        for key in self.data_generator.sampler.custom_config:
            config.data[key] = self.data_generator.sampler.custom_config[key]
        resolved_cfg = om.to_container(config, resolve = True)
        om.save(resolved_cfg, self.path + '/config.yaml')
        
    def train_eval(self, results = None):
        num_batches = self.config.batches_per_epoch
        last_log = 0
        step = 0
        for i in range(self.config.n_steps // num_batches):
            loss = self.train_epoch()
            step += num_batches
            self.scheduler.step(step)

            if step - last_log >= self.config.log_interval:
                last_log = step
                with torch.no_grad():
                    Pm = self.estimate_modelprob()
                    tvdist = (Pm - self.trueprob_estimate).abs().mean()
                    self.data_generator.save_test_data()
                    if self.config.debug.print_log:
                        print(f'Step {step}, Loss: {loss}', f'TV Distance: {tvdist.item()}')
                    # print(f'Step {step}, Loss: {loss}', f'TV Distance: {tvdist.item()}')
                    # print_current_lr(self.optimizer, step)
                    results.append({'datapath':self.data_generator.path, 
                                    'batch_size': self.config.batch_size, 
                                    'embedding_dim':self.model.config.embedding_dim, 
                                    'n_features':self.model.config.n_features,
                                    'step': step,'loss':loss, 'tv_dist': tvdist.item(),
                                    'aggregator': self.model.config.aggregator.type,
                                    'n_layers': self.model.config.aggregator.n_layers,
                                    'n_heads': self.model.config.aggregator.n_heads,
                                    'reduce_type': self.model.config.aggregator.reduce_type,
                                    'learn_adjacency': self.model.config.aggregator.learn_adjacency,
                                    'tie_embeddings': self.model.config.aggregator.tie_embeddings,
                                    'tie_aggregator': self.model.config.aggregator.tie_aggregator}
                                    ) if results is not None else None
                    
        self.save_model()

if __name__ == '__main__':
    from model import Model
    from data import DataGenerator
    from argparse import ArgumentParser
    from omegaconf import OmegaConf as om
    import time
    parser = ArgumentParser()
    parser.add_argument('--config', type = str, default='config.yaml')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--multiple', action='store_true', default=True)
    parser.add_argument('--dryrun', action='store_true', default=False)
    parser.add_argument('--configs', type = str, default='configs.csv')
    parser.add_argument('--datas', type = str, default='datas.csv')
    args = parser.parse_args()
    config = om.load(args.config)

    

    if args.dryrun:
        model = Model(config.model, args).to(config.train.device)
        data_generator = DataGenerator(config.data)
        print(f"""Training on {data_generator.sampler.num_samples} samples
               for {config.train.n_steps} steps 
               with a batch size of {config.train.batch_size} 
               with config {config.model}""")
        trainer = Trainer(model, data_generator, config.train, args)
        X, Xi, I, S = trainer.get_batch(trainer.config.eval_size) # Test Data
        a = time.time()        
        P = torch.from_numpy(
                trainer.data_generator.sampler.estimate_conditional_prob(X, I, S)).to(trainer.config.device)
        b = time.time() - a
        print(f'Time in estimating conditional probability for {trainer.config.eval_size} samples: {b} seconds')
        exit()
    if args.multiple:
        configs = pd.read_csv(args.configs)
        results = []
        # configs = configs[['n_features', 'embedding_dim', 'context_aggregator', 'n_layers']]
        # configs = configs.drop_duplicates()
        data_generators = dict()
        data_configs = pd.read_csv(args.datas)
        for j, r in data_configs.iterrows():
            n_features = int(r['n_features'])
            data_path = str(r['path'])
            for i, row in tqdm(configs.iterrows()):
                config_ = om.structured(config)
                config_.data.path = os.path.join(config.data.dir, data_path)
                config_.model.n_features = config_.data.n_features = n_features
                config_.model.embedding_dim = int(row['embedding_dim'])
                config_.model.aggregator.type = row['context_aggregator']
                config_.model.aggregator.n_layers = int(row['n_layers'])
                config_.model.aggregator.n_heads = int(row['n_heads'])
                config_.model.aggregator.reduce_type = row['reduce_type']
                config_.model.aggregator.learn_adjacency = row['learn_adjacency']
                config_.model.aggregator.tie_embeddings = row['tie_embeddings']
                config_.model.aggregator.tie_aggregator = row['tie_aggregator']
                torch.manual_seed(135)
                torch.cuda.manual_seed(135)
                model = Model(config_.model, args).to(config_.train.device)
                data_generator = data_generators.get(config_.data, DataGenerator(config_.data))
                data_generators[config_.data] = data_generator
                if config.train.debug.print_log:
                    print(f"""Training on {data_generator.sampler.num_samples} 
                        samples for {config_.train.n_steps} steps 
                        with a batch size of {config_.train.batch_size} 
                        with emb_dim {config_.model.embedding_dim}, 
                        n_features {config_.model.n_features} 
                        and context aggregator {config_.model.aggregator.type},
                        and n_layers {config_.model.aggregator.n_layers}""")
                trainer = Trainer(model, data_generator, config_.train, args)
                trainer.train_eval(results)
                pd.DataFrame(results).to_csv('results.csv')
    else:
        torch.manual_seed(135)
        torch.cuda.manual_seed(135)
        model = Model(config.model, args).to(config.train.device)
        print('Model parameters:', sum(model.parameter_count.values()))
        data_generator = DataGenerator(config.data)
        print(f"""Training on {data_generator.sampler.num_samples} samples
               for {config.train.n_steps} steps 
               with emb_dim {config.model.embedding_dim}, 
               n_features {config.data.n_features},
               with a batch size of {config.train.batch_size}
               and context aggregator {config.model.aggregator.type},
               and n_layers {config.model.aggregator.n_layers}""")
        trainer = Trainer(model, data_generator, config.train, args)
        trainer.train_eval()

