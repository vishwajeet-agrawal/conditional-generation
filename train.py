
import torch
import pandas as pd
from torch.nn import functional as F
from functools import cached_property
import glob
import os
import numpy as np
from tqdm import tqdm
from model import GeneralModel as Model
from model import ExpModel
from data import DataGenerator
from argparse import ArgumentParser
from omegaconf import OmegaConf as om
import time
from util import torch_rand

def get_attn_mask(S, device):
    N = S.size(0)
    n = S.size(1)
    attn_mask = torch.ones(N, n, n, device = device)
    for i in range(N):
        attn_mask[i, S[i], S[i]] = 0
    attn_mask = attn_mask.to(torch.bool)
    attn_mask = torch.zeros_like(attn_mask, dtype=torch.float).masked_fill_(attn_mask, float("-inf"))
    return attn_mask

def np_to_torch(X, device, dtype = torch.float32):
    device = torch.device(device)
    return torch.from_numpy(X).type(dtype).to(device)


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
        for i in range(20000):
            if f'{i}' not in [sf.split('/')[-1] for sf in subfiles]:
                os.makedirs(self.config.save_dir + f'/{i}')
                path =  self.config.save_dir + f'/{i}'
                break
        return path
        
    def train_step(self, X, I, S):
        self.optimizer.zero_grad()
        logits = self.model(X, I, S)
        Xi = (X * I).sum(dim = -1)
        loss = F.cross_entropy(logits, Xi)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def to_torch(self, *arrays):
        return [np_to_torch(arr, self.config.device, dtype=torch.int) for arr in arrays]
       
    def get_batch(self, n_samples):
        X, S, I, J = self.data_generator.sample_conditional(n_samples, nomask = self.config.fullcontext)
        
        J = np.eye(X.shape[-1])[J]
        I = np.eye(X.shape[-1])[I]
        S = S + J
        return self.to_torch(X, I, S)
    
    def get_mini_batch(self, *tensors):
        N = tensors[0].size(0)
        batch_num = N // self.config.batch_size
        for i in range(batch_num):
            start = i * self.config.batch_size
            end = (i + 1) * self.config.batch_size
            yield [t[start:end] for t in tensors]

    def train_epoch(self):
        X, I, S = self.get_batch(self.config.batches_per_epoch * self.config.batch_size)
        loss = 0
        for Xb, Ib, Sb in self.get_mini_batch(X, I, S):
            loss += self.train_step(Xb, Ib, Sb) 
        return loss/self.config.batches_per_epoch
    
    @property
    def trueprob_estimate(self):
        P = self.data_generator.estimate_true_prob
        return torch.from_numpy(self.data_generator.estimate_true_prob).to(self.config.device)
 
    def estimate_tvdist_each(self):
        X, S, I, J, P = self.data_generator.test_data
        I = np.eye(X.shape[-1])[I]
        J = np.eye(X.shape[-1])[J]
        X, S, I, J = self.to_torch(X, S, I, J)
        P = torch.from_numpy(P).to(torch.float32).to(self.config.device)
        P = P[:, :4]
        tP_i_S, tP_j_S, tP_i_S_j, tP_j_S_i = P.chunk(4, dim = -1)
        tP_i_S = tP_i_S.squeeze(-1)
        tP_j_S = tP_j_S.squeeze(-1)
        tP_i_S_j = tP_i_S_j.squeeze(-1)
        tP_j_S_i = tP_j_S_i.squeeze(-1)
        
        P_i_S, P_j_S, P_i_S_j, P_j_S_i, lp_i_S, lp_j_S, lp_i_Sj, lp_j_Si = self.model.evaluate_batched(X, (S, I, J), 'probability_ij_S', self.config.batch_size)
        
        tvdist = (tP_i_S - P_i_S).abs() + (tP_j_S - P_j_S).abs() + (tP_i_S_j - P_i_S_j).abs() + (tP_j_S_i - P_j_S_i).abs()
        
        tvdist = tvdist / 4
        
        logtvdist = (torch.log(tP_i_S) -  lp_i_S).abs() \
            + (torch.log(tP_j_S) - lp_j_S).abs() \
            + (torch.log(tP_i_S_j) - lp_i_Sj).abs() \
            + (torch.log(tP_j_S_i) - lp_j_Si).abs()
        
        logtvdist = logtvdist / 4

        return tvdist, logtvdist
    
    # def estimate_tvdist_each1(self):
    #     X, S, I, J, P = self.data_generator.test_data
    #     I = np.eye(X.shape[-1])[I]
    #     J = np.eye(X.shape[-1])[J]
    #     S = S + J
    #     X, S, I = self.to_torch(X, S, I)
    #     tP = torch.from_numpy(P[:, 2]).to(torch.float32).to(self.config.device)
    #     P_iS = self.model.evaluate_batched(X, (I, S), 'probability_i_S', self.config.batch_size)[0]
        
    #     tvdist = (tP - P_iS).abs()
    #     return tvdist
        
    def estimate_tvdist_nocontext(self):
        X, S, I, J, P = self.data_generator.test_data
        I = np.eye(X.shape[-1])[I]
        X, I = self.to_torch(X, I)
        S = 1 - I
        tP = torch.from_numpy(P).to(torch.float32).to(self.config.device)[:, 4]
        P, lP = self.model.evaluate_batched(X, (I, S), 'probability_i_S', self.config.batch_size)
        tvdist = (tP - P).abs().mean().item()
        logtvdist = (torch.log(tP) - lP).abs().mean().item()
        return tvdist, logtvdist
    
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
    
    def spearman_corr(self, x, y):
        x_rank = torch.argsort(torch.argsort(x)).float()
        y_rank = torch.argsort(torch.argsort(y)).float()
        x_mean = x_rank.mean()
        y_mean = y_rank.mean()
        covariane = torch.mean((x_rank - x_mean) * (y_rank - y_mean)).item()
        x_std = x_rank.std().item() + 1e-6
        y_std = y_rank.std().item() + 1e-6
        return covariane / (x_std * y_std)
    
    def pearson_corr(self, x, y):
        # Calculate means
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        # Calculate numerator and denominator for Pearson correlation
        covariance = torch.mean((x - mean_x) * (y - mean_y)).item()
        x_std = x.std().item() + 1e-6
        y_std = y.std().item() + 1e-6
        return covariance / (x_std * y_std)
      
    
    def flatten_metrics(self, metrics):
        metrics_flat = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    metrics_flat[k + '_' + kk] = vv
            else:
                metrics_flat[k] = v
        return metrics_flat
    
    def get_metrics(self):
        metrics = {}
        X, S, I, J, P = self.data_generator.test_data
        I = np.eye(X.shape[-1])[I]
        J = np.eye(X.shape[-1])[J]
        X, S, I, J = self.to_torch(X, S, I, J)
        if self.config.fullcontext:
            ## p(x_i | x_{-i})
            metrics['tvdist'], metrics['logtvdist'] = self.estimate_tvdist_nocontext()
            if self.config.eval.consistency.path:
                metrics['path_ct_std'], metrics['path_ct_cv'] = self.model.evaluate_batched(X, [], 'path_consistency', self.config.batch_size)
        else:
            tvdist, logtvdist = self.estimate_tvdist_each()
            metrics['tvdist'] = tvdist.mean().item()       
            metrics['logtvdist'] = logtvdist.mean().item()         

            if self.config.eval.consistency.autoregressive:
                metrics['auto_ct_std'], metrics['auto_ct_cv'] = self.model.evaluate_batched(X, [], 'autoregressive_consistency', self.config.batch_size)
                
            if self.config.eval.consistency.swap:
                pdiffl1,  pdifflog = self.model.evaluate_batched(X, [S, I, J], 'swap_consistency', self.config.batch_size)
                metrics['swap_ct'] = dict(l1 = pdiffl1.mean().item(), log = pdifflog.mean().item())
                ## correlation between tvdist and swap_ct
                metrics['correlations_swapct_tvdist'] = dict(l1 = self.pearson_corr(tvdist, pdiffl1),
                                                            log = self.pearson_corr(logtvdist, pdifflog))
            if self.config.eval.consistency.path:
                metrics['path_ct_std'], metrics['path_ct_cv'] = self.model.evaluate_batched(X, [], 'path_consistency', self.config.batch_size)

        return metrics


    def train_eval(self, results = None, results_path = None):
        np.random.seed(135)
        num_batches = self.config.batches_per_epoch
        last_log = 0
        last_save = 0
        step = 0
        a = time.time()
        with torch.no_grad():
            metrics = self.get_metrics()
            metrics = self.flatten_metrics(metrics)
            b = time.time()
            print(f'Get metrics time: {b-a}')
            if self.config.print_log:
                print(f'Step {step}, TV Distance: {metrics['tvdist']}')
            results.append({'datapath':self.data_generator.path, 
                                    'batch_size': self.config.batch_size, 
                                    'context_dim':self.model.config.context_dim,
                                    'output_nlayers': self.model.config.output.n_layers,
                                    'output_hidden_dim': self.model.config.output.hidden_dim,
                                    'n_features':self.model.config.n_features,
                                    'step': step,'loss':0,
                                    **metrics}) if results is not None else None

        for i in range(self.config.n_steps // num_batches):
            
            loss = self.train_epoch()
            step += num_batches
            self.scheduler.step(step)
            if (step <= 500 and step % 100 == 0) or (step - last_log >= self.config.log_interval):
                b = time.time()
                print(f'Training Time: {b-a}')
                last_log = step
                with torch.no_grad():
                    a = time.time()
                    metrics = self.get_metrics()
                    metrics = self.flatten_metrics(metrics)
                    b = time.time()
                    print(f'Get metrics time: {b-a}')
                    if self.config.print_log:
                        print(f'Step {step}, Loss: {loss}', f'TV Distance: {metrics['tvdist']}, Log TV Distance: {metrics["logtvdist"]}')

                    results.append({'datapath':self.data_generator.path, 
                                    'batch_size': self.config.batch_size, 
                                    'context_dim':self.model.config.context_dim,
                                    'output_nlayers': self.model.config.output.n_layers,
                                    'output_hidden_dim': self.model.config.output.hidden_dim,
                                    'n_features':self.model.config.n_features,
                                    'step': step,'loss':loss,
                                    **metrics}) if results is not None else None
                a = time.time()  
                    
            if step - last_save >= self.config.save_interval:
                last_save = step
                torch.save(self.model.state_dict(), self.path + f'/model_{step}.pth')
                pd.DataFrame(results).to_csv(results_path, index=False)
            
        
def get_model_data_params(row, model, data_generator):
    params = om.create(row)    
    params = om.merge(params, data_generator.sampler.custom_config)
    params = om.merge(params, om.create({'numparams': sum(model.parameter_count.values())}))
    return om.to_container(params, resolve = True)


if __name__ == '__main__':
    
    
    parser = ArgumentParser()
    parser.add_argument('--config', type = str, default='config.yaml')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--multiple', action='store_true', default=False)
    parser.add_argument('--dryrun', action='store_true', default=False)
    parser.add_argument('--configs', type = str, default='configs_all.csv')
    parser.add_argument('--datas', type = str, default='datas.csv')
    parser.add_argument('--results', type = str, default='results_all.csv')
    parser.add_argument('--results_temp', type = str, default='results_temp_25.csv')
    parser.add_argument('--save_dir', type = str, default='checkpoints')
    parser.add_argument('--nlayers',type=int, default = 0)
    parser.add_argument('--context_dim', type=int, default = 128)
    parser.add_argument('--hidden_dim', type=int, default= 32)
    parser.add_argument('--datapath', type=str, default='all')
    parser.add_argument('--nfeatures', type=int, default=10)
    parser.add_argument('--batchsize', type=int, default=128)
    args = parser.parse_args()
    config = om.load(args.config)
    
    if args.dryrun:
        model = Model(config.model).to(config.train.device)
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

        ## run for multiple datasets and configurations
        configs = pd.read_csv(args.configs)
        columns = ['n_features','datapath','embedding_dim','aggregator','n_layers','n_heads','i_in_context']
        
        if os.path.exists(args.results):
            # Load previous results
            results_pd = pd.read_csv(args.results)
            results = results_pd.to_dict('records')

            ## get sets of configurations that have already been run
            results_f = results_pd[columns]
            results_f = results_f.drop_duplicates()
            results_f = results_f.to_dict('records')
            

        else:
            results = []
            results_f = [{}]

        

        # configs = configs[['n_features', 'embedding_dim', 'context_aggregator', 'n_layers']]
        # configs = configs.drop_duplicates()
        data_generators = dict()
         
        for i, row in tqdm(configs.iterrows()):
            config_ = om.structured(config)
            
            config_.data.load = True
            config_.data.save = False

            config_.data.path = row['datapath']
            config_.model.n_features = config_.data.n_features = int(row['n_features'])
            config_.model.context_dim = int(row['context_dim'])
            config_.model.output.n_layers = row['n_layers']
            config_.model.output.hidden_dim = int(row['hidden_dim'])

            config_.train.save_dir = args.save_dir

            torch.manual_seed(135)
            torch.mps.manual_seed(135)
            torch.cuda.manual_seed(135)
            model = Model(config_.model).to(config_.train.device)
            data_generator = data_generators.get(config_.data, DataGenerator(config_.data))
            data_generators[config_.data] = data_generator
            if config.train.debug.print_log:
                print(f"""Training on {data_generator.sampler.num_samples} 
                    samples for {config_.train.n_steps} steps 
                    with a batch size of {config_.train.batch_size} 
                    with context dim {config_.model.context_dim}, 
                    hidden dim {config_.model.output.hidden_dim},
                    n_layers {config_.model.output.n_layers},
                    n_features {config_.model.n_features} """)
            trainer = Trainer(model, data_generator, config_.train, args)

            trainer.train_eval(results)
            pd.DataFrame(results).to_csv(args.results, index = False)

    else:
        ## run for a single configuration
        torch_rand(135)
        model = Model(config.model).to(config.train.device)
        # print('Model parameters:', sum(model.parameter_count.values()))
        data_generator = DataGenerator(config.data)
        # print(f"""Training on {data_generator.sampler.num_samples} samples
        #        for {config.train.n_steps} steps 
        #        on {config.data.path},
        #        n_features {config.data.n_features}
        #        and context dim {config.model.context_dim}, 
        #        and hidden dim {config.model.output.hidden_dim},
        #        and n_layers {config.model.output.n_layers},
        #        with a batch size of {config.train.batch_size} """)
        trainer = Trainer(model, data_generator, config.train, args)
        results = []
        trainer.train_eval(results, args.results_temp)
        pd.DataFrame(results).to_csv(args.results_temp, index=False)

