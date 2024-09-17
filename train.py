
import torch
import pandas as pd
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
    

class Trainer:
    def __init__(self, model, data_generator, config, args):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
        self.config = config
        self.data_generator = data_generator
        self.args = args

    def train_step(self, X, Xi, I, S):
        self.optimizer.zero_grad()
        loss = - self.model.logprob(X, Xi, I, S).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def get_batch(self, n_samples):
        X, Xi, I, S = self.data_generator.sample_conditional(n_samples)
        X = np_to_torch(X, self.config.device)
        Xi = np_to_torch(Xi, self.config.device)
        I = np_to_torch(I, self.config.device)
        S = np_to_torch(S, self.config.device)
        return X, Xi, I, S
    
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
    
    def train_eval(self, results = None):
        
        X, Xi, I, S = self.get_batch(self.config.eval_size) # Test Data
        P = torch.from_numpy(self.data_generator.sampler.estimate_conditional_prob(X, I, S)).to(self.config.device)

        num_batches = self.config.batches_per_epoch
        last_log = 0
        step = 0
        for i in range(self.config.n_steps // num_batches):
            loss = self.train_epoch()
            step += num_batches
            if step - last_log >= self.config.log_interval:
                last_log = step
                with torch.no_grad():
                    Pm = self.model.estimate_prob(X, Xi, I, S, self.config.batch_size)
                    tvdist = (Pm - P).abs().mean()
                    results.append({'datapath':self.data_generator.config.source.path, 'batch_size': self.config.batch_size, 'embedding_dim':self.model.config.embedding_dim, 'n_features':self.model.config.n_features,'caggregate':self.model.config.context_aggregator, 'step': step,'loss':loss, 'tv_dist': tvdist.item()}) if results is not None else None
        torch.save(self.model.state_dict(), self.config.model_path)


if __name__ == '__main__':
    from model import Model
    from data import DataGenerator
    from argparse import ArgumentParser
    from omegaconf import OmegaConf as om
    import time
    parser = ArgumentParser()
    parser.add_argument('--config', type = str, default='config.yaml')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--multiple', action='store_true', default=False)
    parser.add_argument('--dryrun', action='store_true', default=False)
    parser.add_argument('--configs', type = str, default='configs.csv')
    args = parser.parse_args()
    config = om.load(args.config)
    if args.dryrun:
        model = Model(config.model, args).to(config.train.device)
        data_generator = DataGenerator(config.data)
        print(f'Training on {data_generator.sampler.num_samples} samples for {config.train.n_steps} steps with a batch size of {config.train.batch_size} with config {config.model}')
        trainer = Trainer(model, data_generator, config.train, args)
        X, Xi, I, S = trainer.get_batch(trainer.config.eval_size) # Test Data
        a = time.time()        
        P = torch.from_numpy(trainer.data_generator.sampler.estimate_conditional_prob(X, I, S)).to(trainer.config.device)
        b = time.time() - a
        print(f'Time in estimating conditional probability for {trainer.config.eval_size} samples: {b} seconds')

        exit()
    if args.multiple:
        configs = pd.read_csv(args.configs)
        results = []
        for i, row in configs.iterrows():
            config_ = om.structured(config)
            config_.data.source.path = row['path']
            config_.model.n_features = config_.data.n_features = int(row['n_features'])
            config_.model.embedding_dim = int(row['embedding_dim'])
            config_.model.context_aggregator = row['context_aggregator']
            model = Model(config_.model, args).to(config_.train.device)
            data_generator = DataGenerator(config_.data)
            print(f'Training on {data_generator.sampler.num_samples} samples for {config_.train.n_steps} steps with a batch size of {config_.train.batch_size} with emb_dim {config_.model.embedding_dim}, n_features {config_.model.n_features} and context aggregator {config_.model.context_aggregator}')
            trainer = Trainer(model, data_generator, config_.train, args)
            trainer.train_eval(results)
            pd.DataFrame(results).to_csv('results.csv')
    else:
        model = Model(config.model, args).to(config.train.device)
        data_generator = DataGenerator(config.data)
        print(f'Training on {data_generator.sampler.num_samples} samples for {config.train.n_steps} steps with a batch size of {config.train.batch_size}')
        trainer = Trainer(model, data_generator, config.train, args)
        trainer.train_eval()

