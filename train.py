
import torch
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

    def train(self):
        X, Xi, I, S = self.get_batch(self.config.load_factor * self.config.batch_size)
        for i in range(self.config.n_steps//self.config.load_factor):
            mini_batch = self.get_mini_batch(X, Xi, I, S)
            for j, (Xb, Xib, Ib, Sb) in enumerate(mini_batch):
                loss = self.train_step(Xb, Xib, Ib, Sb)
                if self.args.debug and (i*self.config.load_factor + j) % self.config.debug.print_interval == 0:
                    print(f'Step {i*self.config.load_factor + j}: Loss = {loss}')
        torch.save(self.model.state_dict(), self.config.model_path)
        
if __name__ == '__main__':
    from model import Model
    from data import DataGenerator
    from argparse import ArgumentParser
    from omegaconf import OmegaConf as om

    parser = ArgumentParser()
    parser.add_argument('--config', type = str, default='config.yaml')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    config = om.load(args.config)
    model = Model(config.model, args).to(config.train.device)
    data_generator = DataGenerator(config.data)
    trainer = Trainer(model, data_generator, config.train, args)

    trainer.train()

