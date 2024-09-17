
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

def np_to_device(X, device):
    device = torch.device(device)
    return torch.from_numpy(X).to(device)

class Trainer:
    def __init__(self, model, data_generator, config):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
        self.config = config
        self.data_generator = data_generator

    def train_step(self, X, Xi, I, S):
        self.optimizer.zero_grad()
        loss = - self.model.logprob(X, Xi, I, S).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def get_batch(self, n_samples):
        X, Xi, I, S = self.data_generator.sample_conditional(n_samples)
        X = np_to_device(X, self.config.device)
        Xi = np_to_device(Xi, self.config.device)
        I = np_to_device(I, self.config.device)
        S = np_to_device(S, self.config.device)
        return X, Xi, I, S
    def get_mini_batch(self, X, Xi, I, S):
        N = X.size(0)
        batch_num = N // self.config.batch_size
        for i in range(batch_num):
            start = i * self.config.batch_size
            end = (i + 1) * self.config.batch_size
            yield X[start:end], Xi[start:end], I[start:end], S[start:end]

    def train(self):
        X, Xi, I, S = self.get_batch(config.n_samples)
        for i in range(config.n_epochs):
            loss = 0
            for mini_batch in self.get_mini_batch(X, Xi, I, S):
                Xb, Xib, Ib, Sb = mini_batch
                mb_loss = self.train_step(Xb, Xib, Ib, Sb)
                loss += mb_loss
            loss = loss / (config.n_samples // self.config.batch_size)
            if i % 2 == 0:
                print(f'Epoch {i}: Loss = {loss}')
        torch.save(self.model.state_dict(), self.config.model_path)
        
if __name__ == '__main__':
    from model import Model
    from data import DataGenerator
    from argparse import ArgumentParser
    from omegaconf import OmegaConf as om

    parser = ArgumentParser()
    parser.add_argument('--config', type = str, default='config.yaml')
    args = parser.parse_args()
    config = om.load(args.config)
    model = Model(config.model)
    data_generator = DataGenerator(config.data)
    trainer = Trainer(model, data_generator, config.train)

    trainer.train()
