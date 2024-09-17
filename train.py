
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
    def __init__(self, model, config):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
        self.config = config
        

    def train_step(self, X, Xi, I, S, attn_mask):
        self.optimizer.zero_grad()
        loss = - self.model.logprob(X, Xi, I, S, attn_mask).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    

    def train(self, data_generator, n_steps, n_samples):
        X, Xi, I, S = data_generator.sample_conditional(n_samples)
        X = np_to_device(X, self.config.device)
        Xi = np_to_device(Xi, self.config.device)
        I = np_to_device(I, self.config.device)
        S = np_to_device(S, self.config.device)

        attn_mask = get_attn_mask(S)
        for i in range(n_steps):
            loss = self.train_step(X, Xi, I, S, attn_mask)
            if i % 100 == 0:
                print(f'Step {i}: Loss = {loss}')
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
    trainer = Trainer(model, config.train)

    trainer.train(data_generator, config.train.n_steps, config.train.n_samples)