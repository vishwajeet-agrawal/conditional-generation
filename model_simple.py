import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf as om
from config import ScoreFunctionType, ContextAggregatorType, DataType
import numpy as np
from torch.nn import functional as F
from torch.distributions import Normal
from util import stable_softmax
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=config.embedding_dim, nhead=config.attention_nheads), num_layers=config.attention_nlayers)
    def forward(self, X, attn_mask):
        return self.transformer(X, attn_mask)

class ModelSimple(nn.Module):
    def __init__(self, config, args):
        super(ModelSimple, self).__init__()
        self.config = config
        self.args = args
        self.embedding_dim = config.embedding_dim
        self.n_features = config.n_features
        self.xi_featurizer_weight = nn.Parameter(torch.empty(config.n_features, config.embedding_dim))
        self.xi_featurizer_bias = nn.Parameter(torch.empty(config.n_features, config.embedding_dim))
        self.xs_featurizer_weight = nn.Parameter(torch.empty(config.n_features, config.embedding_dim))
        self.xs_featurizer_bias = nn.Parameter(torch.empty(config.n_features, config.embedding_dim))
        self.xs_featurizer_ibias = nn.Parameter(torch.empty(config.n_features, config.embedding_dim))
        self._reset_parameters()
        if self.config.context_aggregator == ContextAggregatorType.transformer:
            self.transformer = TransformerEncoder(config)

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.xi_featurizer_weight)
        nn.init.xavier_normal_(self.xi_featurizer_bias)
        nn.init.xavier_normal_(self.xs_featurizer_weight)
        nn.init.xavier_normal_(self.xs_featurizer_bias)
        nn.init.xavier_normal_(self.xs_featurizer_ibias)

    def map_xi(self, Xi, I):    
        return Xi.unsqueeze(1) * torch.matmul( I, self.xi_featurizer_weight) + torch.matmul(I, self.xi_featurizer_bias)

    def map_xs(self, X, I, S):
        FX = X.unsqueeze(2) * self.xs_featurizer_weight.unsqueeze(0) + self.xs_featurizer_bias.unsqueeze(0) + torch.matmul(I, self.xs_featurizer_bias).unsqueeze(1)

        if self.config.context_aggregator == ContextAggregatorType.avg:
            return (FX * S.unsqueeze(2)).mean(dim = 1)
        elif self.config.context_aggregator == ContextAggregatorType.sum:
            return (FX * S.unsqueeze(2)).sum(dim = 1)
        elif self.config.context_aggregator == ContextAggregatorType.attention:
            attn_mask = (1 - (S.unsqueeze(1) * S.unsqueeze(2))).to(torch.bool)
            attn_mask = torch.zeros_like(attn_mask, dtype=torch.float).masked_fill_(attn_mask, float("-inf"))
            attention_weights = torch.baddbmm(attn_mask, FX/np.sqrt(self.config.embedding_dim), FX.transpose(1, 2))
            attention_weights = stable_softmax(attention_weights)
            return torch.bmm(attention_weights, FX).sum(dim = 1)/ S.sum(dim = 1).unsqueeze(1)
        elif self.config.context_aggregator == ContextAggregatorType.transformer:
            attn_mask = (1 - (S.unsqueeze(1) * S.unsqueeze(2))).to(torch.bool)
            FX = self.transformer(FX, attn_mask)
            return (FX * S.unsqueeze(2)).sum(dim = 1)/ S.sum(dim = 1).unsqueeze(1)
        else:
            raise ValueError(f'Unknown context aggregator: {self.config.context_aggregator}') 
    
    def accuracy_binary(self, X, Xi, I, S):
        Fxs = self.map_xs(X, I, S)
        Xi0 = torch.zeros_like(Xi)
        Xi1 = torch.ones_like(Xi)
        Fxi0 = self.map_xi(Xi0, I)
        Fxi1 = self.map_xi(Xi1, I)
        s0 = self.score(Fxi0, Fxs)
        s1 = self.score(Fxi1, Fxs)
        return ((s0 > s1) == (Xi == 0)).float().mean()
    
    def accuracy(self, X, Xi, I, S):
        if self.config.data_type == DataType.binary:
            return self.accuracy_binary(X, Xi, I, S)
        elif self.config.data_type == DataType.continuous:
            raise ValueError(f'Accuracy not implemented for continuous data')

    def score(self, Fxi, Fxs):
        if self.config.score_function == ScoreFunctionType.exp:
            return torch.exp((Fxi * Fxs).sum(dim = -1))
        else:
            raise ValueError(f'Unknown score function: {self.config.score_function}')
    
    def logz(self, I, Fxs):
        N = I.size(0)
        if self.config.data_type == DataType.binary:
            Xi0 = torch.zeros(N, device = I.device) 
            Xi1 = torch.ones(N, device = I.device)
            Fxi0 = self.map_xi(Xi0, I)
            Fxi1 = self.map_xi(Xi1, I)
            s0 = self.score(Fxi0, Fxs)
            s1 = self.score(Fxi1, Fxs)
            logz = torch.log(s0 + s1)
        elif self.config.data_type == DataType.continuous:
            normal_dist = Normal(self.config.marginal.mean, self.config.marginal.std)
            Xz = np.random.normal(self.config.marginal.mean, self.config.marginal.std, size = (N, self.config.marginal.n_points))
            pdf_inv = torch.exp(-normal_dist.log_prob(Xz))
            logz = torch.log((pdf_inv.unsqueeze(1) * torch.exp(Xz.unsqueeze(1) * torch.matmul(I, self.xi_featurizer_weight).unsqueeze(2) + torch.matmul(I, self.xi_featurizer_bias).unsqueeze(2))).sum(dim = 1))
        else:
            raise ValueError(f'Unknown data type: {self.config.data_type}')
        return logz
    
    def logprob(self, X, Xi, I, S):

        Fxi = self.map_xi(Xi, I)
        Fxs = self.map_xs(X, I, S)
        ulogp = torch.log(self.score(Fxi, Fxs))
        logz = self.logz(I, Fxs)
        
        return ulogp - logz
   

    def estimate_prob(self, X, Xi, I, S, batch_size):
        N = X.size(0)
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        if self.config.data_type == DataType.binary:
            N = X.size(0)
            s0s = None
            s1s = None
            for b in range(num_batches):
                start = b * batch_size
                end = (b + 1) * batch_size
                Fxs = self.map_xs(X[start:end], I[start:end], S[start:end])
                effective_batch_size = Fxs.size(0)
                Xi0 = torch.zeros(effective_batch_size, device = I.device) 
                Xi1 = torch.ones(effective_batch_size, device = I.device)
                Fxi0 = self.map_xi(Xi0, I[start:end])
                Fxi1 = self.map_xi(Xi1, I[start:end])
                s0 = self.score(Fxi0, Fxs)
                s1 = self.score(Fxi1, Fxs)
                if b == 0:
                    s0s = s0
                    s1s = s1
                else:
                    s0s = torch.cat([s0s, s0], dim = 0)
                    s1s = torch.cat([s1s, s1], dim = 0)
            return (s0s * (Xi == 0).float() + s1s * (Xi == 1).float())/(s0s + s1s)
        elif self.config.data_type == DataType.continuous:
            raise NotImplementedError(f'Probability estimation not implemented for continuous data')
        else:
            raise ValueError(f'Unknown data type: {self.config.data_type}')
                