import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf as om
from config import ScoreFunctionType, ContextAggregator, DataType
import numpy as np
from torch.nn import functional as F
from torch.distributions import Normal

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.xi_featurizer_weight = nn.Parameter(torch.empty(config.n_features, config.embedding_dim))
        self.xi_featurizer_bias = nn.Parameter(torch.empty(config.n_features, config.embedding_dim))
        self.xs_featurizer_weight = nn.Parameter(torch.empty(config.n_features, config.embedding_dim))
        self.xs_featurizer_bias = nn.Parameter(torch.empty(config.n_features, config.embedding_dim))
        # self.context_aggregator_weight = nn.Parameter(torch.empty(config.n_features, config.embedding_dim))
        # self.context_aggregator_bias = nn.Parameter(torch.empty(config.embedding_dim))
        self.xs_featurizer_ibias = nn.Parameter(torch.empty(config.n_features, config.embedding_dim))
        
    def _reset_parameters(self):
        nn.init.xavier_normal_(self.xi_featurizer_weight)
        nn.init.xavier_normal_(self.xi_featurizer_bias)
        nn.init.xavier_normal_(self.xs_featurizer_weight)
        nn.init.xavier_normal_(self.xs_featurizer_bias)

    def map_xi(self, Xi, I):    
        return Xi * torch.matmul( I, self.xi_featurizer_weight) + torch.matmul(I, self.xi_featurizer_bias)
    
    def map_xs(self, X, I, S, attn_mask = None):
        FX = X * self.xs_featurizer_weight.unsqueeze(0) + self.xs_featurizer_bias.unsqueeze(0)
    
        if self.config.context_aggregator == ContextAggregator.avg:
            return (FX * S.unsqueeze(2)).mean(dim = 1) + torch.matmul(I, self.xs_featurizer_ibias)
        elif self.config.context_aggregator == ContextAggregator.sum:
            return (FX * S.unsqueeze(2)).sum(dim = 1) + torch.matmul(I, self.xs_featurizer_ibias)
        # elif self.config.context_aggregator == ContextAggregator.linear:
        #     return (FX * S.unsqueeze(2) * self.context_aggregator.unsqueeze(0)) + self.context_aggregator_bias.unsqueeze(0) + torch.matmul(I, self.i_bias)
        elif self.config.context_aggregator == ContextAggregator.attention:
            attention_weights = torch.baddbmm(attn_mask, FX/np.sqrt(self.config.embedding_dim), FX.transpose(1, 2))
            attention_weights = F.softmax(attention_weights, dim = -1)
            FX = torch.bmm(attention_weights, FX) + torch.matmul(I, self.xs_featurizer_bias)
            return FX.mean(dim  = 1)
        
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
    
    def logprob(self, X, Xi, I, S, attn_mask = None):
        Fxi = self.map_xi(Xi, I)
        Fxs = self.map_xs(X, I, S, attn_mask)
        ulogp = torch.log(self.score(Fxi, Fxs))
        logz = self.logz(I, Fxs)
        return ulogp - logz

class ModelBinary(nn.Module):
    def __init__(self, config):
        super(ModelBinary, self).__init__()
        self.config = config