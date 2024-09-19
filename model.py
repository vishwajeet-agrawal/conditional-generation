import torch
import torch.nn as nn
import yaml
from abc import abstractmethod
from omegaconf import OmegaConf as om
from config import ScoreFunctionType, ContextAggregator, DataType
import numpy as np
from torch.nn import functional as F
from torch.distributions import Normal
from util import stable_softmax
import math
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=config.embedding_dim, nhead=config.attention_nheads), num_layers=config.attention_nlayers)
    def forward(self, X, attn_mask):
        return self.transformer(X, attn_mask)

class FFN(nn.Module):
    # Also implement layer normalization
    def __init__(self, embedding_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    def forward(self, X):
        return X + self.mlp(X)
      
class ContextAggregator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.n_features = config.n_features
        self.n_vocab = config.n_vocab
        self.n_layers = config.n_layers

        self.positional_embeddings = nn.Parameter(torch.empty(self.n_features, self.embedding_dim))
        if not config.tie_embeddings:
            self.embeddings = nn.ModuleList([nn.Embedding(self.n_vocab + 1, self.embedding_dim) for _ in range(self.n_features)])
        else:
            self.embeddings = nn.Embedding(self.n_vocab + 1, self.embedding_dim)
            self.embeddings = nn.ModuleList([self.embeddings for _ in range(self.n_features)])
        
        if config.tie_ffn:
            self.ffns = FFN(self.embedding_dim)
            self.ffns = nn.ModuleList([self.ffns for _ in range(config.n_layers)])
        else:
            self.ffns = nn.ModuleList([FFN(self.embedding_dim) for _ in range(config.n_layers)])

    def reset_parameters(self):
        nn.init.xavier_normal_(self.positional_embeddings)
        
    def token_to_embeddings(self, X, S):
        Xm = X * S + (1 - S) * self.n_vocab
        embeddings = [self.embeddings[i](Xm[:, i]) for i in range(self.n_features)]
        embeddings = torch.stack(embeddings, dim = 1)
        return embeddings + self.positional_embeddings.unsqueeze(0)
        
    def forward(self, X, I, S):
        x = self.token_to_embeddings(X, S)
        for i in range(self.n_layers):
            x = self.aggregate(x, i, I + S)
            x = self.ffns[i](x)
        return torch.bmm(x.transpose(1, 2), I.unsqueeze(-1)).squeeze(-1)
    
    @abstractmethod
    def aggregate(self, x, i, mask):
        return ...
   
    
class MLPAggregator(ContextAggregator):
    def __init__(self, config):
        super(MLPAggregator, self).__init__(config)
        self.tie_aggregate = config.tie_aggregate
        if self.tie_aggregate:
            self.aggregator = nn.Parameter(nn.Empty(self.n_features, self.n_features))
            self.aggregator = nn.ModuleList([self.aggregator for _ in range(config.n_layers)])
        else:
            self.aggregator = nn.ModuleList([nn.Parameter(nn.Empty(self.n_features, self.n_features))for _ in range(config.n_layers)])
    def reset_parameters(self):
        if self.tie_aggregate:
            nn.init.xavier_normal_(self.aggregator[0])
        else:
            [nn.init.xavier_normal_(self.aggregator[i]) for i in range(self.n_layers)]
        
    def aggregate(self, x, i, m):
        return torch.baddbmm(x, self.aggregator[i].unsqueeze(0) * m.unsqueeze(1), x)

class TransformerAggregator(ContextAggregator):
    def __init__(self, config):
        super(TransformerAggregator, self).__init__(config)
        self.n_heads = config.n_heads
        if config.tie_qkv:
            qkv_proj = nn.Parameter(nn.Empty(3 * self.embedding_dim, self.embedding_dim))
            out_proj = nn.Parameter(nn.Empty(self.embedding_dim, self.embedding_dim))
            nn.init.xavier_normal_(qkv_proj)
            nn.init.xavier_normal_(out_proj)
            self.qkv_proj = nn.ModuleList([qkv_proj for _ in range(self.n_layers)])
            self.out_proj = nn.ModuleList([out_proj for _ in range(self.n_layers)])
        else:
            self.qkv_proj = nn.ModuleList([nn.Parameter(nn.Empty(3 * self.embedding_dim, self.embedding_dim))])
            self.out_proj = nn.ModuleList([nn.Parameter(nn.Empty(self.embedding_dim, self.embedding_dim))])
            [nn.init.xavier_normal_(self.qkv_proj[i]) for i in range(self.n_layers)]
            [nn.init.xavier_normal_(self.out_proj[i]) for i in range(self.n_layers)]

    def get_attn_mask(self, S):
        attn_mask = (1 - (S.unsqueeze(1) * S.unsqueeze(2))).to(torch.bool)
        attn_mask = torch.zeros_like(attn_mask, dtype=torch.float).masked_fill_(attn_mask, float("-inf"))
        return attn_mask

    def aggregate(self, x, i, m):
        attn_mask = self.get_attn_mask(m)
        B, T, C = x.size()
        wq, wk, wv = torch.split(self.qkv_proj[i], 3, dim = 0)
        q = F.linear(x, wq, 0)
        k = F.linear(x, wk, 0)
        v = F.linear(x, wv, 0)
        d_dim = self.embedding_dim//self.n_heads
        q = q.view(B, T, self.n_heads, d_dim).transpose(1, 2).view(B * self.n_heads, T, d_dim)
        k = k.view(B, T, self.n_heads, d_dim).transpose(1, 2).view(B * self.n_heads, T, d_dim)
        v = v.view(B, T, self.n_heads, d_dim).transpose(1, 2).view(B * self.n_heads, T, d_dim)

        q_scaled = q * math.sqrt(1.0 / float(q.size(-1)))
        attn_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        attn_weights = F.softmax(attn_weights, dim = -1)
        v = torch.bmm(attn_weights, v)
        v = v.view(B, self.n_heads, T, d_dim).transpose(1, 2).reshape(B, T, C)
        return self.out_proj[i](v) + x


class Model(nn.Module):
    def __init__(self, config, args):
        super(Model, self).__init__()
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
        if self.config.context_aggregator == ContextAggregator.transformer:
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

        if self.config.context_aggregator == ContextAggregator.avg:
            return (FX * S.unsqueeze(2)).mean(dim = 1)
        elif self.config.context_aggregator == ContextAggregator.sum:
            return (FX * S.unsqueeze(2)).sum(dim = 1)
        elif self.config.context_aggregator == ContextAggregator.attention:
            attn_mask = (1 - (S.unsqueeze(1) * S.unsqueeze(2))).to(torch.bool)
            attn_mask = torch.zeros_like(attn_mask, dtype=torch.float).masked_fill_(attn_mask, float("-inf"))
            attention_weights = torch.baddbmm(attn_mask, FX/np.sqrt(self.config.embedding_dim), FX.transpose(1, 2))
            attention_weights = stable_softmax(attention_weights)
            return torch.bmm(attention_weights, FX).sum(dim = 1)/ S.sum(dim = 1).unsqueeze(1)
        elif self.config.context_aggregator == ContextAggregator.transformer:
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
                