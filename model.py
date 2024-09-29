import torch
import torch.nn as nn
import yaml
from abc import abstractmethod
from omegaconf import OmegaConf as om
from config import ScoreFunctionType, ContextAggregatorType, DataType, AggregatorReduceType
import numpy as np
from torch.nn import functional as F
from torch.distributions import Normal
from util import stable_softmax
import math
from functools import cached_property


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)
        
class LayerNorm(nn.Module):
    def __init__(self, config):
        super(LayerNorm, self).__init__()
        self.normalized_shape = (config.embedding_dim,)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias)
    
class FFN(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    def forward(self, X):
        return self.mlp(X)
      
class ContextAggregator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pass_i = config.aggregator.i_in_context
        self.embedding_dim = config.embedding_dim
        self.n_features = config.n_features
        self.n_vocab = config.n_vocab
        self.n_layers = config.aggregator.n_layers
        self.dropout = Dropout(config.dropout)
        factory_kwargs = {'device': config.device, 'dtype': torch.float32}
        self.agg_norm = nn.ModuleList([LayerNorm(config) for _ in range(self.n_layers)])
        self.ffn_norm = nn.ModuleList([LayerNorm(config) for _ in range(self.n_layers)])
        self.positional_embeddings = nn.Parameter(torch.empty(self.n_features, self.embedding_dim, **factory_kwargs))
        self.tie_embeddings = config.aggregator.tie_embeddings
        if not self.tie_embeddings:
            self.embeddings = nn.ModuleList([nn.Embedding(self.n_vocab + 1, self.embedding_dim) for _ in range(self.n_features)])
        else:
            self.embeddings_ = nn.Embedding(self.n_vocab + 1, self.embedding_dim)
            self.embeddings = nn.ModuleList([self.embeddings_ for _ in range(self.n_features)])
        
        self.ffns = nn.ModuleList([FFN(self.embedding_dim) for _ in range(self.n_layers)])
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_normal_(self.positional_embeddings)
        self.register_parameter('pos_embeddings', self.positional_embeddings)
        
    def token_to_embeddings(self, X, S):
        Xm = X * S + (1 - S) * self.n_vocab
        Xm_stacked = Xm.transpose(0, 1)  # Shape: (n_features, batch_size)
        embeddings = torch.nn.parallel.parallel_apply(self.embeddings, Xm_stacked)
        # embeddings = [self.embeddings[i](Xm[:, i]) for i in range(self.n_features)]
        embeddings = torch.stack(embeddings, dim = 1)
        embeddings =  embeddings + self.positional_embeddings.unsqueeze(0)
        return embeddings
        
    def forward(self, X, I, S):
        x = self.token_to_embeddings(X, S)
        S_ = I + S if self.pass_i else S
        for i in range(self.n_layers):
            x_ = self.aggregate(x, i, S_)
            x = x + self.dropout(self.agg_norm[i](x_))
            x_ = self.ffns[i](x)
            x = x + self.dropout(self.ffn_norm[i](x_))

        return torch.bmm(x.transpose(1, 2), I.unsqueeze(-1).float()).squeeze(-1)
    
    def count_parameters(self):
        n_pos = sum(p.numel() for p in self.positional_embeddings)
        n_ffn = sum(sum(p.numel() for p in self.ffns[i].parameters()) for i in range(self.n_layers))
        if self.tie_embeddings:
            n_emb = sum(p.numel() for p in self.embeddings[0].parameters())
        else:
            n_emb = sum(sum(p.numel() for p in self.embeddings[i].parameters()) for i in range(self.n_features))
        n_norm = sum(sum(p.numel() for p in self.agg_norm[i].parameters()) for i in range(self.n_layers))
        n_norm += sum(sum(p.numel() for p in self.ffn_norm[i].parameters()) for i in range(self.n_layers))
        return {'pos_embs': n_pos, 'ffns': n_ffn, 'embeddings': n_emb, 'norms': n_norm}
        

    @abstractmethod
    def aggregate(self, x, i, m):
        return ...
   
    
class MLPAggregator(ContextAggregator):
    def __init__(self, config):
        super(MLPAggregator, self).__init__(config)
        self.tie_aggregate = config.aggregator.tie_aggregator
        self.reduce_type = config.aggregator.reduce_type
        self.learn_adjacency = config.aggregator.learn_adjacency
        self.n_heads = config.aggregator.n_heads
        factory_kwargs = {'device': config.device, 'dtype': torch.float32}
        assert self.embedding_dim % self.n_heads == 0
        if not config.aggregator.learn_adjacency:
            self.aggregator = torch.ones((self.n_layers, self.n_heads, self.n_features, self.n_features)
                                         , device = config.device)/ self.n_features
        else:    
            if self.tie_aggregate:
                self.aggregator_ = nn.Parameter(torch.empty((self.n_heads, self.n_features, self.n_features), **factory_kwargs))
                self.aggregator = [self.aggregator_ for _ in range(self.n_layers)]
                nn.init.xavier_normal_(self.aggregator[0])
            else:
                self.aggregator = [
                    nn.Parameter(torch.empty((self.n_heads, self.n_features, self.n_features), **factory_kwargs))
                    for _ in range(self.n_layers)]
                [nn.init.xavier_normal_(self.aggregator[i]) for i in range(self.n_layers)]
            [self.register_parameter(f'aggregator_{i}', self.aggregator[i]) for i in range(self.n_layers)]

    @property
    def parameter_count(self):
        params_count = ContextAggregator.count_parameters(self)
        if self.learn_adjacency:
            if self.tie_aggregate:
                params_count['agg'] = sum(p.numel() for p in self.aggregator[0])
            else:
                params_count['agg'] = sum(sum(p.numel() for p in self.aggregator[i]) for i in range(self.n_layers))
        else:
            params_count['agg'] = 0
        return params_count
  
    def aggregate(self, x, i, m):
        B, T, C = x.size()
        d_dim = self.embedding_dim//self.n_heads
        x = x.view(B, T, self.n_heads, d_dim).transpose(1, 2).reshape(B*self.n_heads, T, d_dim)
        agg = self.aggregator[i].unsqueeze(0).repeat(B, 1, 1, 1).view(B*self.n_heads, T, T)
        m_r = m.unsqueeze(1).repeat(1, self.n_heads, 1).view(B*self.n_heads, T)
        y = torch.bmm(agg * m_r.unsqueeze(1), x)
        y = y.view(B, self.n_heads, T, d_dim).transpose(1, 2).reshape(B, T, C)
        
        if self.reduce_type == AggregatorReduceType.sum:
            return y
        elif self.reduce_type == AggregatorReduceType.mean:
            
            return y / m.sum(dim = -1).unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError(f'Unknown aggregator reduce type: {self.config.aggregator_reduce}')


class TransformerAggregator(ContextAggregator):
    def __init__(self, config):
        super(TransformerAggregator, self).__init__(config)
        self.n_heads = config.aggregator.n_heads
        self.qkv_proj = nn.ModuleList([nn.Linear(self.embedding_dim, 3 * self.embedding_dim, bias = False) 
                                      for _ in range(self.n_layers)])
        self.out_proj = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim, bias = False) 
                                       for _ in range(self.n_layers)])

    def get_attn_mask(self, S):
        attn_mask = (1 - (S.unsqueeze(1) * S.unsqueeze(2))).to(torch.bool)
        attn_mask = torch.zeros_like(attn_mask, dtype=torch.float).masked_fill_(attn_mask, float("-inf"))
        return attn_mask
    
    @property
    def parameter_count(self):
        params_count = ContextAggregator.count_parameters(self)
        params_count['agg'] = sum(p.numel() for p in self.qkv_proj.parameters())
        params_count['agg'] += sum(p.numel() for p in self.out_proj.parameters())
        return params_count
    
    def aggregate(self, x, i, m):
        attn_mask = self.get_attn_mask(m)
        B, T, C = x.size()
        qkv = self.qkv_proj[i](x)
        q, k, v = torch.split(qkv, self.embedding_dim, dim = -1)
        
        d_dim = self.embedding_dim//self.n_heads
        assert d_dim * self.n_heads == self.embedding_dim
        q = q.view(B, T, self.n_heads, d_dim).transpose(1, 2).reshape(B * self.n_heads, T, d_dim)
        k = k.view(B, T, self.n_heads, d_dim).transpose(1, 2).reshape(B * self.n_heads, T, d_dim)
        v = v.view(B, T, self.n_heads, d_dim).transpose(1, 2).reshape(B * self.n_heads, T, d_dim)

        q_scaled = q * math.sqrt(1.0 / float(q.size(-1)))
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1).view(B * self.n_heads, T, T)
        attn_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        attn_weights = stable_softmax(attn_weights)
        v = torch.bmm(attn_weights, v)
        v = v.view(B, self.n_heads, T, d_dim).transpose(1, 2).reshape(B, T, C)
        return self.out_proj[i](v)

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.embeddings = nn.Parameter(torch.empty((config.n_features, config.embedding_dim, config.n_vocab)))
        nn.init.xavier_normal_(self.embeddings)
        # self.embeddings = nn.ModuleList([nn.Embedding(config.n_vocab, config.embedding_dim) for i in range(config.embedding_dim)])
        # self.embedding = nn.Parameter(torch.empty((config.n_features, config.n_vocab, config.embedding_dim)))
        # self.embedding = nn.Embedding(config.n_vocab , config.embedding_dim)
        self.context_agg = MLPAggregator(config) if config.aggregator.type == ContextAggregatorType.mlp\
                else TransformerAggregator(config) if config.aggregator.type == ContextAggregatorType.transformer else None
          
    def forward(self, X, I, S):
        Fxs = self.context_agg(X, I, S)
        # Xr = (X * I).sum(dim=-1)
        # Xr_oh = F.one_hot(Xr.long(), self.config.n_vocab)
        
        embedding = torch.einsum('ij,jkl->ikl', I.float(), self.embeddings)
        # embedding = torch.bmm(I.unsqueeze(1), self.embeddings.unsqueeze(0).repeat(I.size(0), 1, 1, 1)).squeeze(1)
        # Fxi = torch.bmm(Xr_oh.unsqueeze(1).float(), embedding).squeeze(1)
        # print(Fxs.shape, embedding.shape)
        logits = torch.bmm(Fxs.unsqueeze(1), embedding).squeeze(1) 
        # logits = F.linear(Fxs, embedding.transpose(0, 1), None)
        # logits = (Fxs * Fxi).sum(dim = -1)
        # print(logits.shape)
        return logits
    
    def predict(self, X, I, S):
        return self.forward(X, I, S).argmax(dim = 1)
    
    def accuracy(self, X, Xi, I, S):
        Xip = self.predict(X, I, S)
        return (Xip == Xi).float().mean()
    
    def estimate_prob(self, X, Xi, I, S, batch_size = None):
        N = X.size(0)
        batch_size = batch_size if batch_size is not None else N
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        P = None
        for b in range(num_batches):
            start = b * batch_size
            end = (b + 1) * batch_size
            logits = self.forward(X[start:end], I[start:end], S[start:end])
            probs = torch.exp(-F.cross_entropy(logits, Xi[start:end], reduction='none'))       
            if b == 0:
                P = probs
            else:
                P = torch.cat([P, probs], dim = 0)
        return P
    
    @property 
    def parameter_count(self):
        params_count = self.context_agg.parameter_count
        params_count['embeddings'] += self.config.n_features* self.config.n_vocab* self.config.embedding_dim
        return params_count
