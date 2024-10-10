import torch
import torch.nn as nn
import yaml
from abc import abstractmethod
from omegaconf import OmegaConf as om
import numpy as np
from torch.nn import functional as F
from util import generate_permutations, torch_rand, stable_softmax
import math
from functools import cached_property
from config import AggregatorReduceType, ContextAggregatorType, DataType, ScoreFunctionType

class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)

class BaseModel(nn.Module):
    def __init___(self):
        super(BaseModel, self).__init__() 
                
    @abstractmethod
    def forward(self, X, I, S):
        return ...
    
    def logprob(self, X, I, S):
        logits = self(X, I, S)
        Xi = (X * I).sum(dim = -1)
        logp = - F.cross_entropy(logits, Xi, reduction='none')
        return logp
    
    def prob(self, X, I, S):
        
        return torch.exp(self.logprob(X, I, S))
    
    def predict(self, X, I, S):
        return self.forward(X, I, S).argmax(dim = 1)
    
    def accuracy(self, X, I, S):
        Xi = (X * I).sum(dim = -1)
        Xip = self.predict(X, I, S)
        return (Xip == Xi).float().mean()
    
  
    def prob_b(self, X, I, S):
        lP = self.logprob(X, I, S)
        P = torch.exp(lP)
        return P, lP
    
    def evaluate_batched(self, X, masks, metric, batch_size = None, seed=None):
        fn = None
        reduction = 'none'
        if metric == 'probability_i_S':
            fn = lambda *x, seed:  self.prob_b(*x)
        elif metric == 'probability_ij_S':
            fn = lambda *x, seed: self.evaluate_prob_double(*x)
        elif metric == 'swap_consistency':
            fn = lambda *x, seed:self.evaluate_swap_consistency(*x)
        elif metric == 'path_consistency':
            fn = self.evaluate_path_consistency
            reduction = 'mean'
        elif metric == 'autoregressive_consistency':
            fn = self.evaluate_autoregressive_consistency
            reduction = 'mean'
        else:
            raise ValueError(f'Unknown metric: {metric}')
        
        N = X.size(0)
        batch_size = batch_size if batch_size is not None else N
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        outputs = {i: [] for i in range(15)}
        nl = 0
        for b in range(num_batches):
            start = b * batch_size
            end = (b + 1) * batch_size
            masks_batched = [m[start:end] for m in masks]
            output = fn(X[start:end], *masks_batched, seed = seed)
            
            [outputs[i].append(o) for i, o in enumerate(output)]
            nl = len(output)

        outputs = {i: torch.cat(outputs[i], dim = 0) for i in range(nl)}
        
        if reduction == 'mean':
            outputs = [o.mean().item() for i, o in outputs.items()]
        else:
            outputs = [o for i, o in outputs.items()]
        return outputs

    def evaluate_swap_consistency(self, X, S, I, J):
        

        lp_i_S = self.logprob(X, I, S)
        lp_j_S = self.logprob(X, J, S)
        lp_i_Sj = self.logprob(X, I, S + J)
        lp_j_Si = self.logprob(X, J, S + I)
        
        p_i_S = torch.exp(lp_i_S)
        p_j_S = torch.exp(lp_j_S)
        p_i_Sj = torch.exp(lp_i_Sj)
        p_j_Si = torch.exp(lp_j_Si)

        p_diff_l1 = ((p_i_Sj * p_j_S)**(1/2) - (p_i_S * p_j_Si)**(1/2)).abs()
        # p_diff_l2 = ((p_i_Sj * p_j_S) - (p_i_S * p_j_Si)).abs()

        p_diff_log = (lp_i_Sj + lp_j_S - lp_i_S - lp_j_Si).abs()/2
                                            
        return p_diff_l1,  p_diff_log
    

    def evaluate_prob_double(self, X, S, I, J):
        lp_i_S = self.logprob(X, I, S)
        lp_j_S = self.logprob(X, J, S)
        lp_i_Sj = self.logprob(X, I, S + J)
        lp_j_Si = self.logprob(X, J, S + I)
        p_i_S = torch.exp(lp_i_S)
        p_j_S = torch.exp(lp_j_S)
        p_i_Sj = torch.exp(lp_i_Sj)
        p_j_Si = torch.exp(lp_j_Si)

        return p_i_S, p_j_S, p_i_Sj, p_j_Si, lp_i_S, lp_j_S, lp_i_Sj, lp_j_Si
     
    def evaluate_joint(self, X, P):
        """
            get log prob of of X evaluated according to permutation P
        """
        n_features = X.size(-1)
        logp = torch.zeros(X.size(0), device = X.device)
        for i in range(n_features):
            I = torch.zeros(X.size(0), n_features, device = X.device, dtype=torch.long)
            I.scatter_(1, P[:, i].view(-1, 1), 1)
            S = torch.zeros(X.size(0), n_features, device = X.device, dtype=torch.long)
            S.scatter_(1, P[:, :i], 1)
            logits = self(X, I, S)
            Xi = (X * I).sum(dim = -1)
            logp += - F.cross_entropy(logits, Xi, reduction='none')
        return logp/n_features
    
    def evaluate_joint_ratio(self, X1, X2, P):
        """
            get difference in log prob of X1 and X2 evaluated according to permutation P
        """
        
        logp = torch.zeros(X1.size(0), device = X1.device)
        n_features = X1.size(-1)
        for i in range(n_features):
            I = torch.zeros(X1.size(0), n_features, device = X1.device, dtype=torch.long)
            I.scatter_(1, P[:, i].view(-1, 1), 1)
            S1 = torch.zeros_like(I)
            S2 = torch.zeros_like(I)
            S1.scatter_(1, P[:, i+1:], 1)
            S2.scatter_(1, P[:, :i], 1)
            Xc = X1 * S1 + X2 * S2
            Xn = Xc + I * X1
            Xd = Xc + I * X2
            logprobn = self.logprob(Xn, I, 1 - I)
            logprobd = self.logprob(Xd, I, 1 - I)
            logp += logprobn - logprobd
        return logp/n_features

    def evaluate_autoregressive_consistency(self, X, seed = None):
        N = X.size(0)
        num_permutations = self.config.eval.consistency.num_permutations
        permutations = generate_permutations(self.config.n_features, num_permutations * N, X.device, seed = seed)
        # print(permutations.device)
        X_repeated = X.repeat_interleave(num_permutations, dim = 0)
        logP = self.evaluate_joint(X_repeated, permutations)
        logP = logP.reshape(N, num_permutations)
        logPstd = logP.std(dim=-1)
        logPmean = logP.mean(dim=-1).abs()
        logPstdr = logPstd / (logPmean + 1e-8)
        ##  TODO: normalize variances before taking mean
        return logPstd, logPstdr
    
    def evaluate_path_consistency(self, X, seed = None):
        N = X.size(0)
        n_features = X.size(-1)
        num_permutations = self.config.eval.consistency.num_permutations
        if seed is not None:
            torch_rand(seed)
        permutations = generate_permutations(n_features, num_permutations * N, X.device)
        X1 = X.repeat_interleave(num_permutations, dim = 0)
        X2 = X[torch.randperm(X.size(0))]
        X2 = X2.repeat_interleave(num_permutations, dim = 0)
        logPratio = self.evaluate_joint_ratio(X1, X2, permutations)
        logPratio = logPratio.reshape(N, num_permutations)
        metric_mean = logPratio.mean(dim=-1).abs()
        metric_std = logPratio.std(dim=-1)
        metric_stdr = metric_std / (metric_mean + 1e-8)
        
        ## TODO: normalize variances before taking mean 
        return metric_std, metric_stdr

class LinearContext(nn.Module):
    def __init__(self, config):
        super(LinearContext, self).__init__()
        self.config = config
        self.weights = nn.Embedding((config.n_vocab + 1) * config.n_features * config.n_features, config.context_dim)
        self.bias = nn.Embedding(config.n_features, config.context_dim)
    
    def forward(self, X, I, S):
        Ipos = I.argmax(dim = -1)
        Jpos = torch.arange(self.config.n_features, device = X.device)[None, :]
        
        Xt = Ipos[:, None] * self.config.n_features * (self.config.n_vocab + 1) + Jpos * (self.config.n_vocab + 1) + X * S + (1 - S) * self.config.n_vocab  
        Fxs =  self.weights(Xt).mean(dim = 1) + self.bias(Ipos)
        return Fxs
    
    @property
    def parameter_count(self):
        params_count = sum(p.numel() for p in self.parameters())
        return params_count
    
class ContextToOutput(nn.Module):
    def __init__(self, config):
        super(ContextToOutput, self).__init__()
        self.config = config
        self.context_dim = config.context_dim
        self.n_features = config.n_features
        self.hidden_dim = config.output.hidden_dim
        self.n_layers = config.output.n_layers - 1  

        if self.n_layers >= 0:
            ## or config.output.nlayers >= 1   (number of non linear layers)
            self.input_layer = nn.Linear(self.context_dim, self.hidden_dim * self.n_features)
            self.middle_layers = nn.ModuleList(
                [nn.Linear(self.hidden_dim, self.hidden_dim * self.n_features) for _ in range(self.n_layers)]
            )
            self.act = nn.ReLU()
            self.output_layer = nn.Linear(self.hidden_dim, self.config.n_vocab * self.n_features)
        else:
            ## a single linear layer (no non-linearity )
            self.skip_layer = nn.Linear(self.context_dim, self.config.n_vocab * self.n_features)
    
    def select_features(self, H, I):
        H = H.view(H.size(0), self.n_features, -1)
        H = (H * I.unsqueeze(-1)).sum(dim = 1)
        return H
    
    def forward(self, Fxs, I):
        if self.n_layers >= 0:
            i_context = self.input_layer(Fxs)
            i_context = self.select_features(i_context, I)
            i_context = self.act(i_context)
            for l in range(self.n_layers):
                i_context = self.middle_layers[l](i_context)
                i_context = self.select_features(i_context, I)
                i_context = self.act(i_context)
            logits = self.output_layer(i_context)
            logits = self.select_features(logits, I)
        else:
            logits = self.skip_layer(Fxs)
            logits = self.select_features(logits, I)
        return logits
    
    @property
    def parameter_count(self):
        params_count = sum(p.numel() for p in self.parameters())
        return params_count 
    
class GeneralModel(BaseModel):
    ## P( | Xs) = g_i(psi_i(Xs))
    ## psi_i(Xs) : V^|S| -> R^d  (summarize context features to predict feature i)
    ## g_i : R^d -> R^V  (logits) (predict feature i from summarized context features)

    def __init__(self, config):
        super(GeneralModel, self).__init__()
        self.config = config
        self.context_featurizer = LinearContext(config)
        self.output_model = ContextToOutput(config)

    def forward(self, X, I, S):
        Fxs = self.context_featurizer(X, I, S)
        logits = self.output_model(Fxs, I)
        return logits
    
    @property
    def parameter_count(self):
        params_count = dict()
        params_count['context_model'] = self.context_featurizer.parameter_count
        params_count['output_model'] = self.output_model.parameter_count
        return params_count
    

## P(X_i = 1 | X_S) = g_i ( psi_i (X_S))  
## psi_i(X_S) : 2^|S| -> R^d  (summarize context features to predict feature i) (n^2 parameters here)
## g_i : R^d -> R^2  (logits) (predict feature i from summarized context features)

## psi_i(X_S) = (\sum_{j \in S} : w_{ijX_j}) + b_i

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
      
## rho( \sum_{j\in S} w_{ijX_j}) ## n x n x |V|
class ContextAggregator(nn.Module):
    def __init__(self, config):
        super().__init__()
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
        embeddings = [self.embeddings[i](Xm[:, i]) for i in range(self.n_features)]
        embeddings = torch.stack(embeddings, dim = 1)
        embeddings =  embeddings + self.positional_embeddings.unsqueeze(0)
        return embeddings
        
    def forward(self, X, I, S):
        x = self.token_to_embeddings(X, S)
        for i in range(self.n_layers):
            x_ = self.aggregate(x, i, I + S)
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

class ExpModel(BaseModel):
    def __init__(self, config):
        super(ExpModel, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.n_vocab, config.embedding_dim)
        self.context_agg = MLPAggregator(config) if config.aggregator.type == ContextAggregatorType.mlp\
                else TransformerAggregator(config) if config.aggregator.type == ContextAggregatorType.transformer else None
          
    def forward(self, X, I, S):
        Fxs = self.context_agg(X, I, S)
        logits = F.linear(Fxs, self.embedding.weight, None)
        return logits
    
    @property 
    def parameter_count(self):
        params_count = self.context_agg.parameter_count
        params_count['embeddings'] += sum(p.numel() for p in self.embedding.parameters())
        return params_count
