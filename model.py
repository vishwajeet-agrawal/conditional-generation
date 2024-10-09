import torch
import torch.nn as nn
import yaml
from abc import abstractmethod
from omegaconf import OmegaConf as om
import numpy as np
from torch.nn import functional as F
from util import generate_permutations
import math
from functools import cached_property


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)

class BaseModel(nn.Module):
    def __init___(self, config, args):
        self.config = config
        self.args = args
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
    
  
    
    def evaluate_batched(self, X, masks, metric, batch_size = None):
        fn = None
        reduction = 'none'
        if metric == 'probability_i_S':
            fn = self.prob
        elif metric == 'probability_ij_S':
            fn = self.evaluate_prob_double
        elif metric == 'swap_consistency':
            fn = self.evaluate_swap_consistency
        elif metric == 'path_consistency':
            fn = self.evaluate_path_consistency
            reduction = 'mean'
        elif metric == 'autoregressive_consistency':
            fn = self.evaluate_autoregressive_consistency
        else:
            raise ValueError(f'Unknown metric: {metric}')
        
        N = X.size(0)
        batch_size = batch_size if batch_size is not None else N
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        outputs = {i: [] for i in range(10)}
        nl = 0
        for b in range(num_batches):
            start = b * batch_size
            end = (b + 1) * batch_size
            masks = [m[start:end] for m in masks]
            output = fn(X[start:end], *masks)
            [outputs[i].append(o) for i, o in enumerate(output)]
            nl = len(output)

        outputs = {i: torch.cat(outputs[i], dim = 0) for i in range(nl)}
        if reduction == 'mean':
            outputs = {i: o.mean() for i, o in outputs.items()}
        
        return outputs.values()

    def evaluate_swap_consistency(self, X, S, I, J):
        self.eval()

        lp_i_S = self.logprob(X, I, S)
        lp_j_S = self.logprob(X, J, S)
        lp_i_Sj = self.logprob(X, I, S + J)
        lp_j_Si = self.logprob(X, J, S + I)
        
        p_i_S = torch.exp(lp_i_S)
        p_j_S = torch.exp(lp_j_S)
        p_i_Sj = torch.exp(lp_i_Sj)
        p_j_Si = torch.exp(lp_j_Si)

        p_diff_l1 = ((p_i_Sj * p_j_S)**(1/2) - (p_i_S * p_j_Si)**(1/2)).abs()
        p_diff_l2 = ((p_i_Sj * p_j_S) - (p_i_S * p_j_Si)).abs()

        p_diff_log = (lp_i_Sj + lp_j_S - lp_i_S - lp_j_Si).abs()
                                            
        return p_diff_l1, p_diff_l2, p_diff_log
    

    def evaluate_prob_double(self, X, S, I, J):
        self.eval()
        p_i_S = self.prob(X, I, S)
        p_j_S = self.prob(X, J, S)
        p_i_Sj = self.prob(X, I, S + J)
        p_j_Si = self.prob(X, J, S + I)
        return p_i_S, p_j_S, p_i_Sj, p_j_Si
     
    def evaluate_joint(self, X, P):
        """
            get log prob of of X evaluated according to permutation P
        """
        self.eval()
        logp = torch.zeros(X.size(0), device = X.device)
        for i in range(self.n_features):
            I = torch.zeros(X.size(0), self.n_features, device = X.device)
            I.scatter_add_(1, P[:, i], 1)
            S = torch.zeros(X.size(0), self.n_features, device = X.device)
            S.scatter_add_(1, P[:, :i], 1)
            logits = self(X, I, S)
            Xi = (X * I).sum(dim = -1)
            logp += - F.cross_entropy(logits, Xi, reduction='none')
        return logp
    
    def evaluate_joint_ratio(self, X1, X2, P):
        """
            get difference in log prob of X1 and X2 evaluated according to permutation P
        """
        self.eval()
        logp = torch.zeros(X1.size(0), device = X1.device)
        for i in range(self.n_features):
            I = torch.zeros(X1.size(0), self.n_features, device = X1.device)
            I.scatter_add_(1, P[:, i], 1)
            S1 = torch.zeros_like(I)
            S2 = torch.zeros_like(I)
            S1.scatter_add_(1, P[:, i+1:], 1)
            S2.scatter_add_(1, P[:, :i], 1)
            Xc = X1 * S1 + X2 * S2
            Xn = Xc + I * X1
            Xd = Xc + I * X2
            logprobn = self.logprob(Xn, I, 1 - I)
            logprobd = self.logprob(Xd, I, 1 - I)
            logp += logprobn - logprobd
        return logp

    def evaluate_autoregressive_consistency(self, X):
        N = X.size(0)
        num_permutations = self.config.eval.consistency.num_permutations
        permutations = generate_permutations(self.n_features, num_permutations * N, X.device)
        X_repeated = X.repeat_interleave(num_permutations, dim = 0)
        logP = self.evaluate_joint(X_repeated, permutations)
        logP = logP.reshape(N, num_permutations)
        logPstd = logP.std(dim=-1)
        logPmean = logP.mean(dim=-1)
        return logPmean, logPstd
    
    def evaluate_path_consistency(self, X):
        N = X.size(0)
        num_permutations = self.config.eval.consistency.num_permutations
        permutations = generate_permutations(self.n_features, num_permutations * N, X.device)
        X1 = X.repeat_interleave(num_permutations, dim = 0)
        X2 = X[torch.randperm(X.size(0))]
        X2 = X2.repeat_interleave(num_permutations, dim = 0)
        logPratio = self.evaluate_joint_ratio(X1, X2, permutations)
        logPratio = logPratio.reshape(N, num_permutations)
        return logPratio.std(dim=-1).mean()

class LinearContext(nn.Module):
    def __init__(self, config):
        super(LinearContext, self).__init__()
        self.config = config
        self.weights = nn.Embedding((config.n_vocab + 1) * config.n_features * config.n_features, config.context_dim)
        self.bias = nn.Embedding(self.n_features, config.context_dim)
    
    def forward(self, X, I, S):
        Ipos = I.argmax(dim = -1)[:, None]
        Jpos = torch.arange(self.config.n_features, device = X.device)[None, :]
    
        Xt = Ipos * self.config.n_features * (self.config.n_vocab + 1) + Jpos * (self.config.n_vocab + 1) + X * S + (1 - S) * self.config.n_vocab  
        return self.weights(Xt).mean(dim = 1) + self.bias(Ipos)
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
        self.n_layers = config.output.n_layers
        self.input_layer = nn.Linear(self.context_dim, self.hidden_dim * self.n_features)
        self.middle_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim * self.n_features) for _ in range(self.n_layers)]
        )
        self.act = nn.ReLU()
        self.output_layer = nn.Linear(self.hidden_dim, self.config.n_vocab)

    def forward(self, Fxs, I):
        i_context = self.input_layer(Fxs)
        i_context = i_context.view(-1, self.n_features, self.hidden_dim)
        i_context = (i_context * I.unsqueeze(-1)).sum(dim = 1)
        for l in range(self.n_layers):
            i_context = self.middle_layers[l](i_context)
            i_context = i_context.view(-1, self.n_features, self.hidden_dim)
            i_context = (i_context * I.unsqueeze(-1)).sum(dim = 1)
            i_context = self.act(i_context)

        logits = self.output_layer(i_context)
        return logits
    @property
    def parameter_count(self):
        params_count = sum(p.numel() for p in self.parameters())
        return params_count 
    
class GeneralModel(BaseModel):
    ## P( | Xs) = g_i(psi_i(Xs))
    ## psi_i(Xs) : V^|S| -> R^d  (summarize context features to predict feature i)
    ## g_i : R^d -> R^V  (logits) (predict feature i from summarized context features)

    def __init__(self, config, args):
        super(GeneralModel, self).__init__(config, args)
        self.context_featurizer = LinearContext(config)
        self.output_model = ContextToOutput(config)

    def forward(self, X, I, S):
        Fxs = self.context_featurizer(X, I, S)
        logits = self.output_model(Fxs)
        return logits
    
    @property
    def parameter_count(self):
        params_count = dict()
        params_count['context_model'] = self.context_featurizer.parameter_count
        params_count['output_model'] = self.output_model.parameter_count
        return params_count