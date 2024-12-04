
import torch 
from torch.nn import functional as F
from transformers import BertTokenizer

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def logprob(model, X, I):
    X_ = X * (1 - I) + I * tokenizer.mask_token_id
    batch_attention_masks = (X_ != tokenizer.pad_token_id).long()
    logits = model(input_ids=X_, attention_mask=batch_attention_masks).logits
    logits = (logits * I.unsqueeze(-1)).sum(dim=1)
    Xi = (X * I).sum(dim = -1)
    logp = - F.cross_entropy(logits, Xi, reduction='none')
    return logp
def generate_permutations(n, num_permutations, device, seed = None):
    if seed is not None:
        torch_rand(seed)
    noise = torch.rand(num_permutations, n, device=device)
    permutations = noise.argsort(dim = -1)
    return permutations
def torch_rand(seed):
    if torch.backends.cudnn.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

def evaluate_path_consistency(model, X, num_permutations = 100, seed = None):
    N = X.size(0)
    n_features = X.size(-1)
    if seed is not None:
        torch_rand(seed)
    permutations = generate_permutations(n_features, num_permutations * N, X.device)
    X1 = X.repeat_interleave(num_permutations, dim = 0)
    X2 = X[torch.randperm(X.size(0))]
    X2 = X2.repeat_interleave(num_permutations, dim = 0)
    logPratio = evaluate_joint_ratio(model, X1, X2, permutations)
    logPratio = logPratio.reshape(N, num_permutations)
    metric_mean = logPratio.mean(dim=-1).abs()
    metric_std = logPratio.std(dim=-1)
    metric_stdr = metric_std / (metric_mean + 1e-8)
    
    ## TODO: normalize variances before taking mean 
    return metric_std, metric_stdr
def evaluate_joint_ratio(model, X1, X2, P):
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
        logprobn = logprob(model, Xn, I)
        logprobd = logprob(model, Xd, I)
        logp += logprobn - logprobd
        print(logp.shape)
        print(logp)
    return logp/n_features