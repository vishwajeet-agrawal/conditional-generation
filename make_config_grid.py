from omegaconf import OmegaConf as om
import numpy as np
import pandas as pd
from itertools import product
if __name__ == '__main__':
    configs = om.load('configs.yaml')
    hyperparams = product(*[configs.n_features, configs.embedding_dim, 
                            configs.aggregator, configs.n_layers, configs.tie_embeddings, configs.n_heads, configs.reduce_type, 
                             configs.learn_adjacency, configs.tie_aggregator])
    df = pd.DataFrame(hyperparams, columns=['n_features', 'embedding_dim', 
                                            'context_aggregator', 'n_layers', 'tie_embeddings',
                                            'n_heads', 'reduce_type',  'learn_adjacency','tie_aggregator' ])
    
    df = df[df['embedding_dim'] % df['n_heads'] == 0]
    
    df.loc[df['context_aggregator']=='transformer', 'tie_aggregator'] = pd.NA
    df.loc[df['context_aggregator']=='transformer', 'learn_adjacency'] = pd.NA
    df.loc[df['context_aggregator']=='transformer', 'reduce_type'] = pd.NA
    df.loc[df['context_aggregator']=='mlp', 'n_heads'] = 1
    df.loc[(df['context_aggregator']=='mlp') & (df['learn_adjacency']==False), 'tie_aggregator'] = pd.NA
    
    df = df.drop_duplicates()

    df.to_csv('configs.csv', index=False)