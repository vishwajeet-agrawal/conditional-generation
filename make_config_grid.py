from omegaconf import OmegaConf as om
import numpy as np
import pandas as pd
from itertools import product
if __name__ == '__main__':
    configs = om.load('configs.yaml')
    hyperparams = product(*[configs.context_dim, configs.n_layers, configs.hidden_dim])
    df = pd.DataFrame(hyperparams, columns=['context_dim', 'n_layers', 'hidden_dim' ])
    
   
    
    df = df.drop_duplicates()

    df.to_csv('configs.csv', index=False)