import torch
from model import GeneralModel
from data import DataGenerator
import os
from omegaconf import OmegaConf as om
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
from itertools import product
from util import torch_rand
def get_model_data_params(row_, model, data_generator):

    params = om.create(row_.loc[0].to_dict())    
    params = om.merge(params, data_generator.sampler.custom_config)
    params = om.merge(params, om.create({'numparams': sum(model.parameter_count.values())}))
    return om.to_container(params, resolve = True)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', type = str, default='config.yaml')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--configs', type = str, default='configs.yaml')
    parser.add_argument('--datas', type = str, default='datas.csv')
    parser.add_argument('--configs_save', type = str, default='configs_all.csv')
    
    configs = om.load('configs.yaml')
    hyperparams = product(*configs.values())
    configs = pd.DataFrame(hyperparams, columns=configs.keys())


    args = parser.parse_args()
    config = om.load(args.config)
    configs_all = []

    data_generators = dict()
    data_configs = pd.read_csv(args.datas)
    
    for j, r in data_configs.iterrows():
        n_features = int(r['n_features'])
        data_path = str(r['path'])
        for i, row in tqdm(configs.iterrows()):
            config_ = om.structured(config)
            config_.data.path = os.path.join(config.data.dir, data_path)
            # config_.train.save_dir = args.save_dir
            config_.model.n_features = config_.data.n_features = n_features
            config_.model.context_dim = int(row['context_dim'])
            config_.model.output.hidden_dim = int(row['hidden_dim'])
            config_.model.output.n_layers = int(row['n_layers'])
            row_ = pd.DataFrame({'datapath': config_.data.path,
                                    'n_features': n_features,
                                    'context_dim': config_.model.context_dim,
                                    'n_layers': config_.model.output.n_layers,
                                    'hidden_dim': config_.model.output.hidden_dim}, index = [0])

            torch_rand(135)
            model = GeneralModel(config_.model).to(config_.train.device)
            path = config_.data.path + '/config.yaml'
            with open(path, 'r') as f:
                config__ = om.load(f)
            data_stats = om.create({'size_of_cpt': config__['size_of_cpt'], 'num_edges': config__['num_edges'], 'max_parents': config__['max_parents']})

            params = om.create(row_.loc[0].to_dict())    
            params = om.merge(params, data_stats)
            params = om.merge(params, om.create({'numparams': sum(model.parameter_count.values())}))
            configs_all.append(om.to_container(params, resolve = True))

            pd.DataFrame(configs_all).to_csv(args.configs_save, index = False)
