import torch
from train import Trainer
from model import Model
from data import DataGenerator
import os
from omegaconf import OmegaConf as om
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm

def get_model_data_params(row, model, data_generator):
    params = om.create(row_.loc[0].to_dict())    
    params = om.merge(params, data_generator.sampler.custom_config)
    params = om.merge(params, om.create({'numparams': sum(model.parameter_count.values())}))
    return om.to_container(params, resolve = True)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', type = str, default='config.yaml')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--configs', type = str, default='configs.csv')
    parser.add_argument('--datas', type = str, default='datas.csv')
    parser.add_argument('--configs_save', type = str, default='configs_all.csv')
    

    args = parser.parse_args()
    config = om.load(args.config)
    configs_all = []
    configs = pd.read_csv(args.configs)

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
            config_.model.embedding_dim = int(row['embedding_dim'])
            config_.model.aggregator.type = row['context_aggregator']
            config_.model.aggregator.n_layers = int(row['n_layers'])
            config_.model.aggregator.n_heads = int(row['n_heads'])
            config_.model.aggregator.reduce_type = row['reduce_type']
            config_.model.aggregator.learn_adjacency = row['learn_adjacency']
            config_.model.aggregator.tie_embeddings = row['tie_embeddings']
            config_.model.aggregator.tie_aggregator = row['tie_aggregator']

            row_ = pd.DataFrame({'datapath': config_.data.path,
                                    'n_features': n_features,
                                    'embedding_dim': config_.model.embedding_dim,
                                    'aggregator': config_.model.aggregator.type,
                                    'n_layers': config_.model.aggregator.n_layers,
                                    'n_heads': config_.model.aggregator.n_heads,
                                    'reduce_type': config_.model.aggregator.reduce_type,
                                    'learn_adjacency': config_.model.aggregator.learn_adjacency,
                                    'tie_embeddings': config_.model.aggregator.tie_embeddings,
                                    'tie_aggregator': config_.model.aggregator.tie_aggregator}, index = [0])
            row_ = row_.fillna('')

            torch.manual_seed(135)
            torch.cuda.manual_seed(135)
            model = Model(config_.model, args).to(config_.train.device)
            data_generator = data_generators.get(config_.data, DataGenerator(config_.data))
            data_generators[config_.data] = data_generator

            configs_all.append(get_model_data_params(row_, model, data_generator))

            pd.DataFrame(configs_all).to_csv(args.configs_save, index = False)
