import numpy as np
from omegaconf import OmegaConf as om
from config import SourceType, JointDistributionType, MaskDistributionType, DataType
import networkx as nx
import numpy as np
from itertools import product
from abc import abstractmethod
import pandas as pd
from dataclasses import dataclass
import torch
import json
from functools import cached_property
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import GibbsSampling
import glob
import os
import pickle
from tqdm import tqdm
class BaseGenerator:
    def __init__(self, config):
        self.config = config
        self.n_features = config.n_features
    @abstractmethod
    def sample_joint(self, N):
        pass

    @property
    def num_samples(self):
        return np.inf
    
    @abstractmethod
    def estimate_conditional_prob(self, X, I, S):
        pass

class DataFromFile(BaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.path = config.source.path
        if config.source.path.endswith('.csv'):
            self.samples = pd.read_csv(config.source.path).values
        elif config.source.path.endswith('.npy'):
            self.samples = np.load(config.source.path)
        else:
            raise ValueError(f'Unknown file type: {config.source.path.split(".")[-1]}')
        
    def sample_joint(self, N):
        idx = np.random.choice(len(self.samples), N)
        return self.samples[idx]
    @property
    def num_samples(self):
        return len(self.samples)
    
    
    def estimate_conditional_prob(self, X, I, S):
        I = np.eye(X.size(1))[I]

        if self.config.type == DataType.binary:
            p = torch.zeros(S.shape[0], device = X.device)
            samples = torch.tensor(self.samples, device = X.device, dtype=torch.float)
            S = S.bool()
            I = I.bool()
            for i in range(S.shape[0]):
                Ssamples = (samples[:, S[i]] == X[i, S[i]]).all(dim = 1)
                if Ssamples.sum() == 0:
                    p[i] = 0.5
                else:
                    p[i] = (samples[Ssamples, I[i]] == X[i, I[i]]).float().mean() 
            return p
        else:
            raise NotImplementedError('Continuous data is not supported yet')
    
class DataFromDistribution(BaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.distribution = config.source.distribution
    def sample_joint(self, N):
        if self.distribution == JointDistributionType.normal:
            return np.random.normal(size=(N, self.n_features))
        elif self.distribution == JointDistributionType.normal:
            return np.random.uniform(size=(N, self.n_features))
        else:
            raise ValueError(f'Unknown distribution type: {self.distribution}')


class DataFromDAG(BaseGenerator):
    
    def __init__(self, config):
        super().__init__(config)   
        self.seed = config.get('dist_seed', 0)
        self.n_features = config.n_features
        self.edge_probability = config.source.edge_probability      
        self.size_of_cpt = 0
        self.num_edges = 0
        self.max_parents = 0
        if config.load:
            self.G = self.load(config.path)
        else:
            self.G = self.create_random_dag(self.edge_probability)
            self.assign_cpt(self.G)

        self._cp_cache = {}
        self.bn = self.networkx_to_pgmpy()
        self.infer_ve = VariableElimination(self.bn)
        # print('Create Gibbs Sampling')
        # self.infer_gb = GibbsSampling(self.bn)
        # print('End Gibbs Sampling')
        # self.infer_bp = BeliefPropagation(self.bn)
        # self.infer_wlw = WeightedLikelihoodWeighting(self.bn)
        # self.infer_gb = GibbsSampling(self.bn)
        if config.load:
            config = self.load_stat(config.path)
            self.custom_config = {'size_of_cpt': config['size_of_cpt'], 'num_edges': config['num_edges'], 'max_parents': config['max_parents']}
        else:
            self.custom_config = {'size_of_cpt': self.size_of_cpt, 'num_edges': self.num_edges, 'max_parents': self.max_parents}

    def networkx_to_pgmpy(self):
        G = self.G
        """
        Convert a NetworkX DiGraph with CPTs to a pgmpy BayesianNetwork.
        
        :param G: NetworkX DiGraph with 'cpt' and 'parents' attributes for each node
        :return: pgmpy BayesianNetwork
        """
        # Create a pgmpy BayesianNetwork with the same structure
        node_mapping = {node: str(node) for node in G.nodes()}
        edges = [(node_mapping[u], node_mapping[v]) for u, v in G.edges()]

        bn = BayesianNetwork(edges)
        for node in G.nodes():
            bn.add_node(node_mapping[node])
        for node in G.nodes():
            parents = [node_mapping[p] for p in G.nodes[node]['parents']]
            cpt = G.nodes[node]['cpt']
            
            # Determine the number of parent configurations
            n_parent_configs = 2**len(parents) if parents else 1
            
            # Reshape the CPT to match pgmpy's expected format
            # perm_order = (len(parents),) + tuple(range(len(parents)))
            # cpt.transpose(perm_order).reshape(2, n_parent_configs)
            # print(perm_order, n_parent_configs)
            reshaped_cpt = cpt.reshape((n_parent_configs, 2)).transpose(1, 0)
            # reshaped_cpt = cpt.reshape((2, n_parent_configs)).transpose(perm_order)
            
            # Create a TabularCPD object
            cpd = TabularCPD(
                variable=node_mapping[node],
                variable_card=2,
                values=reshaped_cpt,
                evidence=parents,
                evidence_card=[2] * len(parents)
            )
            
            # Add the CPD to the BayesianNetwork
            bn.add_cpds(cpd)
        
        # Check if the model is valid
        assert bn.check_model()

        return bn

    def create_random_dag(self, edge_probability=0.3):
        """Create a random directed acyclic graph (DAG)."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_features))
        np.random.seed(self.seed)
        num_edges = 0
        max_parents = 0
        for i in range(self.n_features):
            num_edges_i = 0
            for j in range(i+1, self.n_features):
                if np.random.random() < edge_probability:
                    G.add_edge(i, j)
                    num_edges_i += 1
                if num_edges_i >= self.config.source.max_parents:
                    break
            max_parents = max(max_parents, num_edges_i)
            num_edges += num_edges_i
        self.num_edges = num_edges
        self.max_parents = max_parents
        return G

    def assign_cpt(self, G):
        """Assign conditional probability tables (CPTs) to each node."""
        np.random.seed(self.seed)
        for node in G.nodes():
            parents = list(G.predecessors(node))
            n_parents = len(parents)
            
            # Create CPT
            cpt_shape = [2] * (n_parents + 1)
            cpt = np.random.dirichlet(np.ones(2), size=cpt_shape[:-1]).reshape(cpt_shape)
            self.size_of_cpt += cpt.flatten().shape[0]
            # print(cpt.shape)
            # exit()
            # Assign CPT to node
            G.nodes[node]['cpt'] = cpt
            G.nodes[node]['parents'] = parents

    def load(self, path):
        path = path + '/dist.pkl'
        with open(path, 'rb') as f:
            G = pickle.load(f)
        return G
    def load_stat(self, path):
        path = path + '/config.yaml'
        with open(path, 'r') as f:
            config = om.load(f)
        return config
    def save(self, path):
        ans = dict()
        ans['n_nodes'] = self.n_features
        ans['edges'] = []
        ans['cpt'] = dict()
        for node in self.G.nodes():
            ans['edges'].extend([(node, parent) for parent in self.G.nodes[node]['parents']])
            ans['cpt'][node] = self.G.nodes[node]['cpt'].tolist()

        with open(path + '/dist.json', 'w') as f:
            json.dump(ans, f)
        with open(path + '/dist.pkl', 'wb') as f:
            pickle.dump(self.G, f)

    def reset_cp_cache(self):
        self._cp_cache = {}

    def _conditional_probability(self, i, S, evidence):
        evidence = {str(k): v for k, v in evidence.items()}
        key = (str(i), tuple(evidence.items()))
        if key not in self._cp_cache:   
            self._cp_cache[key] =  self.infer_ve.query([str(i)], evidence = evidence)
        return self._cp_cache[key]
    
    def sample_joint(self, n_samples, seed = None):
        if seed is not None:
            np.random.seed(seed)
        G = self.G
        """Generate samples from the Bayesian Network."""
        samples = np.zeros((n_samples, len(G.nodes)), dtype=int)
        
        for node in nx.topological_sort(G):
            parents = G.nodes[node]['parents']
            cpt = G.nodes[node]['cpt']
            
            if not parents:
                prob = cpt[1]
                samples[:, node] = np.random.random(n_samples) < prob
            else:
                parent_values = samples[:, parents]
                parent_configs = [tuple(row) for row in parent_values]
                probs = np.array([cpt[config + (1,)] for config in parent_configs])
                samples[:, node] = np.random.random(n_samples) < probs
        
        return samples
    def estimate_conditional_prob(self, X, I, S):
        P = np.zeros(X.shape[0])
        self.reset_cp_cache()
        
        for j in range(X.shape[0]):
            i = I[j]
            s = {k for k in range(self.n_features) if S[j, k] == 1}
            # print(i, s)
            assert i not in s
            result = self._conditional_probability(i, s, evidence = {k: X[j, k] for k in s})
            
            P[j] = result.values[X[j, i]]
        return P

class DataGenerator:
    def __init__(self, config, save=True):
        self.config = config
        self.n_features = self.config.n_features
        self.test_data_saved = False
        self.path = self.set_path(config.dir) if not config.get('load', False) else config.path
        self.test_seed = config.get('test_seed', 0)
        
        if self.config.source.type == SourceType.samples:
            self.sampler = DataFromFile(self.config)
        elif self.config.source.type == SourceType.distribution:
            self.sampler = DataFromDistribution(self.config)
        elif self.config.source.type == SourceType.dag:
            self.sampler = DataFromDAG(self.config)
        else:
            raise ValueError(f'Unknown source type: {self.config.source.type}')
       

        if config.save:
            self.save_dist()
            self.save_test_data()

    def set_path(self, dir):
        subfiles = glob.glob(dir + '/*')
        if len(subfiles) == 0:
            fol = dir + '/1'
            os.makedirs(fol)
            return fol
        else:
            for i in range(10000):
                fol = dir + f'/{i}'
                if f'{i}' not in [sf.split('/')[-1] for sf in subfiles]:
                    os.makedirs(fol)
                    return fol
           
<<<<<<< HEAD
    def sample_conditional(self, N, seed = None, nomask = False):  
        if seed is not None:
            np.random.seed(seed)
        X = self.sampler.sample_joint(N)
        S = np.zeros((N, self.n_features), dtype = int)        
        I = np.zeros(N, dtype = int)
        J = np.zeros(N, dtype = int)
=======
    def sample_conditional(self, N, seed = None):  
        X = self.sampler.sample_joint(N, seed)
        S = np.zeros((N, self.n_features), dtype = int)
>>>>>>> 3da66ed (restore data)
        
        if not nomask:
            if self.config.mask.distribution == MaskDistributionType.truncnorm:
                mean = self.config.mask.params.mean
                std = self.config.mask.params.std
                masksize = np.random.normal(mean, std, size = (N,))
            elif self.config.mask.distribution == MaskDistributionType.uniform:
                assert self.config.mask.params.min >= 0 and self.config.mask.params.max <= 1 and self.config.mask.params.min < self.config.mask.params.max
                minms = max(int(self.config.mask.params.min * self.n_features), 1)
                maxms = int(self.config.mask.params.max * self.n_features)
                masksize = np.random.uniform(minms, maxms + 1, size = (N,))
            elif self.config.mask.distribution == MaskDistributionType.delta:
                masksize = np.ones(N) * self.config.mask.params.value * self.n_features
            else:
                raise ValueError(f'Unknown distribution type: {self.config.mask.distribution}')
        else:
            masksize = np.ones(N) * self.n_features

        masksize = np.clip(masksize, 2, self.n_features)
        for i in range(N):
            indices = np.random.permutation(self.n_features)[:int(masksize[i])]
           
            index_i = indices[0]
            index_j = indices[1]
            indices_s = indices[2:]
            S[i, indices_s] = 1
            I[i] = index_i
            J[i] = index_j

        return X, S, I, J
    
    def sample_marginal(self, N, seed=None):
        if seed is not None:
            np.random.seed(seed)
        X = self.sampler.sample_joint(N, seed)
  
        
        S = np.zeros((N, self.n_features))
        if self.config.mask.distribution == MaskDistributionType.truncnorm:
            mean = self.config.mask.params.mean
            std = self.config.mask.params.std
            masksize = np.random.normal(mean, std, size = (N,))
        elif self.config.mask.distribution == MaskDistributionType.uniform:
            minms = max(int(self.config.mask.params.min * self.n_features) - 1, 0)
            maxms = min(int(self.config.mask.params.max * self.n_features), self.n_features)
            masksize = np.random.uniform(minms, maxms, size = (N,))
        elif self.config.mask.distribution == MaskDistributionType.delta:
            masksize = np.ones(N) * self.config.mask.params.value
        for i in range(N):
            masksize = np.clip(masksize, 0, self.n_features - 1)
            indices = np.random.choice(np.arange(self.n_features), int(masksize[i]), replace=False)
            S[i, indices] = 1
        return X, S
    
    def save_test_data(self):
        if not self.config.load:
            X, S, I, J, P = self.test_data
            path = self.path + '/test_data.npz'
            np.savez(path, X = X, S = S, I = I, J = J, P = P)

            
    def save_dist(self):
        if not self.config.load:
            self.sampler.save(self.path)
            path = self.path + '/config.yaml'
            resolved_cfg = om.to_container(self.config, resolve=True)
            resolved_cfg.pop('save', None)
            resolved_cfg.pop('load', None)
            resolved_cfg.pop('dir', None)
            for key in self.sampler.custom_config:
                resolved_cfg[key] = self.sampler.custom_config[key]
            om.save(resolved_cfg, path)
        

    @cached_property
    def test_data(self):
        if self.config.load:
            path = self.path + '/test_data.npz'
            data = np.load(path)
            X = data['X']
            I = data['I']
            J = data['J']
            S = data['S']
            P = data['P']
            return X, S, I, J, P
        X, S, I, J = self.sample_conditional(self.config.eval_size, seed = self.test_seed)
        P_i_S = self.sampler.estimate_conditional_prob(X, I, S) # p(x_i | x_s)
        P_j_S = self.sampler.estimate_conditional_prob(X, J, S) # p(x_j | x_s)
        P_i_S_j = self.sampler.estimate_conditional_prob(X, I, S + np.eye(self.n_features)[J]) # p(x_i | x_s, x_j)
        P_j_S_i = self.sampler.estimate_conditional_prob(X, J, S + np.eye(self.n_features)[I]) # p(x_j | x_s, x_i)
        P_i_r = self.sampler.estimate_conditional_prob(X, I, 1 - np.eye(self.n_features)[I]) # for p(x_i | x_{-i})
        P = np.stack([P_i_S, P_j_S, P_i_S_j, P_j_S_i, P_i_r], axis = 1)

        return X, S, I, J, P
    

    @property
    def estimate_true_prob(self):
        return self.test_data[-1]
    

if __name__ == '__main__':
    from omegaconf import OmegaConf as om
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument('--config', type = str, default='config_data.yaml')
    parser.add_argument('--test_samples', type = int, default=10000)
    parser.add_argument('--dir', type = str, default='data')
    parser.add_argument('--nf', type = int, default=10)
    parser.add_argument('--ep', type = float, default=0.2)
    parser.add_argument('--configs', type = str, default='data_configs.csv')
    parser.add_argument('--multiple', action='store_true', default=True)
    parser.add_argument('--mask_dist', type = str, default='uniform')
    parser.add_argument('--mask_min', type = float, default=0)
    parser.add_argument('--mask_max', type = float, default=1)
    parser.add_argument('--n_repeats', type = float, default=1)
    # exit()
    # raise NotImplementedError('This script is not ready yet')
    args = parser.parse_args()
    datas = []
    # config = om.load(args.config)
    if args.multiple:
        configs = pd.read_csv(args.configs)
        for i in tqdm(range(len(configs))):
            config = configs.iloc[i]
            for j in range(args.n_repeats):
                print(f'Generating data {i} with seed {j} with config {config}')
                
                data_generator = DataGenerator(om.create({'n_features': int(config['n_features']), 
                                                        'source': {'type': 'dag', 
                                                                    'edge_probability': float(config['edge_probability']),
                                                                    'max_parents': int(config['max_parents'])},
                                                        'dir' : args.dir,
                                                        'eval_size': args.test_samples,
                                                        'load': False,
                                                        'save': True,
                                                        'test_seed': j + 101,
                                                        'dist_seed': j + 101,
                                                        'mask': {
                                                            'distribution':args.mask_dist,
                                                            'params': {'min': args.mask_min, 'max': args.mask_max}
                                                        }}))
                datas.append({'path': data_generator.path,
                            'n_features': data_generator.n_features,
                            'source': 'dag',
                            'max_parents': data_generator.sampler.max_parents,
                            'num_edges': data_generator.sampler.num_edges,
                            'size_of_cpt': data_generator.sampler.size_of_cpt,
                            })
        pd.DataFrame(datas).to_csv('datas.csv', index = False)
    else:
        config = om.create({'n_features': args.nf, 'source': {'type': 'dag', 'edge_probability': args.ep}})
        data_generator = DataGenerator(config)
        args.path = 'datasets/' + args.path
        data_generator.sampler.save(args.n_samples, args.path)