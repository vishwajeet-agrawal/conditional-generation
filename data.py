import numpy as np
from omegaconf import OmegaConf as om
from config import SourceType, JointDistributionType, MaskDistributionType, DataType
import networkx as nx
import numpy as np
from itertools import product
from abc import abstractmethod
import pandas as pd
import torch
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

class BaseGenerator:
    def __init__(self, config):
        self.config = config
        self.n_features = config.n_features
    @abstractmethod
    def sample_joint(self, N):
        pass
    @abstractmethod 
    def save(self, N, path):
        X = self.sample_joint(N)
        pd.DataFrame(X).to_csv(path, index =  False)
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
    def __init__(self,  config):
        self.n_features = config.n_features
        self.max_parents = config.source.max_parents
        self.G = self.create_random_dag(config.source.edge_probability)
        self.assign_cpt(self.G)
        self._cp_cache = {}
        self.bn = self.networkx_to_pgmpy()
        self.infer = VariableElimination(self.bn)

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
        
        for i in range(self.n_features):
            num_edges_i = 0
            for j in range(i+1, self.n_features):
                if np.random.random() < edge_probability:
                    G.add_edge(i, j)
                num_edges_i += 1
                if num_edges_i >= self.max_parents:
                    break
    
        return G

    def assign_cpt(self, G):
        """Assign conditional probability tables (CPTs) to each node."""
        for node in G.nodes():
            parents = list(G.predecessors(node))
            n_parents = len(parents)
            
            # Create CPT
            cpt_shape = [2] * (n_parents + 1)
            cpt = np.random.dirichlet(np.ones(2), size=cpt_shape[:-1]).reshape(cpt_shape)
            # print(cpt.shape)
            print(node, cpt.sum(axis = -1), file= open('cpt.txt', 'a'))
            # exit()
            # Assign CPT to node
            G.nodes[node]['cpt'] = cpt
            G.nodes[node]['parents'] = parents
    
    def reset_cp_cache(self):
        self._cp_cache = {}
    def _conditional_probability(self, i, S, evidence):
        evidence = {str(k): v for k, v in evidence.items()}
        key = (str(i), tuple(evidence.items()))
        if key not in self._cp_cache:   
            self._cp_cache[key] =  self.infer.query([str(i)], evidence = evidence)
        return self._cp_cache[key]
    
    def sample_joint(self, n_samples):
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
            i = [k for k in range(self.n_features) if I[j, k] == 1][0]
            s = {k for k in range(self.n_features) if S[j, k] == 1}
            assert i not in s
            result = self._conditional_probability(i, s, evidence = {k: X[j, k].cpu().item() for k in s})
            
            P[j] = result.values[int(X[j, i].item())]
        return P

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.n_features = self.config.n_features
        if self.config.source.type == SourceType.samples:
            self.sampler = DataFromFile(self.config)
        elif self.config.source.type == SourceType.distribution:
            self.sampler = DataFromDistribution(self.config)
        elif self.config.source.type == SourceType.dag:
            self.sampler = DataFromDAG(self.config)
        else:
            raise ValueError(f'Unknown source type: {self.config.source.type}')
    def sample_conditional(self, N):  
        X = self.sampler.sample_joint(N)
        S = np.zeros((N, self.n_features))
        
        Xi = np.zeros(N)
        Ii = np.zeros(N)
        if self.config.mask.distribution == MaskDistributionType.truncnorm:
            mean = self.config.mask.params.mean
            std = self.config.mask.params.std
            masksize = np.random.normal(mean, std, size = (N,))
        elif self.config.mask.distribution == MaskDistributionType.uniform:
            assert self.config.mask.params.min >= 0 and self.config.mask.params.max <= 1 and self.config.mask.params.min < self.config.mask.params.max
            minms = max(int(self.config.mask.params.min * self.n_features), 1)
            maxms = int(self.config.mask.params.max * self.n_features) + 1
            masksize = np.random.uniform(minms, maxms, size = (N,))
        elif self.config.mask.distribution == MaskDistributionType.delta:
            masksize = np.ones(N) * self.config.mask.params.value
        masksize = np.clip(masksize, 1, self.n_features)
        for i in range(N):
            indices = np.random.choice(np.arange(self.n_features), int(masksize[i]), replace=False)
            index_i = indices[0]
            indices_s = indices[1:]
            S[i, indices_s] = 1
            Xi[i] = X[i, index_i]
            Ii[i] = index_i
        Ii = np.eye(self.n_features)[Ii.astype(int)]
        return X, Xi, Ii, S
    
    def sample_marginal(self, N):
        X = self.sample_joint(N)
        S = np.zeros((N, self.n_features))
        if self.config.data.mask.distribution == MaskDistributionType.truncnorm:
            mean = self.config.data.mask.params.mean
            std = self.config.data.mask.params.std
            masksize = np.random.normal(mean, std, size = (N,))
        elif self.config.data.mask.distribution == MaskDistributionType.uniform:
            minms = max(int(self.config.mask.params.min * self.n_features) - 1, 0)
            maxms = min(int(self.config.mask.params.max * self.n_features), self.n_features)
            masksize = np.random.uniform(minms, maxms, size = (N,))
        elif self.config.data.mask.distribution == MaskDistributionType.delta:
            masksize = np.ones(N) * self.config.data.mask.params.value
        for i in range(N):
            masksize = np.clip(masksize, 0, self.n_features - 1)
            indices = np.random.choice(np.arange(self.n_features), int(masksize[i]), replace=False)
            S[i, indices] = 1
        return X, S

    def estimate_prob(self, X, S, N):
        P = np.zeros(X.shape[0], self.n_features)
        Xj = self.sampler.sample_joint(N)
        if self.config.type == DataType.binary:
            for i in range(X.shape[0]):
                s = S[i] 
                indices = (Xj[:, s] == X[i, s]).all(dim = 1)
                if indices.sum() == 0:
                    P[i] = 0.5
                else:
                    P[i] = Xj[indices].mean().cpu().numpy()
        return P

if __name__ == '__main__':
    from omegaconf import OmegaConf as om
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument('--config', type = str, default='config_data.yaml')
    parser.add_argument('--n_samples', type = int, default=10000)
    parser.add_argument('--path', type = str, default='data.csv')
    parser.add_argument('--nf', type = int, default=10)
    parser.add_argument('--ep', type = float, default=0.2)
    parser.add_argument('--configs', type = str, default='datasets/info.csv')
    parser.add_argument('--multiple', action='store_true', default=False)
    args = parser.parse_args()
    # config = om.load(args.config)
    if args.multiple:
        configs = pd.read_csv(args.configs)
        for i in range(len(configs)):
            config = configs.iloc[i]
            data_generator = DataGenerator(om.create({'n_features': int(config['n_features']), 'source': {'type': 'dag', 'edge_probability': float(config['edge_probability'])}}))
            data_generator.sampler.save(args.n_samples, 'datasets/'+config['path'])
    else:
        config = om.create({'n_features': args.nf, 'source': {'type': 'dag', 'edge_probability': args.ep}})
        data_generator = DataGenerator(config)
        args.path = 'datasets/' + args.path
        data_generator.sampler.save(args.n_samples, args.path)