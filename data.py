import numpy as np
from omegaconf import OmegaConf as om
from config import SourceType, JointDistributionType, MaskDistributionType, DataType
import networkx as nx
import numpy as np
from itertools import product
from abc import abstractmethod
import pandas as pd
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

class DataFromFile(BaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        if config.path.endswith('.csv'):
            self.samples = pd.read_csv(config.source.path).values
        elif config.path.endswith('.npy'):
            self.samples = np.load(config.source.path)
        else:
            raise ValueError(f'Unknown file type: {config.path.split(".")[-1]}')
        
    def sample_joint(self, N):
        idx = np.random.choice(len(self.samples), N)
        return self.samples[idx]

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
        self.G = self.create_random_dag(config.edge_probability)
        self.assign_cpt(self.G)

    def create_random_dag(self, edge_probability=0.3):
        """Create a random directed acyclic graph (DAG)."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_features))
        
        for i in range(self.n_features):
            for j in range(i+1, self.n_features):
                if np.random.random() < edge_probability:
                    G.add_edge(i, j)
        
        return G

    def assign_cpt(self, G):
        """Assign conditional probability tables (CPTs) to each node."""
        for node in G.nodes():
            parents = list(G.predecessors(node))
            n_parents = len(parents)
            
            # Create CPT
            cpt_shape = [2] * (n_parents + 1)
            cpt = np.random.dirichlet(np.ones(2), size=cpt_shape[:-1]).reshape(cpt_shape)
            
            # Assign CPT to node
            G.nodes[node]['cpt'] = cpt
            G.nodes[node]['parents'] = parents

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
        if self.config.data.mask.distribution == MaskDistributionType.truncnorm:
            mean = self.config.data.mask.params.mean
            std = self.config.data.mask.params.std
            masksize = np.random.normal(mean, std, size = (N,))
        elif self.config.data.mask.distribution == MaskDistributionType.uniform:
            minms = int(self.config.data.mask.params.min * self.n_features)
            maxms = int(self.config.data.mask.params.max * self.n_features)
            masksize = np.random.uniform(minms, maxms, size = (N,))
        elif self.config.data.mask.distribution == MaskDistributionType.delta:
            masksize = np.ones(N) * self.config.data.mask.params.value
        for i in range(N):
            masksize = np.clip(masksize, 1, self.n_features)
            indices = np.random.choice(np.arange(self.n_features), int(masksize[i]), replace=False)
            index_i = indices[0]
            indices_s = indices[1:]
            S[i, indices_s] = 1
            Xi[i] = X[i, index_i]
            Ii[i] = index_i
        return X, Xi, Ii, S
    
    def sample_marginal(self, N):
        X = self.sample_joint(N)
        S = np.zeros((N, self.n_features))
        if self.config.data.mask.distribution == MaskDistributionType.truncnorm:
            mean = self.config.data.mask.params.mean
            std = self.config.data.mask.params.std
            masksize = np.random.normal(mean, std, size = (N,))
        elif self.config.data.mask.distribution == MaskDistributionType.uniform:
            masksize = np.random.uniform(self.config.data.mask.params.min, self.config.data.mask.params.max, size = (N,))
        elif self.config.data.mask.distribution == MaskDistributionType.delta:
            masksize = np.ones(N) * self.config.data.mask.params.value
        for i in range(N):
            masksize = np.clip(masksize, 1, self.n_features)
            indices = np.random.choice(np.arange(self.n_features), int(masksize[i]), replace=False)
            S[i, indices] = 1
        return X, S


if __name__ == '__main__':
    from omegaconf import OmegaConf as om
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--config', type = str, default='config_data.yaml')
    parser.add_argument('--path', type = str, default='data.csv')
    args = parser.parse_args()
    config = om.load(args.config)
    data_generator = DataGenerator(config)
    data_generator.sampler.save(config.n_samples, args.path)