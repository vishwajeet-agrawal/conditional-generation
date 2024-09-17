import numpy as np
from omegaconf import OmegaConf as om
from config import SourceType, JointDistributionType, MaskDistributionType
class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.n_features = self.config.get('n_features')
        if self.config.source.type == SourceType.samples:
            self.samples = np.load(self.config.source.path)
        else:
            self.samples = None
    def sample_joint(self, N):
        if self.config.data.source.type == SourceType.samples:
            idx = np.random.choice(len(self.samples), N)
            return self.samples[idx]            
        elif self.config.data.source.type == SourceType.distribution:
            if self.config.data.source.distribution == JointDistributionType.normal:
                return np.random.normal(size=(N, self.n_features))
            elif self.config.data.source.distribution == JointDistributionType.uniform:
                return np.random.uniform(size=(N, self.n_features))
            else:
                raise ValueError(f'Unknown distribution type: {self.config.data.source.distribution}')
        else:
            raise ValueError(f'Unknown data source type: {self.config.data.source.type}')
    
    def sample_conditional(self, N):  
        X = self.sample_joint(N)
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
