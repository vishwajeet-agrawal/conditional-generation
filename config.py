from util import StrEnum
class SourceType(StrEnum):
    samples = 'samples'
    distribution = 'distribution'
    dag = 'dag'
    
class JointDistributionType(StrEnum):
    normal = 'normal'
    uniform = 'uniform'
class MaskDistributionType(StrEnum):
    truncnorm = 'truncnorm'
    delta = 'delta'
    uniform = 'uniform'

class ScoreFunctionType(StrEnum):
    exp = 'exp'
class DataType(StrEnum):
    binary = 'binary'
    continuous = 'continuous'

class ContextAggregatorType(StrEnum):
    sum = 'sum'
    avg = 'avg'
    linear = 'linear'
    attention = 'attention' 
    transformer = 'transformer'
    mlp = 'mlp'

class AggregatorReduceType(StrEnum):
    sum = "sum"
    mean = "mean"