from enum import Enum
class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"
    
import torch
def stable_softmax(X):
    all_neg_inf = torch.all(X == float("-inf"), dim = -1)
    max_ = torch.max(X, dim = -1)[0]
    max_score = torch.where(all_neg_inf, 0, max_)
    exp_scores = torch.exp(X - max_score.unsqueeze(-1))
    sum_scores = exp_scores.sum(dim = -1, keepdim=True)
    sum_scores = torch.where(sum_scores == 0, 1, sum_scores)
    return exp_scores / sum_scores

