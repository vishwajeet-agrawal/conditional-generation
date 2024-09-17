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
def get_attn_mask(S, device):
    N = S.size(0)
    n = S.size(1)
    attn_mask = (1 - (S.unsqueeze(1) * S.unsqueeze(2))).to(torch.bool)
    attn_mask = torch.zeros_like(attn_mask, dtype=torch.float).masked_fill_(attn_mask, float("-inf"))
    return attn_mask