from .modules import Module, Linear, MSELoss
from .functional import functional
from ..tensor import Tensor

def ReLU(tensor: Tensor) -> Tensor:
    return tensor.relu()