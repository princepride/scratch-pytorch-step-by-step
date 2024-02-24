from .tensor import Tensor
from typing import Tuple
import numpy as np

def cat(tensors: Tuple[Tensor, ...], dim: int = 0) -> Tensor:
    return Tensor.cat(tensors, dim)

def tanh(tensor:Tensor) -> Tensor:
    return tensor.tanh()

def from_numpy(ndarray: np.ndarray) -> Tensor:
    return Tensor.from_numpy(ndarray)