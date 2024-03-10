from .tensor import Tensor
from typing import Tuple
import numpy as np

def cat(tensors: Tuple[Tensor, ...], dim: int = 0) -> Tensor:
    return Tensor.cat(tensors, dim)

def tanh(tensor:Tensor) -> Tensor:
    return tensor.tanh()

def sigmoid(tensor:Tensor) -> Tensor:
    return tensor.sigmoid()

def relu(tensor:Tensor) -> Tensor:
    return tensor.relu()

def from_numpy(ndarray: np.ndarray, trainable=True) -> Tensor:
    return Tensor.from_numpy(ndarray, trainable=trainable)

def zeros(shape, trainable=True):
    return Tensor.from_numpy(np.zeros(shape, dtype=np.float32), trainable=trainable)