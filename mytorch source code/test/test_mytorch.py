import mytorch
from mytorch.tensor import Tensor
import pytest
import numpy as np

def test_cat():
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])

    p = Tensor([1,2,3,4])
    q = Tensor([5,6,7,8])
    assert mytorch.cat([p, q], dim=0) == Tensor([1,2,3,4,5,6,7,8])

    # 正确的拼接
    c = mytorch.cat([a, b], dim=0)
    assert c == Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    
    # 测试维度不匹配
    d = Tensor([[9, 10, 11]])
    with pytest.raises(ValueError):
        mytorch.cat([a, d], dim=0)
    
    # 测试拼接维度超出范围
    with pytest.raises(ValueError):
        mytorch.cat([a, b], dim=3)
    
    # 测试空列表
    with pytest.raises(ValueError):
        mytorch.cat([], dim=0)
    
    # 测试梯度反向传播
    c.grad = np.array([[2, 1], [4, 3], [6, 5], [8, 7]])
    c._backward()
    assert np.array_equal(a.grad, [[2, 1], [4, 3]])
    assert np.array_equal(b.grad, [[6, 5], [8, 7]])