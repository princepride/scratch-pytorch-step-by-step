import mytorch
from mytorch.tensor import Tensor
import pytest
import numpy as np
import torch

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

def test_tanh():
    test_cases = [
        5,
        np.array([0.0]),
        np.array([-1.0, 0.0, 1.0]),
        np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]),
    ]

    for data in test_cases:
        tensor = Tensor(data)
        result = mytorch.tanh(tensor)

        torch_tensor = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        expected = torch.tanh(torch_tensor)

        # 检查前向传播的值是否接近
        assert np.allclose(result.data, expected.detach().numpy(), atol=1e-6)

        # 为了进行梯度检查，我们需要对result和expected进行反向传播
        result.grad = np.ones_like(result.data)
        result._backward()
        
        expected.backward(torch.ones_like(expected))
        
        # 检查梯度是否接近
        assert np.allclose(tensor.grad, torch_tensor.grad.detach().numpy(), atol=1e-6)

def test_sigmoid():
    test_cases = [
        5,
        np.array([0.0]),
        np.array([-1.0, 0.0, 1.0]),
        np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]),
    ]

    for data in test_cases:
        tensor = Tensor(data)
        result = mytorch.sigmoid(tensor)

        torch_tensor = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        expected = torch.sigmoid(torch_tensor)

        # 检查前向传播的值是否接近
        assert np.allclose(result.data, expected.detach().numpy(), atol=1e-6)

        # 检查梯度是否接近
        result.grad = np.ones_like(result.data)
        result._backward()

        expected.backward(torch.ones_like(expected))
        assert np.allclose(tensor.grad, torch_tensor.grad.detach().numpy(), atol=1e-6)
    
def test_relu():
    test_cases = [
        5,
        np.array([-1.0, 0.0, 1.0]),
        np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]]),
        np.array([0.0]),
    ]

    for data in test_cases:
        tensor = Tensor(data)
        result = mytorch.relu(tensor)

        torch_tensor = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        expected = torch.relu(torch_tensor)

        # 检查前向传播的值是否一致
        assert np.allclose(result.data, expected.detach().numpy())

        # 准备反向传播测试
        result.grad = np.ones_like(result.data)
        result._backward()

        expected.backward(torch.ones_like(expected))

        # 检查梯度是否一致
        assert np.allclose(tensor.grad, torch_tensor.grad.detach().numpy())

def test_from_numpy():
    assert mytorch.from_numpy(np.array(5)) == Tensor(5, trainable=True)
    assert mytorch.from_numpy(np.array(5)) == Tensor(5, trainable=True)
    assert mytorch.from_numpy(np.array([1,2,3]), trainable=False) == Tensor([1,2,3], trainable=False)
    assert mytorch.from_numpy(np.array([[1,2,3],[4,5,6]])) == Tensor([[1,2,3],[4,5,6]], trainable=True)
    assert mytorch.from_numpy(np.array([1,2,3])) != Tensor([1,2,3], trainable=False)
    assert mytorch.from_numpy(np.array([1,2,3]), trainable=False) != Tensor([1,2,3])

def test_zeros():
    assert mytorch.zeros((1, 5)) == Tensor.from_numpy([[0,0,0,0,0]])
    assert mytorch.zeros((3, ), False) == Tensor.from_numpy([0,0,0], False)