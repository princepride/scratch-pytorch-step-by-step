from mytorch.tensor import Tensor
import math
import numpy as np
import pytest

def test_init():
    assert Tensor(5) == Tensor(5.0)
    assert Tensor([1,2,3]) == Tensor(np.array([1,2,3]), trainable=True)
    assert Tensor([1,2,3], trainable=False) == Tensor(np.array([1,2,3]), trainable=False)
    assert (Tensor([[1,2,3],[4,5,6]]).grad == np.array([[0,0,0],[0,0,0]])).all()

    with pytest.raises(TypeError) as e:
        Tensor(Tensor(5))
    assert str(e.value) == "Tensor被用于初始化的数据类型不能是Tensor类型"

    with pytest.raises(TypeError) as e:
        Tensor((5,6,7))
    assert str(e.value) == "未知的初始化数据类型, Tensor类只可用int, float, List以及np.ndarray进行初始化"

    with pytest.raises(TypeError) as e:
        Tensor({'a':'123','b':'12'})
    assert str(e.value) == "未知的初始化数据类型, Tensor类只可用int, float, List以及np.ndarray进行初始化"

def test_from_numpy():
    assert Tensor.from_numpy(np.array(5)) == Tensor(5, trainable=True)
    assert Tensor.from_numpy(np.array(5)) == Tensor(5, trainable=True)
    assert Tensor.from_numpy(np.array([1,2,3]), trainable=False) == Tensor([1,2,3], trainable=False)
    assert Tensor.from_numpy(np.array([[1,2,3],[4,5,6]])) == Tensor([[1,2,3],[4,5,6]], trainable=True)
    assert Tensor.from_numpy(np.array([1,2,3])) != Tensor([1,2,3], trainable=False)
    assert Tensor.from_numpy(np.array([1,2,3]), trainable=False) != Tensor([1,2,3])

def test_unsqueeze():
    assert Tensor(5).unsqueeze(0) == Tensor([5])
    assert Tensor([5,4,5]).unsqueeze(0) == Tensor([[5,4,5]])
    assert Tensor([1,2,3]).unsqueeze(1) == Tensor([[1],[2],[3]])

    a = Tensor([[1,2,3],[4,5,6]])
    b = a.unsqueeze(1)
    print(b)
    assert b == Tensor([[[1,2,3]],[[4,5,6]]])
    b.grad = np.array([[[[9,8,7]]],[[[6,5,4]]]])
    b._backward()
    assert (a.grad == np.array([[9,8,7],[6,5,4]])).all()

def test_cat():
    a = Tensor([5])
    b = Tensor([6])
    c = Tensor([[1,2]])
    d = Tensor([4,5,6])