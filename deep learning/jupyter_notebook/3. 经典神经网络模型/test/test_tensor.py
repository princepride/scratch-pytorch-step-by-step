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

def test_eq():
    # 测试数据相同的情况
    tensor1 = Tensor(np.array([1, 2, 3]), trainable=True)
    tensor2 = Tensor(np.array([1, 2, 3]), trainable=True)
    assert tensor1 == tensor2
    tensor3 = Tensor(np.array([1, 2, 3]), trainable=False)
    assert not tensor1 == tensor3
    tensor4 = Tensor(np.array([4, 5, 6]), trainable=True)
    assert not tensor1 == tensor4
    assert not tensor1 == [1, 2, 3]

def test_ne():
    tensor1 = Tensor(np.array([1, 2, 3]), trainable=True)
    tensor2 = Tensor(np.array([1, 2, 3]), trainable=False)
    tensor3 = Tensor(np.array([4, 5, 6]), trainable=True)
    assert tensor1 != tensor2
    assert tensor1 != tensor3
    tensor4 = Tensor(np.array([1, 2, 3]), trainable=True)
    assert not (tensor1 != tensor4)

def test_hash():
    tensor1 = Tensor(np.array([1, 2, 3]), trainable=True)
    tensor2 = Tensor(np.array([1, 2, 3]), trainable=True)
    tensor3 = Tensor(np.array([1, 2, 3]), trainable=False)

    assert hash(tensor1) != hash(tensor2)
    assert hash(tensor1) == hash(tensor1)
    assert hash(tensor1) != hash(tensor3)

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
    assert b == Tensor([[[1,2,3]],[[4,5,6]]])
    b.grad = np.array([[[[9,8,7]]],[[[6,5,4]]]])
    b._backward()
    assert (a.grad == np.array([[9,8,7],[6,5,4]])).all()
    with pytest.raises(TypeError) as e:
        Tensor(5).unsqueeze(0.9)
    assert str(e.value) == "在unsqueeze函数中, axis必须是一个整数"

    with pytest.raises(ValueError) as e:
        Tensor([[1],[2],[3]]).unsqueeze(3)
    assert str(e.value) == "axis 3 越界。有效范围是 [-3, 2]"


def test_reshape():
    assert Tensor([[1,2,3],[4,5,6]]).reshape((6,)) == Tensor([1,2,3,4,5,6])
    assert Tensor([[1,2,3],[4,5,6]]).reshape((3,2)) == Tensor([[1,2],[3,4],[5,6]])
    assert Tensor([1,2,3,4,5,6]).reshape((2,1,3)) == Tensor([[[1,2,3]],[[4,5,6]]])

    a = Tensor([[1,2],[3,4]])
    b = a.reshape((4,))
    assert b == Tensor([1,2,3,4])
    b.grad = np.array([5,6,7,8])
    b._backward()
    assert (a.grad == np.array([[5,6],[7,8]])).all()

    c = Tensor([[1,2,3],[3,4,5]])
    d = c.reshape((3,2,1))
    assert d == Tensor([[[1],[2]],[[3],[3]],[[4],[5]]])
    d.grad = np.array([[[1],[2]],[[3],[3]],[[4],[5]]])
    d._backward()
    assert (c.grad == np.array([[1,2,3],[3,4,5]])).all()

def test_cat():
    a = Tensor([5])
    b = Tensor([6])
    c = Tensor([[1,2]])
    d = Tensor([4,5,6])