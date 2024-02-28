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

def test_size():
    a = Tensor([[[1],[2]],[[3],[3]],[[4],[5]]])
    assert a.size() == (3,2,1)
    assert a.size(1) == 2
    assert a.size(-2) == 2
    assert a.size(-3) == 3
    with pytest.raises(TypeError):
        a.size(1.)
    with pytest.raises(IndexError):
        a.size(-4)

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

    with pytest.raises(TypeError) as e:
        Tensor([1,2,3,4,5,6]).reshape([1,2,3])
    assert str(e.value) == "new_shape 必须是一个元组(tuple)。"

    with pytest.raises(TypeError) as e:
        Tensor([1,2,3,4,5,6]).reshape((1.,2.,3.))
    assert str(e.value) == "new_shape 中的所有元素都必须是整型(int)。"

def test_squeeze():
    a = Tensor([[[1, 2, 3]]])
    b = a.squeeze()
    assert b.size() == (3,)

    c = Tensor([[[1],[2]],[[3],[3]],[[4],[5]]])
    d = c.squeeze(axis=2)
    assert d.size() == (3, 2)
    assert c.squeeze(-1).size() == (3, 2)

    e = Tensor([1, 2, 3])
    f = e.squeeze()
    assert f.size() == (3,)

    g = Tensor([[[[1]]]])
    h = g.squeeze()
    assert h.size() == ()

    with pytest.raises(ValueError):
        i = Tensor([[1, 2, 3]])
        i.squeeze(axis=1)

    with pytest.raises(ValueError):
        j = Tensor([1, 2, 3])
        j.squeeze(axis=3)

    with pytest.raises(TypeError):
        k = Tensor([1, 2, 3])
        k.squeeze(axis='0')

    with pytest.raises(ValueError):
        l = Tensor([1, 2, 3])
        l.squeeze(axis=-2)

def test_permute():
    # 正确的使用场景
    a = Tensor([[1, 2], [3, 4]])
    b = a.permute(1, 0)
    assert b == Tensor([[1, 3], [2, 4]])
    
    # 测试维度数不匹配
    with pytest.raises(ValueError):
        a.permute(0)
    
    # 测试维度超出范围
    with pytest.raises(ValueError):
        a.permute(0, 2)
    
    # 测试维度数据类型不正确
    with pytest.raises(TypeError):
        a.permute(0, '1')  # '1' 不是整型
    
    # 使用负数维度
    c = Tensor([[1, 2, 3], [4, 5, 6]])
    d = c.permute(-1, -2)
    assert d == Tensor([[1, 4], [2, 5], [3, 6]])
    
    # 测试梯度反向传播
    d.grad = np.array([[1, 2], [3, 4], [5, 6]])
    d._backward()
    assert np.array_equal(c.grad, np.array([[1, 3, 5], [2, 4, 6]]))

def test_transpose():
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = a.transpose(0, 1)
    assert b == Tensor([[1, 4], [2, 5], [3, 6]])
    
    with pytest.raises(TypeError):
        a.transpose('0', 1)  # 测试参数类型错误
    
    with pytest.raises(ValueError):
        a.transpose(0, 2)  # 测试维度范围错误

    # 测试负索引
    c = a.transpose(-1, -2)
    assert c == Tensor([[1, 4], [2, 5], [3, 6]])
    
    # 测试梯度反向传播
    c.grad = np.array([[1, 2], [3, 4], [5, 6]])
    c._backward()
    assert np.array_equal(a.grad, np.array([[1, 3, 5], [2, 4, 6]]))

def test_cat():
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])

    p = Tensor([1,2,3,4])
    q = Tensor([5,6,7,8])
    assert Tensor.cat([p, q], dim=0) == Tensor([1,2,3,4,5,6,7,8])
    
    # 正确的拼接
    c = Tensor.cat([a, b], dim=0)
    assert c == Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    
    # 测试维度不匹配
    d = Tensor([[9, 10, 11]])
    with pytest.raises(ValueError):
        Tensor.cat([a, d], dim=0)
    
    # 测试拼接维度超出范围
    with pytest.raises(ValueError):
        Tensor.cat([a, b], dim=3)
    
    # 测试空列表
    with pytest.raises(ValueError):
        Tensor.cat([], dim=0)
    
    # 测试梯度反向传播
    c.grad = np.array([[2, 1], [4, 3], [6, 5], [8, 7]])
    c._backward()
    assert np.array_equal(a.grad, [[2, 1], [4, 3]])
    assert np.array_equal(b.grad, [[6, 5], [8, 7]])

def test_add():
    # 测试Tensor和Tensor相加
    a = Tensor(np.array([1, 2, 3], dtype=np.float32))
    b = Tensor(np.array([4, 5, 6], dtype=np.float32))
    c = a + b
    assert np.array_equal(c.data, np.array([5, 7, 9]))

    # 测试Tensor和标量相加
    d = a + 1
    assert np.array_equal(d.data, np.array([2, 3, 4]))

    # 测试Tensor和NumPy数组相加
    e = a + np.array([1, 1, 1], dtype=np.float32)
    assert e == Tensor([2, 3, 4])

    # 测试形状不匹配
    with pytest.raises(ValueError):
        f = a + Tensor(np.array([1, 2]))

    # 测试梯度反向传播
    g = Tensor(np.array([1, 2, 3], dtype=np.float32))
    h = Tensor(np.array([4, 5, 6], dtype=np.float32))
    i = g + h
    i.grad = np.array([1, 2, 3], dtype=np.float32)  # 假设最终梯度为1
    i._backward()
    assert np.array_equal(g.grad, np.array([1, 2, 3]))
    assert np.array_equal(h.grad, np.array([1, 2, 3]))

    # 测试与不支持的类型相加
    with pytest.raises(TypeError):
        _ = a + "string"

def test_radd():
    # 创建一个Tensor对象
    a = Tensor([1, 2, 3])
    
    # 标量在左侧时的加法
    result = 10 + a
    assert result == Tensor([11, 12, 13])
    
    # 测试梯度反向传播
    result.grad = np.array([1, 1, 1])
    result._backward()
    assert np.array_equal(a.grad, np.array([1, 1, 1]))
    
    # 测试与np.ndarray的加法
    b = np.array([10, 20, 30])
    result = b + a
    print(result)
    print(a+b)
    assert result == Tensor([11, 22, 33])
    
    # 测试与不支持的类型相加
    with pytest.raises(TypeError):
        _ = "string" + a
