from mytorch.tensor import Tensor
import torch
import torch.nn.functional as F
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

    with pytest.raises(TypeError):
        Tensor((5,6,7))

    with pytest.raises(TypeError):
        Tensor({'a':'123','b':'12'})

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

def test_sub():
    # 创建两个Tensor对象
    a = Tensor([4, 5, 6])
    b = Tensor([1, 2, 3])

    # Tensor与Tensor相减
    result = a - b
    assert result == Tensor([3, 3, 3])

    # Tensor与标量相减
    scalar_subtraction = a - 1
    assert scalar_subtraction == Tensor([3, 4, 5])

    # 测试梯度反向传播
    result.grad = np.array([1, 2, 3], dtype=np.float32)
    result._backward()
    assert np.array_equal(a.grad, np.array([1, 2, 3]))
    assert np.array_equal(b.grad, np.array([-1, -2, -3]))

    # 测试与np.ndarray相减
    c = np.array([2, 4, 6], dtype=np.float32)
    ndarray_subtraction = a - c
    assert ndarray_subtraction == Tensor([2, 1, 0])

    # 测试形状不匹配的Tensor相减
    d = Tensor(np.array([10, 20], dtype=np.float32))
    with pytest.raises(ValueError):
        _ = a - d

    # 测试与不支持的类型相减
    with pytest.raises(TypeError):
        _ = a - "string"

def test_neg():
    # 创建一个Tensor对象
    a = Tensor([[1, -2, 3]])
    
    # 对Tensor应用负号操作
    result = -a
    assert result == Tensor([[-1, 2, -3]])
    
    # 测试梯度反向传播
    result.grad = np.array([[1, 1, 1]])
    result._backward()
    assert np.array_equal(a.grad, np.array([[-1, -1, -1]]))

def test_mul():
    # 创建两个Tensor对象
    a = Tensor([4, 5, 6])
    b = Tensor([1, 2, 3])

    # Tensor与Tensor相乘
    result = a * b
    assert result == Tensor([4, 10, 18])

    e = Tensor([[1,2,3],[3,2,1]])
    f = Tensor([[2,3,1],[1,3,2]])
    assert e * f == Tensor([[2,6,3],[3,6,2]])

    # Tensor与标量相乘
    scalar_multiplication = a * 2
    assert scalar_multiplication == Tensor([8, 10, 12])

    # 测试梯度反向传播
    result.grad = np.array([1, 1, 1], dtype=np.float32)
    result._backward()
    assert np.array_equal(a.grad, np.array([1, 2, 3]))
    assert np.array_equal(b.grad, np.array([4, 5, 6]))

    # 测试与np.ndarray相乘
    c = np.array([2, 4, 6], dtype=np.float32)
    ndarray_multiplication = a * c
    assert ndarray_multiplication == Tensor([8, 20, 36])

    # 测试形状不匹配的Tensor相乘
    d = Tensor(np.array([10, 20], dtype=np.float32))
    with pytest.raises(ValueError):
        _ = a * d

    # 测试与不支持的类型相乘
    with pytest.raises(TypeError):
        _ = a * "string"

def test_rmul():
    # 创建一个Tensor对象
    a = Tensor([4, 5, 6])

    # 标量在左侧时的乘法
    result = 3 * a
    assert result == Tensor([12, 15, 18])

    # 测试梯度反向传播
    result.grad = np.array([1, 1, 1], dtype=np.float32)
    result._backward()
    assert np.array_equal(a.grad, np.array([3, 3, 3 ]))

    # 测试与np.ndarray相乘
    c = np.array([2, 3, 4], dtype=np.float32)
    result = c * a  # 这应该调用 a.__rmul__(c)
    assert result == Tensor([8, 15, 24])

def test_matmul():
    # 向量与矩阵相乘
    v = Tensor([1, 2])
    m = Tensor([[1, 2], [3, 4]])
    vm_result = v.matmul(m)
    assert vm_result == Tensor([7, 10])

    # 矩阵与向量相乘
    mv_result = m.matmul(v)
    assert mv_result == Tensor([5, 11])

    # 矩阵与矩阵相乘
    m2 = Tensor([[2, 0], [0, 2]])
    mm_result = m.matmul(m2)
    assert mm_result == Tensor([[2, 4], [6, 8]])

    # 测试梯度反向传播
    mm_result.grad = np.ones(mm_result.data.shape, dtype=np.float32)
    mm_result._backward()
    assert np.array_equal(m.grad, np.array([[2, 2], [2, 2]]))

    # 矩阵与矩阵相乘的反向传播
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    result = a.matmul(b)
    result.grad = np.array([[1, 0], [0, 1]], dtype=np.float32)  # 模拟梯度
    result._backward()
    assert np.array_equal(a.grad, np.array([[5, 7], [6, 8]]))
    assert np.array_equal(b.grad, np.array([[1, 3], [2, 4]]))

    # 批量矩阵乘法的反向传播
    a = Tensor(np.random.rand(2, 3, 4))
    b = Tensor(np.random.rand(2, 4, 5))
    result = a.matmul(b)
    result.grad = np.ones(result.data.shape, dtype=np.float32)  # 假设梯度全为1
    result._backward()
    assert a.grad.shape == a.data.shape
    assert b.grad.shape == b.data.shape

    # 向量与矩阵相乘的反向传播
    a = Tensor([1, 2, 3],)
    b = Tensor([[1, 2], [3, 4], [5, 6]])
    result = a.matmul(b)
    result.grad = np.array([1, 1], dtype=np.float32)
    result._backward()
    assert np.array_equal(a.grad, np.array([3, 7, 11]))
    assert np.array_equal(b.grad, np.array([[1, 1], [2, 2], [3, 3]]))

    # 矩阵与向量相乘的反向传播
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([1, 2])
    result = a.matmul(b)
    result.grad = np.array([1, 1], dtype=np.float32)
    result._backward()
    assert np.array_equal(a.grad, np.array([[1, 2], [1, 2]]))
    assert np.array_equal(b.grad, np.array([4, 6]))

    # 批量矩阵乘法
    b1 = Tensor(np.random.rand(2, 3, 4))
    b2 = Tensor(np.random.rand(2, 4, 5))
    batch_result = b1.matmul(b2)
    assert batch_result.data.shape == (2, 3, 5)

    # 测试形状不匹配的张量相乘
    m3 = Tensor([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError):
        _ = m.matmul(m3)

    # 测试与不支持的类型相乘
    with pytest.raises(TypeError):
        _ = m.matmul("string")

def test_truediv():
    # 创建Tensor对象
    a = Tensor([10.0, 20.0, 30.0])
    b = Tensor([2.0, 4.0, 6.0])
    c = Tensor([0.5, 0.25, 0.125])

    # Tensor与Tensor相除
    result_ab = a / b
    assert np.allclose(result_ab.data, np.array([5.0, 5.0, 5.0]))

    # Tensor与标量相除
    result_a2 = a / 2.0
    assert np.allclose(result_a2.data, np.array([5.0, 10.0, 15.0]))

    # 反向传播测试：Tensor与Tensor相除
    result_ab.grad = np.ones_like(result_ab.data, dtype=np.float32)
    result_ab._backward()
    assert np.allclose(a.grad, 1 / b.data)
    assert np.allclose(b.grad, -a.data / np.square(b.data))

    # 手动重置梯度进行下一组测试
    a.grad = np.zeros_like(a.data, dtype=np.float32)
    b.grad = np.zeros_like(b.data, dtype=np.float32)

    # 反向传播测试：Tensor与标量相除
    result_a2.grad = np.array([1, 2, 3], dtype=np.float32)
    result_a2._backward()
    assert np.allclose(a.grad, np.array([0.5, 1.0, 1.5]))

    # 手动重置梯度进行更复杂的测试
    a.grad = np.zeros_like(a.data, dtype=np.float32)
    b.grad = np.zeros_like(b.data, dtype=np.float32)
    c.grad = np.zeros_like(c.data, dtype=np.float32)

    # 更复杂的反向传播测试：链式除法
    result_abc = a / b / c
    result_abc.grad = np.array([1, 1, 1], dtype=np.float32)
    result_abc._backward()
    # 预期梯度计算涉及更复杂的链式规则，这里仅验证是否执行无误
    assert a.grad is not None
    assert b.grad is not None
    assert c.grad is not None

    # 测试与不支持的类型相除
    with pytest.raises(TypeError):
        _ = a / "string"

    # 测试形状不匹配的Tensor相除
    d = Tensor(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(ValueError):
        _ = a / d

def test_pow():
    # 创建一个Tensor对象
    a = Tensor([1, 2, 3])

    # Tensor的幂运算
    result = a ** 2
    assert np.array_equal(result.data, np.array([1, 4, 9]))

    # 测试梯度反向传播
    result.grad = np.ones_like(result.data, dtype=np.float32)
    result._backward()
    assert np.array_equal(a.grad, np.array([2, 4, 6]))

    # 测试与不合法的幂指数
    with pytest.raises(TypeError):
        _ = a ** "2"

    # 更复杂的幂运算测试
    b = Tensor([4, 5, 6])
    result_b = b ** 0.5  # 平方根
    assert np.allclose(result_b.data, np.sqrt(np.array([4, 5, 6])))

    # 反向传播测试：更复杂的幂运算
    result_b.grad = np.ones_like(result_b.data, dtype=np.float32)
    result_b._backward()
    assert np.allclose(b.grad, 1 / (2 * np.sqrt(b.data)))

def test_dropout():
    # 创建一个Tensor对象
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    a = Tensor(data)
    
    # 训练模式下的dropout
    dropout_a = a.dropout(dropout_probability=0.5, is_training=True)
    # 检查dropout是否应用（部分元素应该为0）
    assert np.any(dropout_a.data == 0)
    
    # 非训练模式下的dropout
    dropout_a_not_training = a.dropout(dropout_probability=0.5, is_training=False)
    # 检查Tensor是否未被修改
    assert np.array_equal(dropout_a_not_training.data, data)
    
    # 反向传播测试
    # 假设dropout后的Tensor对象可以计算梯度并调用_backward()
    dropout_a.grad = np.ones_like(dropout_a.data)
    dropout_a._backward()
    # 检查梯度是否仅在未被丢弃的元素上更新
    assert np.all((a.grad == 0) | (a.grad == 1 / (1 - 0.5)))

def test_tanh():
    test_cases = [
        5,
        np.array([0.0]),
        np.array([-1.0, 0.0, 1.0]),
        np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]),
    ]

    for data in test_cases:
        tensor = Tensor(data)
        result = tensor.tanh()

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
        result = tensor.sigmoid()

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
        result = tensor.relu()

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

def test_mean():
    np.random.seed(42)  # 为了可重复性
    data = np.random.rand(2, 3, 4)  # 随机生成一个3维数组

    # 使用自定义Tensor类计算全局平均值
    tensor = Tensor(data)
    mean_value_custom = tensor.mean()
    assert np.allclose(mean_value_custom.data, np.mean(data))

    # 使用PyTorch计算全局平均值
    tensor_torch = torch.tensor(data)
    mean_value_torch = tensor_torch.mean()
    assert np.allclose(mean_value_custom.data, mean_value_torch.numpy())

    # 测试在特定维度上求平均值，并保持维度
    mean_dim_custom = tensor.mean(dim=1, keepdims=True)
    mean_dim_torch = tensor_torch.mean(dim=1, keepdims=True)
    assert np.allclose(mean_dim_custom.data, mean_dim_torch.numpy())

    # 测试在特定维度上求平均值，不保持维度
    mean_dim_custom_no_keep = tensor.mean(dim=2, keepdims=False)
    mean_dim_torch_no_keep = tensor_torch.mean(dim=2, keepdims=False)
    assert np.allclose(mean_dim_custom_no_keep.data, mean_dim_torch_no_keep.numpy())

def test_backward():
    # 初始化数据和权重
    input_data = np.array([1.0, 2.0, -1.0], dtype=np.float32)
    weights = np.array([0.5, -0.5, 1.0], dtype=np.float32)
    
    # 自定义Tensor计算
    input_tensor = Tensor(input_data)
    weights_tensor = Tensor(weights)
    output_tensor = input_tensor * weights_tensor  # 简单的线性层
    sigmoid_output = output_tensor.sigmoid()  # Sigmoid激活函数
    sigmoid_output.backward()  # 反向传播
    
    # 使用PyTorch进行相同的计算
    input_torch = torch.tensor(input_data, requires_grad=True)
    weights_torch = torch.tensor(weights, requires_grad=True)
    output_torch = input_torch * weights_torch  # 线性层
    sigmoid_output_torch = torch.sigmoid(output_torch)  # Sigmoid激活
    sigmoid_output_torch.backward(torch.ones_like(sigmoid_output_torch))  # 反向传播
    
    # 比较输入和权重的梯度
    assert np.allclose(input_tensor.grad, input_torch.grad.numpy(), atol=1e-6)
    assert np.allclose(weights_tensor.grad, weights_torch.grad.numpy(), atol=1e-6)

    with pytest.raises(ValueError):
        sigmoid_output.backward([1.])
    
def test_gradient_descent_opt():
    # 初始化数据、权重和目标值
    input_data = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    weights = torch.tensor([0.5, 1., 1.5], requires_grad=True)
    target = torch.tensor([2.0, 3.0, 4.0])
    
    # 自定义Tensor计算
    input_tensor = Tensor([1.0, 2.0, 3.0])
    weights_tensor = Tensor([0.5, 1., 1.5])
    predicted = input_tensor * weights_tensor
    loss = (predicted - Tensor([2.0, 3.0, 4.0])) ** 2
    loss.backward()
    loss.gradient_descent_opt(learning_rate=0.01, grad_zero=True)
    
    # 使用PyTorch进行相同的计算
    predicted_torch = input_data * weights
    loss_torch = (predicted_torch - target) ** 2
    loss_torch.backward(torch.ones_like(loss_torch))
    with torch.no_grad():  # 禁用梯度计算
        weights -= 0.01 * weights.grad
        input_data -= 0.01 * input_data.grad
    # 重置梯度
    weights.grad.zero_()  
    input_data.grad.zero_()
    # 比较权重的更新结果
    assert np.allclose(weights_tensor.data, weights.detach().numpy(), atol=1e-6)
    assert np.allclose(weights_tensor.grad, weights.grad.numpy(), atol=1e-6)
    assert np.allclose(input_tensor.data, input_data.detach().numpy(), atol=1e-6)
    assert np.allclose(input_tensor.grad, input_data.grad.numpy(), atol=1e-6)

def test_adam_opt():
    # 初始化一些数据和参数
    data = np.random.randn(10, 3).astype(np.float32)
    target = np.random.randn(10, 1).astype(np.float32)
    
    # 自定义Tensor模型参数
    custom_param = Tensor(np.array([[2],[3],[4]]).astype(np.float32), trainable=True)
    
    # PyTorch模型参数
    torch_param = torch.tensor(np.array([[2],[3],[4]]).astype(np.float32), requires_grad=True)
    
    # 自定义Tensor的优化
    for _ in range(10):  # 假设我们迭代100次
        # 模拟一次前向传播和梯度计算
        custom_loss = ((Tensor(data, trainable=False).matmul(custom_param) - Tensor(target, trainable=False))**2).mean()
        custom_loss.backward()
        custom_loss.adam_opt(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, grad_zero=True)
    
    # PyTorch的优化
    torch_optimizer = torch.optim.Adam([torch_param], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    for _ in range(10):  # 同样迭代100次
        torch_optimizer.zero_grad()
        torch_loss = torch.mean((torch.matmul(torch.tensor(data), torch_param) - torch.tensor(target))**2)
        torch_loss.backward()
        torch_optimizer.step()

    # 比较参数更新结果
    assert np.allclose(custom_param.data, torch_param.detach().numpy(), atol=1e-2)
    # assert np.allclose(input_tensor.grad, input_torch.grad.numpy(), atol=1e-6)