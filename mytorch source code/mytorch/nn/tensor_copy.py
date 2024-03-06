import math
import numpy as np
from typing import List
class Tensor:

    # 给予Tensor类更高的优先级
    __array_priority__ = 1.0
    def __init__(self, data, _prev=(), trainable=True, _op='', label=''):
        if isinstance(data, Tensor):
            raise TypeError("Tensor被用于初始化的数据类型不能是Tensor类型")
        elif np.isscalar(data):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, List):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            pass
        else:
            raise TypeError(f"未知的初始化数据类型, Tensor类只可用int, float, List以及np.ndarray进行初始化, 现在传入data的数据类型是{type(data)}")
        self.data = data
        self.grad = np.zeros_like(data, dtype=np.float32)
        self._backward = lambda: None
        self._prev = set(_prev)
        self.trainable = trainable
        self._op = _op
        self.label = label

        self.optim_step = 0  # 添加优化步骤计数器

    def __eq__(self, other):
        """
        比较当前Tensor和另一个Tensor的数据和训练标志。
        
        参数:
        other (Tensor): 要比较的另一个Tensor对象。
        
        返回:
        bool: 如果数据和训练标志相同，则为True；否则为False。
        """
        if not isinstance(other, Tensor):
            return NotImplemented
        return (np.array_equal(self.data, other.data) and 
                self.trainable == other.trainable and
                np.array_equal(self.grad, other.grad))
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    # 当我们重写__eq__方法而没有同时重写__hash__方法时，Python会默认将我们的对象视为不可哈希的。
    # 这是因为__eq__方法定义了对象相等性的行为，而哈希值(hash value)是用来在哈希表中快速比较键的一种机制。
    # 如果两个对象被视为相等（即它们的__eq__方法返回True），那么它们的哈希值也应该相同。
    # 为了保持一致性，当我们定义__eq__方法时，如果不定义__hash__方法，Python会将类的实例变为不可哈希的，
    # 从而防止了使用不一致的哈希值导致的潜在问题。
    def __hash__(self):
        return hash((id(self), self.trainable, self._op, self.label))
    
    @staticmethod
    def from_numpy(ndarray, trainable=True):
        if isinstance(ndarray, Tensor):
            ndarray.trainable = trainable
            return ndarray
        if isinstance(ndarray, np.ndarray):
            return Tensor(ndarray, trainable = trainable)
        if isinstance(ndarray, List):
            return Tensor(ndarray, trainable= trainable)
        raise TypeError("Input must be a NumPy array")
    
    def size(self, dim=None):
        if dim is None:
            return self.data.shape
        else:
            if not isinstance(dim, int):
                raise TypeError("dim 必须是整型（int）")
            if dim < -len(self.data.shape) or dim >= len(self.data.shape):
                raise IndexError("维度超出范围")
            return self.data.shape[dim]
        
    def unsqueeze(self, axis):
        if not isinstance(axis, int):
            raise TypeError("在unsqueeze函数中, axis必须是一个整数")
        
        if axis < -(self.data.ndim + 1) or axis > self.data.ndim:
            raise ValueError(f"axis {axis} 越界。有效范围是 [{-(self.data.ndim + 1)}, {self.data.ndim}]")
        
        expanded_data = np.expand_dims(self.data, axis=axis)  # 使用expand_dims来增加维度
        out = Tensor(expanded_data, _prev=(self,), _op='unsqueeze')
        
        def _backward():
            # 逆向传播时不需要修改梯度，因为增加的维度不影响原始数据的梯度
            self.grad += np.sum(out.grad, axis=axis).reshape(self.grad.shape)
        
        out._backward = _backward
        return out
    
    def reshape(self, new_shape):
        if not isinstance(new_shape, tuple):
            raise TypeError("new_shape 必须是一个元组(tuple)。")
        
        if not all(isinstance(dim, int) for dim in new_shape):
            raise TypeError("new_shape 中的所有元素都必须是整型(int)。")

        assert np.prod(new_shape) == np.prod(self.data.shape), "新形状的元素总数必须与原始形状相同"
        
        reshaped_data = self.data.reshape(new_shape)
        out = Tensor(reshaped_data, _prev=(self,), _op='reshape')
        
        def _backward():
            # reshape操作的梯度传递只涉及形状的变化，数据本身不改变
            self.grad += out.grad.reshape(self.data.shape)
            
        out._backward = _backward
        return out
    
    def squeeze(self, axis=None):
        if axis is not None:
            if not isinstance(axis, int):
                raise TypeError("axis 必须是一个整型(int)")
            if axis < -self.data.ndim or axis >= self.data.ndim:
                raise ValueError(f"axis 的值必须在{-self.data.ndim}到{self.data.ndim - 1}之间")
            
            # 如果指定了 axis，检查这个维度是否确实是单维的
            if self.data.shape[axis] != 1:
                raise ValueError(f"指定的 axis={axis} 维度不是单维的，不能被移除")
        if axis is None:
            squeezed_data = np.squeeze(self.data)
        else:
            squeezed_data = np.squeeze(self.data, axis=axis)
        out = Tensor(squeezed_data, _prev=(self,), _op='squeeze')
        
        def _backward():
            # 将梯度“扩展”回去，即在被压缩的维度上增加长度为1的维度
            grad_shape = list(self.data.shape)
            if axis is not None:
                grad_shape.insert(axis, 1)
            else:
                for ax in range(len(squeezed_data.shape), len(grad_shape)):
                    grad_shape.insert(ax, 1)
            self.grad += out.grad.reshape(grad_shape)
        
        out._backward = _backward
        return out
    
    def permute(self, *dims):
        if len(dims) != self.data.ndim:
            raise ValueError(f"permute 需要 {self.data.ndim} 维度的参数")
        
        if not all(isinstance(d, int) for d in dims):
            raise TypeError("dims 中的所有元素必须是整型(int)")
        
        if any(d < -self.data.ndim or d >= self.data.ndim for d in dims):
            raise ValueError(f"dims 中的维度超出了张量的维度范围 [-{self.data.ndim}, {self.data.ndim - 1}]")

        permuted_data = self.data.transpose(*dims)
        out = Tensor(permuted_data, _prev=(self,), _op='permute')
        
        def _backward():
            # 计算逆排列
            inv_dims = np.argsort(dims)
            self.grad += out.grad.transpose(*inv_dims)
            
        out._backward = _backward
        return out

    def transpose(self, dim0, dim1):
        if not isinstance(dim0, int) or not isinstance(dim1, int):
            raise TypeError("dim0 和 dim1 必须是整数")
        if dim0 < -self.data.ndim or dim0 >= self.data.ndim or dim1 < -self.data.ndim or dim1 >= self.data.ndim:
            raise ValueError("dim0 或 dim1 超出了张量的维度范围")

        dim0 = dim0 if dim0 >= 0 else self.data.ndim + dim0
        dim1 = dim1 if dim1 >= 0 else self.data.ndim + dim1

        dims = list(range(self.data.ndim))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        transposed_data = self.data.transpose(*dims)
        out = Tensor(transposed_data, _prev=(self,), _op='transpose')
        
        def _backward():
            self.grad += out.grad.transpose(*dims)
            
        out._backward = _backward
        return out

        
    @staticmethod
    def cat(tensors, dim=0):
        assert all(isinstance(t, Tensor) for t in tensors), "所有输入必须是Tensor对象"
        if not tensors:
            raise ValueError("tensors 不能为空列表")
        
        # 检查dim参数是否合法
        ndim = tensors[0].data.ndim
        if dim < 0:
            dim += ndim
        if not 0 <= dim < ndim:
            raise ValueError(f"dim 参数超出范围，接受的范围是0到{ndim-1}，或者对应的负数索引")
        
        # 检查所有Tensor是否可以在指定维度上拼接
        reference_shape = list(tensors[0].data.shape)
        for t in tensors[1:]:
            t_shape = list(t.data.shape)
            if len(t_shape) != len(reference_shape):
                raise ValueError("所有Tensor的维度数必须相同")
            t_shape[dim] = reference_shape[dim]  # 忽略拼接维度
            if t_shape != reference_shape:
                raise ValueError("所有Tensor在非拼接维度上的大小必须相同")

        data = [t.data for t in tensors]
        concatenated_data = np.concatenate(data, axis=dim)
        
        out = Tensor(concatenated_data, _prev=tuple(tensors), _op='cat')
        
        def _backward():
            grad_splits = np.split(out.grad, np.cumsum([t.data.shape[dim] for t in tensors[:-1]]), axis=dim)
            for t, grad in zip(tensors, grad_splits):
                t.grad += grad
                
        out._backward = _backward
        return out

    def __add__(self, other):
        return self.add(other)

    def add(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(other * np.ones_like(self.data).astype(np.float32), trainable=False)
        elif isinstance(other, np.ndarray):
            other = Tensor(other.astype(np.float32), trainable=False)
        elif isinstance(other, Tensor):
            pass
        else:
            raise TypeError("不支持的数据类型,只支持int,float,np.narray,Tensor数据类型")
        if self.data.shape != other.data.shape:
            raise ValueError("Tensor加法运算形状不匹配：{} 和 {}".format(self.data.shape, other.data.shape))

        out = Tensor(self.data + other.data, (self, other), _op='+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return self.sub(other)

    def sub(self, other):
        if isinstance(other, Tensor):
            if self.data.shape != other.data.shape:
                raise ValueError("形状不匹配：{} 和 {}".format(self.data.shape, other.data.shape))
        elif isinstance(other, (int, float)):
            other = Tensor(other * np.ones_like(self.data).astype(np.float32), trainable=False)
        elif isinstance(other, np.ndarray):
            other = Tensor(other.astype(np.float32), trainable=False)
        else:
            raise TypeError("不支持的数据类型，减法运算只支持 Tensor、int、float 或 np.ndarray 类型")

        out = Tensor(self.data - other.data, (self, other), _op='-')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad -= 1.0 * out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other):
        return (-self).__add__(other)
    
    # 必须重写负号，否则(-self)会报错
    def __neg__(self):
        neg_tensor = Tensor(-self.data, (self,))
        def _backward():
            self.grad -= 1.0 * neg_tensor.grad  
        neg_tensor._backward = _backward
        return neg_tensor
    
    def __mul__(self, other):
        return self.mul(other)

    def mul(self, other):
        if isinstance(other, Tensor):
            if self.data.shape != other.data.shape:
                raise ValueError("元素级乘法要求两个张量形状相同")
            out = Tensor(self.data * other.data, (self, other), _op='*')
        elif isinstance(other, (int, float, np.ndarray)):
            out = Tensor(self.data * other, (self,), _op='*')
        else:
            raise TypeError("乘法运算只支持 Tensor、int、float 或 np.ndarray 类型")
        
        def _backward():
            if isinstance(other, Tensor):
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            else:  # other is either a scalar or np.ndarray
                self.grad += other * out.grad
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __matmul__(self, other):
        return self.matmul(other)
    
    def matmul(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("matmul 方法的参数必须是 Tensor 类型")

        # 验证是否为至少一维张量，并且除了最后两个维度，其他维度匹配或者可广播
        if self.data.ndim < 1 or other.data.ndim < 1:
            raise ValueError("矩阵乘法要求张量至少是一维的")
        if self.data.shape[:-2] != other.data.shape[:-2] and self.data.shape[:-2] != (1,) and other.data.shape[:-2] != (1,):
            raise ValueError("除了最后两个维度外，其他所有维度的大小必须相同或者为1以进行广播")

        out = Tensor(np.matmul(self.data, other.data), (self, other), _op='@')

        def _backward():
            # 反向传播需要根据输入的维度进行调整
            if self.data.ndim > 1 and other.data.ndim > 1:
                grad_A = np.matmul(out.grad, other.data.swapaxes(-1, -2))
                grad_B = np.matmul(self.data.swapaxes(-1, -2), out.grad)
            elif self.data.ndim == 1 and other.data.ndim > 1:  # self是向量，other是矩阵
                grad_A = np.dot(out.grad, other.data.T)
                grad_B = np.dot(self.data.reshape(-1, 1), out.grad.reshape(1, -1))
            elif self.data.ndim > 1 and other.data.ndim == 1:  # self是矩阵，other是向量
                grad_A = np.dot(out.grad.reshape(-1, 1), other.data.reshape(1, -1))
                grad_B = np.dot(self.data.T, out.grad)
            else:  # 处理其他情况，例如两者都是向量
                raise NotImplementedError("目前不支持这种维度的反向传播")

            # 更新梯度，考虑到可能的广播
            self.grad += grad_A.reshape(self.data.shape)
            other.grad += grad_B.reshape(other.data.shape)


        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self.div(other)

    def div(self, other):
        if not isinstance(other, (Tensor, int, float)):
            raise TypeError("除法运算只支持Tensor、int、float类型")

        if isinstance(other, (int, float)):
            other = Tensor(other * np.ones_like(self.data, dtype=np.float32), trainable=False)
        elif isinstance(other, Tensor) and self.data.shape != other.data.shape:
            raise ValueError("形状不匹配：{} 和 {}".format(self.data.shape, other.data.shape))

        out = Tensor(self.data / other.data, (self, other), _op='/')

        def _backward():
            self.grad += np.divide(1, other.data) * out.grad
            if other.trainable:
                other.grad -= np.divide(self.data, np.square(other.data)) * out.grad

        out._backward = _backward
        return out
    
    def __pow__(self, power):
        return self.pow(power)

    # 必须重写平方项，不能用乘以自身表示平方，否则求导会错
    def pow(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError("幂指数只支持int或float类型")
        out = Tensor(self.data ** power, (self,), _op='**')
        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    
    def dropout(self, dropout_probability=0.5, is_training=True):
        if not is_training or dropout_probability == 0:
            # 如果不是训练模式或dropout_probability为0，即不丢弃任何元素，直接返回原Tensor
            return self
        else:
            # 在训练模式下，根据1-dropout_probability随机生成dropout掩码
            keep_prob = 1 - dropout_probability
            mask = np.random.binomial(1, keep_prob, size=self.data.shape) / keep_prob
            # 应用dropout掩码
            dropped_out_data = self.data * mask
            
            out = Tensor(dropped_out_data, _prev=(self,), trainable=False, _op='dropout')
            
            def _backward():
                # 反向传播时只对保留下来的元素求导
                self.grad += out.grad * mask
            
            out._backward = _backward
            return out
    
    def tanh(self):
        x = self.data
        t = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
        out = Tensor(t, (self, ), _op='tanh')

        def _backward():
            # 注意这里使用了 out.data 来计算梯度
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward

        return out
    
    def sigmoid(self):
        x = self.data
        s = 1 / (1 + np.exp(-x))
        out = Tensor(s, (self,), _op='sigmoid')

        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        x = self.data
        r = np.maximum(0, x)  # 使用 numpy 的 maximum 函数
        out = Tensor(r, (self,), _op='relu')
        def _backward():
            self.grad += (x > 0) * out.grad  # Gradient is 1 for x > 0, otherwise 0
        out._backward = _backward
        return out
    
    def mean(self, dim=None, keepdims=False):
        if dim is None:
            # 如果没有指定维度，就计算所有元素的平均值
            forward_value = np.mean(self.data)
            n = self.data.size
        else:
            # 计算指定维度的平均值
            forward_value = np.mean(self.data, axis=dim, keepdims=keepdims)
            n = self.data.shape[dim]

        out = Tensor(forward_value, (self,), _op='mean')

        def _backward():
            if dim is None or keepdims:
                grad = np.ones_like(self.data, dtype=np.float32) * (out.grad / n)
            else:
                # 当不保持维度且指定了维度时，需要扩展梯度以匹配原始数据形状
                expand_shape = list(self.data.shape)
                expand_shape[dim] = 1
                grad = np.ones(expand_shape, dtype=np.float32) * (out.grad / n)
                grad = np.broadcast_to(grad, self.data.shape)
            self.grad += grad

        out._backward = _backward
        return out

    def backward(self, grad=1.):
        if not isinstance(grad, (int, float)):
            raise ValueError("初始梯度必须为整型或字符型, 建议设为1。")
        topo = []
        # 记录下已访问过的Tensor集合
        self.visited = set()
        def build_topo(v):
            if v not in self.visited:
                self.visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # 因为上面得到的topo序列顺序是反的，所以我们在逐步计算偏导数时需要先进行逆序操作
        self.grad = grad
        for node in reversed(topo):
            node._backward()

    def gradient_descent_opt(self, learning_rate=0.001, grad_zero=True):
        if hasattr(self, 'visited'):
            for v in self.visited:
                if v.trainable:
                    v.data -= learning_rate * v.grad
                if grad_zero:
                    v.grad = 0
        else:
            raise AttributeError("没有'visited'这个属性. 请在运行optimization前先运行backward()")

    def adam_opt(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, grad_zero=True):
        self.optim_step += 1  # 递增整个优化过程的时间步
        for tensor in self.visited:
            if tensor.trainable:
                # 初始化动量和速度
                if not hasattr(tensor, 'momentum'):
                    tensor.momentum = np.zeros_like(tensor.data, dtype=np.float32)
                    tensor.velocity = np.zeros_like(tensor.data, dtype=np.float32)

                # 更新动量和速度
                tensor.momentum = beta1 * tensor.momentum + (1 - beta1) * tensor.grad
                tensor.velocity = beta2 * tensor.velocity + (1 - beta2) * np.square(tensor.grad)

                # 计算偏差校正后的估计
                m_hat = tensor.momentum / (1 - beta1 ** self.optim_step)
                v_hat = tensor.velocity / (1 - beta2 ** self.optim_step)

                # 更新参数
                tensor.data -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            if grad_zero:
                tensor.grad = 0
    
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, trainable={self.trainable})"