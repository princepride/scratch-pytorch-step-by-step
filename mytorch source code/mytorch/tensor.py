import math
import numpy as np
from typing import List
class Tensor:

    # 给予Tensor类更高的优先级
    __array_priority__ = 1.0

    """
    Tensor类代表一个多维数组，用于神经网络中的张量运算。
    它支持基本的算术运算，如加减乘除和幂运算，并支持一些激活函数。
    这个类还包含反向传播算法，用于计算和更新梯度。
    """
    def __init__(self, data, _prev=(), trainable=True, _op='', label=''):
        """
        初始化一个Tensor对象。
        
        参数:
        data (array_like): 包含Tensor数据的数组, 支持(int, float, List, np.array)数据类型。
        _prev (tuple, 可选): 与当前Tensor相关的前置Tensor对象集合。
        trainable (bool, 可选): 指示Tensor是否应该在训练过程中更新。
        _op (str, 可选): 与Tensor关联的操作符。
        label (str, 可选): Tensor的标签, 用于调试和可视化。
        """
        if isinstance(data, Tensor):
            raise TypeError("Tensor被用于初始化的数据类型不能是Tensor类型")
        if isinstance(data, int):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, float):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, List):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            pass
        else:
            raise TypeError("未知的初始化数据类型, Tensor类只可用int, float, List以及np.ndarray进行初始化")
        self.data = data
        self.grad = np.zeros_like(data, dtype=np.float32)
        self._backward = lambda: None
        self._prev = set(_prev)
        self.trainable = trainable
        self._op = _op
        self.label = label

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
        """
        比较当前Tensor和另一个Tensor是否不相等。
        
        参数:
        other (Tensor): 要比较的另一个Tensor对象。
        
        返回:
        bool: 如果数据或训练标志不相同，则为True；否则为False。
        """
        return not self.__eq__(other)
    
    # 当我们重写__eq__方法而没有同时重写__hash__方法时，Python会默认将我们的对象视为不可哈希的。
    # 这是因为__eq__方法定义了对象相等性的行为，而哈希值(hash value)是用来在哈希表中快速比较键的一种机制。
    # 如果两个对象被视为相等（即它们的__eq__方法返回True），那么它们的哈希值也应该相同。
    # 为了保持一致性，当我们定义__eq__方法时，如果不定义__hash__方法，Python会将类的实例变为不可哈希的，
    # 从而防止了使用不一致的哈希值导致的潜在问题。
    def __hash__(self):
        """
        生成当前Tensor对象的哈希值。

        哈希值是基于对象的唯一标识符（ID）、是否可训练的标志（trainable）、操作符（_op）以及标签（label）来生成的。
        这确保了即使两个Tensor对象的内容相同，它们的哈希值也将基于它们的唯一性（内存地址）和其他属性而不同，
        除非它们是同一个对象的引用。

        返回:
        int: 代表当前Tensor对象哈希值的整数。
        """
        return hash((id(self), self.trainable, self._op, self.label))
    
    @staticmethod
    def from_numpy(ndarray, trainable=True):
        """
        从NumPy数组创建Tensor对象。
        
        参数:
        ndarray (np.ndarray): 用于创建Tensor的NumPy数组。
        
        返回:
        Tensor: 由NumPy数组创建的Tensor对象。
        """
        if isinstance(ndarray, Tensor):
            ndarray.trainable = trainable
            return ndarray
        if isinstance(ndarray, np.ndarray):
            return Tensor(ndarray, trainable = trainable)
        if isinstance(ndarray, List):
            return Tensor(ndarray, trainable= trainable)
        raise TypeError("Input must be a NumPy array")
    
    def size(self, dim=None):
        """
        参数:
        - dim: 可选，整数，指定想要获取大小的维度。
        
        返回:
        - 如果指定了dim，则为整数；如果没有指定，则为表示张量形状的元组。
        """
        if dim is None:
            return self.data.shape
        else:
            if not isinstance(dim, int):
                raise TypeError("dim 必须是整型（int）")
            if dim < -len(self.data.shape) or dim >= len(self.data.shape):
                raise IndexError("维度超出范围")
            return self.data.shape[dim]
        
    def unsqueeze(self, axis):
        """
        在指定轴上增加一个维度。

        参数:
        axis (int): 要增加新维度的轴。例如, axis=0将在最外层添加一个新轴。

        返回:
        Tensor: 经过增加维度后的新Tensor对象。
        """
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
        """
        将Tensor的形状改变为新的形状。

        参数:
        new_shape (tuple of ints): 目标形状。元素的总数应与原始形状相同。

        返回:
        Tensor: 重塑后的新Tensor对象。
        """
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
        """
        移除张量中的单维度。如果指定了轴（axis），则只移除该轴的单维度。
        
        参数:
        axis (int, 可选): 指定要移除的维度。如果未指定，则移除所有长度为1的维度。

        返回:
        Tensor: 经过压缩维度后的新Tensor对象。
        """
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
        """
        重新排列张量的维度。

        参数:
        *dims: 一个维度的序列，表示要排列成的新顺序。

        返回:
        Tensor: 经过维度重新排列后的新Tensor对象。
        """
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
        """
        交换张量中的两个维度。

        参数:
        dim0 (int): 要交换的第一个维度。
        dim1 (int): 要交换的第二个维度。

        返回:
        Tensor: 经过维度交换后的新Tensor对象。
        """
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
        """
        实现沿指定维度将一系列Tensor对象连接起来的功能。

        参数:
        tensors (list of Tensor): 要连接的Tensor对象列表。
        dim (int, optional): 要沿其连接的维度，默认为0。

        返回:
        Tensor: 连接后的新Tensor对象。

        注意:
        1. 所有输入Tensor的除了连接维度以外的其他维度大小必须相同。
        2. 连接操作不会改变输入Tensor的原始数据，而是创建一个包含所有数据的新Tensor。
        3. 反向传播时，梯度将被正确分配回各个原始Tensor。
        """
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
        """
        实现Tensor对象的加法运算。

        参数:
        other (Tensor, int, float): 与当前Tensor相加的另一个Tensor或标量。

        返回:
        Tensor: 加法运算的结果。
        """
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
        """
        实现右加法。这使得 'other + self' 在 'other' 不是Tensor时也有效。

        参数:
        other (int, float): 与当前Tensor相加的标量。

        返回:
        Tensor: 加法运算的结果。
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        实现Tensor对象的减法运算。

        参数:
        other (Tensor, int, float): 从当前Tensor中减去的另一个Tensor或标量。

        返回:
        Tensor: 减法运算的结果。
        """
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
        """
        实现右减法。这使得 'other - self' 在 'other' 不是Tensor时也有效。

        参数:
        other (int, float): 从标量中减去当前Tensor。

        返回:
        Tensor: 减法运算的结果。
        """
        return (-self).__add__(other)
    
    # 必须重写负号，否则(-self)会报错
    def __neg__(self):
        """
        实现Tensor的负数运算。

        返回:
        Tensor: 当前Tensor的负值。
        """
        neg_tensor = Tensor(-self.data, (self,))
        def _backward():
            self.grad -= 1.0 * neg_tensor.grad  
        neg_tensor._backward = _backward
        return neg_tensor

    def __mul__(self, other):
        """
        实现Tensor对象的乘法运算。

        参数:
        other (Tensor, int, float, np.ndarray): 与当前Tensor相乘的另一个Tensor或标量。

        返回:
        Tensor: 乘法运算的结果。
        """
        if isinstance(other, Tensor):
            # 确保张量与张量的乘法
            if self.data.ndim == 2 and other.data.ndim == 2:
                # 矩阵乘法
                out_data = np.dot(self.data, other.data)
            else:
                # 逐元素乘法或者广播乘法
                out_data = self.data * other.data
            out = Tensor(out_data, (self, other), _op='*')
        elif isinstance(other, (int, float, np.ndarray)):
            # 标量乘法或者与ndarray的乘法
            out = Tensor(self.data * other, (self,), _op='*')
        else:
            raise TypeError("乘法运算只支持 Tensor、int、float 或 np.ndarray 类型")

        def _backward():
            if isinstance(other, Tensor):
                self.grad += out.grad * (other.data if other.data.ndim == self.data.ndim else other.data.T)
                other.grad += out.grad * (self.data if other.data.ndim == self.data.ndim else self.data.T)
            else:  # other is either a scalar or np.ndarray
                self.grad += out.grad * other

        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        """
        实现右乘法。这使得 'other * self' 在 'other' 不是Tensor时也有效。

        参数:
        other (int, float): 与当前Tensor相乘的标量。

        返回:
        Tensor: 乘法运算的结果。
        """
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """
        实现Tensor对象的除法运算。

        参数:
        other (Tensor, int, float): 当前Tensor被除以的另一个Tensor或标量。

        返回:
        Tensor: 除法运算的结果。
        """
        if isinstance(other, (int, float)):
            other = Tensor(other * np.ones_like(self.data).astype(np.float32), trainable=False)
        elif self.data.shape != other.data.shape:
            raise ValueError("形状不匹配：{} 和 {}".format(self.data.shape, other.data.shape))

        out = Tensor(self.data / other.data, (self, other), _op='/')

        def _backward():
            self.grad += np.divide(1, other.data) * out.grad
            other.grad -= np.divide(self.data, np.square(other.data)) * out.grad
        out._backward = _backward
        return out
    
    # 必须重写平方项，不能用乘以自身表示平方，否则求导会错
    def __pow__(self, power):
        """
        实现Tensor对象的幂运算。

        参数:
        power (int, float): 当前Tensor的幂指数。

        返回:
        Tensor: 幂运算的结果。
        """
        out = Tensor(self.data ** power, (self,), _op='**')
        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out
    
    def dropout(self, dropout_probability=0.5, is_training=True):
        """
        实现Tensor对象的dropout操作。在训练过程中随机丢弃一部分神经元（将它们的激活值设为0），
        以避免过拟合。使用Inverted Dropout方法，保证激活值的期望不变。

        参数:
        dropout_probability (float): 丢弃神经元的概率，默认为0.5。
        is_training (bool): 指示当前是否处于训练模式。只有在训练模式下才进行dropout操作。

        返回:
        Tensor: 经过dropout操作的Tensor对象。如果不是训练模式，则返回未修改的Tensor对象。
        """
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
        """
        应用双曲正切激活函数。

        返回:
        Tensor: 应用双曲正切后的Tensor。
        """
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Tensor(t, (self, ), _op='tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def sigmoid(self):
        """
        应用Sigmoid激活函数。

        返回:
        Tensor: 应用Sigmoid后的Tensor。
        """
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Tensor(s, (self,), _op='sigmoid')

        def _backward():
            self.grad += (s * (1 - s)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        """
        应用ReLU激活函数。

        返回:
        Tensor: 应用ReLU后的Tensor。
        """
        x = self.data
        r = np.maximum(0, x)  # 使用 numpy 的 maximum 函数
        out = Tensor(r, (self,), _op='relu')

        def _backward():
            self.grad += (x > 0) * out.grad  # Gradient is 1 for x > 0, otherwise 0
        out._backward = _backward

        return out
    
    def gradient_descent_opt(self, learning_rate=0.001, grad_zero=True):
        """
        使用梯度下降算法优化Tensor中的参数。

        参数:
        learning_rate (float, 可选): 学习率。
        grad_zero (bool, 可选): 是否在优化后将梯度重置为零。
        """
        for v in self.visited:
            if v.trainable:
                v.data -= learning_rate * v.grad
            if grad_zero:
                v.grad = 0

    def adam_opt(self, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, grad_zero=True):
        """
        使用Adam优化算法优化Tensor中的参数。

        参数:
        t (int): 当前优化的时间步。
        learning_rate (float, 可选): 学习率。
        beta1 (float, 可选): 一阶矩估计的指数衰减率。
        beta2 (float, 可选): 二阶矩估计的指数衰减率。
        epsilon (float, 可选): 用于防止除以零的小数。
        grad_zero (bool, 可选): 是否在优化后将梯度重置为零。

        返回:
        int: 更新后的时间步。
        """
        t += 1  # 递增整个优化过程的时间步
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
                m_hat = tensor.momentum / (1 - beta1 ** t)
                v_hat = tensor.velocity / (1 - beta2 ** t)

                # 更新参数
                tensor.data -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            if grad_zero:
                tensor.grad = 0

        return t  # 返回更新后的时间步



    def backward(self, ):
        """
        执行反向传播算法，计算梯度。
        """
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
        self.grad = 1.
        for node in reversed(topo):
            node._backward()
    
    def __repr__(self):
        """
        返回Tensor对象的字符串表示。

        返回:
        str: Tensor对象的字符串表示。
        """
        return f"Tensor(data={self.data}, grad={self.grad}, trainable={self.trainable})"