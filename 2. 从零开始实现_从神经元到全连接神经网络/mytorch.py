import math
import numpy as np
class Tensor:
    """
    Tensor类代表一个多维数组，用于神经网络中的张量运算。
    它支持基本的算术运算，如加减乘除和幂运算，并支持一些激活函数。
    这个类还包含反向传播算法，用于计算和更新梯度。
    """
    def __init__(self, data, _prev=(), trainable=True, _op='', label=''):
        """
        初始化一个Tensor对象。
        
        参数:
        data (array_like): 包含Tensor数据的数组。
        _prev (tuple, 可选): 与当前Tensor相关的前置Tensor对象集合。
        trainable (bool, 可选): 指示Tensor是否应该在训练过程中更新。
        _op (str, 可选): 与Tensor关联的操作符。
        label (str, 可选): Tensor的标签，用于调试和可视化。
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.grad = np.zeros_like(data, dtype=np.float32)
        self._backward = lambda: None
        self._prev = set(_prev)
        self.trainable = trainable
        self._op = _op
        self.label = label

    @staticmethod
    def from_numpy(ndarray):
        """
        从NumPy数组创建Tensor对象。
        
        参数:
        ndarray (np.ndarray): 用于创建Tensor的NumPy数组。
        
        返回:
        Tensor: 由NumPy数组创建的Tensor对象。
        """
        if not isinstance(ndarray, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        return Tensor(ndarray)

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
        elif self.data.shape != other.data.shape:
            raise ValueError("形状不匹配：{} 和 {}".format(self.data.shape, other.data.shape))

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
        if isinstance(other, (int, float)):
            other = Tensor(other * np.ones_like(self.data).astype(np.float32), trainable=False)
        elif self.data.shape != other.data.shape:
            raise ValueError("形状不匹配：{} 和 {}".format(self.data.shape, other.data.shape))
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
        other (Tensor, int, float): 与当前Tensor相乘的另一个Tensor或标量。

        返回:
        Tensor: 乘法运算的结果。
        """
        if isinstance(other, Tensor):
            # 对于两个 Tensor 对象的矩阵乘法
            if self.data.ndim != 2 or other.data.ndim != 2:
                raise ValueError("矩阵乘法要求两个张量都是二维的")

            out = Tensor(np.dot(self.data, other.data), (self, other), _op='·')

            def _backward():
                self.grad += np.dot(out.grad, other.data.T)
                other.grad += np.dot(self.data.T, out.grad)
        else:
            # 对于标量乘法
            out = Tensor(self.data * other, (self,), _op='·')
            def _backward():
                self.grad += other * out.grad

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