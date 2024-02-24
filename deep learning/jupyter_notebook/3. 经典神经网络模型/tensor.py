import math
import numpy as np
class Tensor:
    def __init__(self, data, _prev=(), trainable=True, _op='', label=''):
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
    def cat(tensors, dim=0):
        assert all(isinstance(t, Tensor) for t in tensors), "所有输入必须是Tensor对象"
        data = [t.data for t in tensors]
        concatenated_data = np.concatenate(data, axis=dim)
        
        # 创建一个新的Tensor对象作为cat操作的结果
        out = Tensor(concatenated_data, _prev=tuple(tensors), _op='cat')
        
        def _backward():
            # 分割out.grad并将相应的梯度分配给原始的Tensor对象
            grad_splits = np.split(out.grad, np.cumsum([t.data.shape[dim] for t in tensors[:-1]]), axis=dim)
            for t, grad in zip(tensors, grad_splits):
                t.grad += grad
        
        out._backward = _backward
        return out
    
    @staticmethod
    def from_numpy(ndarray):
        if not isinstance(ndarray, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        return Tensor(ndarray)

    def __add__(self, other):
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
        return self.__add__(other)

    def __sub__(self, other):
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
        return (-self).__add__(other)
    
    # 必须重写负号，否则(-self)会报错
    def __neg__(self):
        neg_tensor = Tensor(-self.data, (self,))
        def _backward():
            self.grad -= 1.0 * neg_tensor.grad  
        neg_tensor._backward = _backward
        return neg_tensor

    def __mul__(self, other):
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
        return self.__mul__(other)
    
    def __truediv__(self, other):
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
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Tensor(t, (self, ), _op='tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Tensor(s, (self,), _op='sigmoid')

        def _backward():
            self.grad += (s * (1 - s)) * out.grad
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
    
    def gradient_descent_opt(self, learning_rate=0.001, grad_zero=True):
        for v in self.visited:
            if v.trainable:
                v.data -= learning_rate * v.grad
            if grad_zero:
                v.grad = 0

    def adam_opt(self, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, grad_zero=True):
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
        return f"Tensor(data={self.data}, grad={self.grad}, trainable={self.trainable})"