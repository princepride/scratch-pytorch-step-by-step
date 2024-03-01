import numpy as np
from ..tensor import Tensor
class Module:
    def __init__(self):
        """
        初始化Module类的实例。

        描述:
        此类作为神经网络模块的基类，用于创建新的模块。它包含参数和子模块的字典。
        """
        self._parameters = {}
        self._modules = {}

    def forward(self, *input):
        """
        前向传播的框架方法。

        描述:
        所有子类都应重写此方法以实现其特定的前向传播功能。

        抛出:
        NotImplementedError: 如果子类没有重写此方法。
        """
        raise NotImplementedError

    def __call__(self, *input):
        """
        使得实例可以像函数那样被调用。

        返回:
        调用forward方法的结果。
        """
        return self.forward(*input)

    def named_parameters(self, memo=None, prefix=''):
        """
        递归获取所有参数的名称和值。

        参数:
        memo (set): 防止重复的集合。
        prefix (str): 参数名称的前缀。

        返回:
        生成器: 生成参数的名称和值。
        """
        if memo is None:
            memo = set()

        for name, param in self._parameters.items():
            if param not in memo:
                memo.add(param)
                yield prefix + name, param

        for name, mod in self._modules.items():
            submodule_prefix = prefix + name + '.'
            for name, param in mod.named_parameters(memo, submodule_prefix):
                yield name, param

    def add_module(self, name, module):
        """
        向模块中添加子模块。

        参数:
        name (str): 子模块的名称。
        module (Module): 要添加的子模块。

        抛出:
        TypeError: 如果module不是Module子类。
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(type(module)))
        self._modules[name] = module

    def __setattr__(self, name, value):
        """
        定制属性设置行为。

        描述:
        当设置的属性是Tensor或Module时，分别更新参数字典或添加子模块。
        """
        if isinstance(value, Tensor):
            object.__setattr__(self, name, value)  # 先设置属性
            self._parameters[name] = value          # 然后添加到参数字典中
        elif isinstance(value, Module):
            object.__setattr__(self, name, value)
            self.add_module(name, value)
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        """
        返回模块的字符串表示。

        返回:
        str: 模块的字符串表示。
        """
        lines = [self.__class__.__name__ + '(']
        for name, module in self._modules.items():
            mod_str = repr(module).replace('\n', '\n  ')
            lines.append(f"  ({name}): {mod_str}")
        lines.append(')')
        return '\n'.join(lines)
    
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        """
        初始化Linear层。

        参数:
        in_features (int): 输入特征的数量。
        out_features (int): 输出特征的数量。
        bias (bool): 是否添加偏置项。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.w = Tensor.from_numpy(np.random.rand(in_features, out_features))
        if bias:
            self.b = Tensor.from_numpy(np.random.rand(1, out_features))
    
    def forward(self, x):
        """
        前向传播方法。

        参数:
        x (Tensor): 输入张量。

        返回:
        Tensor: 经过线性变换的输出。
        """
        return x@self.w+self.b if self.bias else x@self.w
    
    def __repr__(self):
        """
        返回Linear层的字符串表示。

        返回:
        str: Linear层的字符串表示。
        """
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})"
    
class MSELoss(Module):
    def __init__(self):
        """
        初始化均方误差（MSE）损失模块。
        """
        super().__init__()

    def forward(self, pred, target):
        """
        计算预测和目标之间的均方误差。

        参数:
        pred (Tensor): 预测值。
        target (Tensor): 真实目标值。

        返回:
        Tensor: 均方误差值。
        """
        loss = 0
        for i in range(len(target)):
            loss += (pred[i] - target[i]) ** 2
        return loss/len(target)

