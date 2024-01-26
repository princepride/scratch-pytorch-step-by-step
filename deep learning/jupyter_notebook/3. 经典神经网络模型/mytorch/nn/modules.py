import numpy as np
from ..tensor import Tensor
class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input):
        return self.forward(*input)

    def named_parameters(self, memo=None, prefix=''):
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
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(type(module)))
        self._modules[name] = module

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            object.__setattr__(self, name, value)  # 先设置属性
            self._parameters[name] = value          # 然后添加到参数字典中
        elif isinstance(value, Module):
            object.__setattr__(self, name, value)
            self.add_module(name, value)
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        lines = [self.__class__.__name__ + '(']
        for name, module in self._modules.items():
            mod_str = repr(module).replace('\n', '\n  ')
            lines.append(f"  ({name}): {mod_str}")
        lines.append(')')
        return '\n'.join(lines)
    
class Linear(Module):
    def __init__(self, in_features, out_features, bias = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.w = Tensor.from_numpy(np.random.rand(in_features, out_features))
        if bias:
            self.b = Tensor.from_numpy(np.random.rand(1, out_features))
    
    def forward(self, x):
        temp = x*self.w
        return x*self.w+self.b if self.bias else x*self.w
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})"
    
class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        loss = 0
        for i in range(len(target)):
            loss += (pred[i] - target[i]) ** 2
        return loss/len(target)