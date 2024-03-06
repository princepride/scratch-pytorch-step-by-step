import pytest
from mytorch import Tensor
from mytorch.nn.modules import Module, Linear, MSELoss
import numpy as np
import torch

class SimpleModule(Module):
    def __init__(self):
        super().__init__()
        self.param = Tensor(np.random.rand(10, 10))  # 假设Tensor类接受numpy数组

    def forward(self, x):
        # 假设Tensor支持'*'操作
        return self.param * x

# 测试Module能否正确注册参数
def test_parameter_registration():
    module = SimpleModule()
    assert "param" in module._parameters

# 测试添加子模块的功能
def test_add_module():
    parent_module = Module()
    child_module = SimpleModule()
    parent_module.add_module('child', child_module)
    assert 'child' in parent_module._modules

# 测试前向传播是否抛出NotImplementedError
def test_forward_not_implemented():
    module = Module()
    with pytest.raises(NotImplementedError):
        module.forward()

# 测试是否能递归获取所有参数的名称和值
def test_named_parameters():
    parent_module = Module()
    child_module = SimpleModule()
    parent_module.add_module('child', child_module)
    names_and_params = list(parent_module.named_parameters())
    assert len(names_and_params) == 1 and names_and_params[0][0] == 'child.param'

# 测试__setattr__方法的自定义行为
def test_attribute_setting():
    module = Module()
    tensor = Tensor(np.random.rand(5, 5))
    module.some_tensor = tensor
    assert module.some_tensor is tensor and "some_tensor" in module._parameters
    child_module = SimpleModule()
    module.some_module = child_module
    assert module.some_module is child_module and "some_module" in module._modules
    module.some_value = 123
    assert module.some_value == 123

def test_linear_initialization():
    in_features = 5
    out_features = 3
    linear = Linear(in_features, out_features)
    assert linear.w.data.shape == (in_features, out_features)
    if linear.bias:
        assert linear.b.data.shape == (1, out_features)

# 前向传播测试
def test_linear_forward():
    in_features = 2
    out_features = 3
    linear = Linear(in_features, out_features)
    x = Tensor.from_numpy(np.array([[1, 2]]))  # 假设2个输入特征
    output = linear.forward(x)
    assert output.data.shape == (1, out_features)

# 偏置项测试
def test_linear_no_bias():
    in_features = 5
    out_features = 3
    linear = Linear(in_features, out_features, bias=False)
    assert not hasattr(linear, 'b')

# 字符串表示测试
def test_linear_repr():
    in_features = 5
    out_features = 3
    linear = Linear(in_features, out_features)
    expected_repr = f"Linear(in_features={in_features}, out_features={out_features}, bias=True)"
    assert repr(linear) == expected_repr

def test_MSELoss():
    # 生成随机数据
    pred_np = np.random.rand(10, 5)
    target_np = np.random.rand(10, 5)

    # 使用 PyTorch 计算 MSE Loss
    pred_torch = torch.tensor(pred_np, dtype=torch.float32, requires_grad=True)
    target_torch = torch.tensor(target_np, dtype=torch.float32)
    criterion_torch = torch.nn.MSELoss()
    loss_torch = criterion_torch(pred_torch, target_torch)
    
    # 使用自定义 Tensor 类和 MSELoss 类计算 MSE Loss
    pred_custom = Tensor(pred_np)
    target_custom = Tensor(target_np)
    criterion_custom = MSELoss()
    loss_custom = criterion_custom(pred_custom, target_custom)
    
    # 将自定义方法计算的结果转换为 PyTorch Tensor，以便比较
    loss_custom_torch = torch.tensor(loss_custom.data, dtype=torch.float32)
    
    # 比较结果
    assert torch.isclose(loss_torch, loss_custom_torch, atol=1e-6)