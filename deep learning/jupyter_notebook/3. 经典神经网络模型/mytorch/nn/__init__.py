from .modules import Module, Linear, MSELoss
from .functional import functional
from ..tensor import Tensor

def ReLU(tensor: Tensor) -> Tensor:
    return tensor.relu()

class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.data if isinstance(pred, Tensor) else pred
        target = target.data if isinstance(target, Tensor) else target
        def mse_recursive(p, t):
            if type(p) not in [list, tuple] or type(t) not in [list, tuple]:
                return (p - t) ** 2
            else:
                return sum([mse_recursive(pi, ti) for pi, ti in zip(p, t)]) / len(p)

        total_loss = mse_recursive(pred, target)
        return Tensor(total_loss)