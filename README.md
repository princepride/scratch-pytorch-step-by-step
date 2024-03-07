# scratch-pytorch-step-by-step

这个项目旨在一步步教你实现一个语法风格类似于PyTorch的深度学习框架。虽然PyTorch的底层是用C语言实现的，但考虑到完全还原可能会增加学习门槛，本教程将使用Python的基础语法和NumPy来实现深度学习领域的一些基础算法，包括但不限于反向传播、随机梯度下降、Adam优化器、Dropout层等。本项目还会介绍如何实现包括CNN、RNN、LSTM、ResNet、Transformer等在内的模型。

## 目录

| 章节 | 内容                                                                        | 完成度 |
| ---- | --------------------------------------------------------------------------- | ------ |
| 1.1  | Python的基本数据类型                                                        | ✅     |
| 1.2  | Python中的条件控制语句                                                      | ✅     |
| 1.3  | Python中的函数和类的声明和使用                                              | ✅     |
| 1.4  | Numpy这个Python包的用法                                                     | ✅     |
| 2.1  | 用线性回归的方式拟合一个曲线，均方差损失函数                                | ✅     |
| 2.2  | 实现张量的基本计算，sigmoid，tanh，relu激活函数，反向传播算法，随机梯度下降 | ✅     |
| 2.3  | 将张量的支持范围扩充到矩阵，Adam梯度下降，网络模型的可视化                  | ✅     |
| 2.4  | 全连接网络模型，模型信息打印                                                | ✅     |
| 3.1  | 卷积神经网络，卷积层，池化层，Dropout层                                     | ⬜     |
| 3.2  | AlexNet网络的实现                                                           | ⬜     |
| 3.3  | 矩阵拼接，循环神经网络的实现                                                | ✅     |
| 3.4  | 长短期记忆网络的实现                                                        | ✅     |
| 3.5  | 残差网络的实现                                                              | ✅     |
| 3.6  | 用pytorch从头到尾实现transformer模型                                        | ✅     |
| 3.7  | 用mytorch从头到位实现transformer模型                                        | ⬜     |
| 3.8  | 实现mamba模型                                                               | ⬜     |

## 主程序

在 `mytorch source code`文件目录下

## 待办事项

| To do list                | 完成度 |
| ------------------------- | ------ |
| 实现cuda版的矩阵运算      | ⬜     |
| 实现cuda版的反向传播      | ⬜     |
| 实现cuda版Flash Attention | ⬜     |
| 实现cuda版4bit，1bit量化  | ⬜     |

## 如何贡献
欢迎所有对深度学习感兴趣的人士贡献力量！无论是修复bug、添加新特性还是改进文档，您的帮助都是我前进的动力。

1. Fork本仓库。
2. 创建您的功能分支 (git checkout -b feature/AmazingFeature)。
3. 提交您的更改 (git commit -m 'Add some AmazingFeature')。
4. 将您的更改推送到GitHub (git push origin feature/AmazingFeature)。
5. 创建一个新的Pull Request。

## 创建问题
遇到问题或有好的建议？请不犹豫，通过创建一个issue来让我知道。我会尽力解决问题，欢迎任何形式的反馈。

## 社区参与
- 如果你觉得这个项目对你有帮助，不妨给它一个Star✨！
- 分享给更多的朋友，让更多的人了解和参与进来。

## 许可证
本项目根据Apache License 2.0许可证授权。

## 致谢
本项目的灵感来源于Andrej Karpathy的micrograd项目。我们对他在深度学习教育领域所做的贡献表示深深的感谢。
