import numpy as np
class BatchSampler:
    def __init__(self, train_x, train_y, batch_size):
        """
        初始化BatchSampler对象。

        参数:
        train_x (array-like): 训练数据的特征。
        train_y (array-like): 训练数据的标签。
        batch_size (int): 每个批次的大小。

        描述:
        此构造函数初始化训练数据、标签和批次大小，并随机打乱索引以进行批处理。
        """
        self.train_x = train_x
        self.train_y = train_y
        self.batch_size = batch_size
        self.indices = np.arange(len(self.train_x))
        np.random.shuffle(self.indices)  # Shuffle at the start
        self.current_index = 0

    def next_batch(self):
        """
        获取下一个数据批次。

        描述:
        此方法返回下一个批次的数据和标签。当所有数据已被遍历时，
        它会重新打乱索引并从头开始。

        返回:
        tuple: 包含特征和标签的两个数组，分别对应当前批次的数据。
        """
        if self.current_index + self.batch_size > len(self.train_x):
            # Reshuffle the indices and reset the current index
            np.random.shuffle(self.indices)
            self.current_index = 0

        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        return self.train_x[batch_indices], self.train_y[batch_indices]
