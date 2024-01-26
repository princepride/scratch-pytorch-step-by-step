import numpy as np
class BatchSampler:
    def __init__(self, train_x, train_y, batch_size):
        self.train_x = train_x
        self.train_y = train_y
        self.batch_size = batch_size
        self.indices = np.arange(len(self.train_x))
        np.random.shuffle(self.indices)  # Shuffle at the start
        self.current_index = 0

    def next_batch(self):
        if self.current_index + self.batch_size > len(self.train_x):
            # Reshuffle the indices and reset the current index
            np.random.shuffle(self.indices)
            self.current_index = 0

        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        return self.train_x[batch_indices], self.train_y[batch_indices]