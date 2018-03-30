import numpy as np
import torch


class Transition:
    def __init__(self, prev_stat, prd_act, next_stat, real_val) -> None:
        super().__init__()
        self.prev_stat = torch.FloatTensor(prev_stat).view(1, 1, -1)
        self.prd_act = torch.IntTensor(prd_act).view(1, 1, -1)
        self.next_stat = torch.FloatTensor(next_stat).view(1, 1, -1)
        self.real_val = torch.FloatTensor(real_val)


class TransitionContainer:
    def __init__(self, capacity) -> None:
        super().__init__()
        self.capacity = capacity
        self.index = 0
        self.content = []

    def put(self, transition):
        if len(self.content) < self.capacity:
            self.content.append(transition)
        else:
            self.index = (self.index + 1) % self.capacity
            self.content[self.index] = transition

    def get(self, index):
        if len(self.content) <= index:
            return None
        else:
            return self.content[index]

    def clear(self):
        self.content.clear()
        self.index = 0

    def size(self):
        return len(self.content)

    def sample(self, sample_size):
        sample_index = np.random.choice(len(self.content), sample_size)
        ret_list = []
        for i in sample_index:
            ret_list.append(self.content[i])
        return ret_list

