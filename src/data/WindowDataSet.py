import torch
from torch.utils.data import Dataset


class WindowDataSet(Dataset):

    def __init__(self, window_list) -> None:
        super().__init__()
        self.window_list = window_list

    def __getitem__(self, index):
        data = self.window_list[index].originD
        label = self.window_list[index].nextD
        return {'data': torch.FloatTensor(data).view(1, 1, -1), 'label': torch.FloatTensor(label).view(1, 1, -1)}

    def __len__(self):
        return len(self.window_list)
