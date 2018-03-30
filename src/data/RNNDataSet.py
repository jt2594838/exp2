from torch.utils.data import Dataset
import torch

from data.DataWindow import DataWindow


class RNNDataSet(Dataset):

    def __init__(self, window_list, time_len) -> None:
        super().__init__()
        self.input_size = window_list[0].originD.size(1)
        self.process_win_list(window_list, time_len)

    def __getitem__(self, index):
        data = self.content[index].originD
        label = self.content[index].nextD
        return {'data': data, 'label': label}

    def __len__(self):
        return len(self.content)

    def process_win_list(self, window_list, time_len):
        self.content = []
        i = 0
        while i < len(window_list):
            new_data = None
            new_label = None
            for j in range(time_len):
                if i + j >= len(window_list):
                    return
                window = window_list[i + j]
                win_data = window.originD
                if new_data is None:
                    new_data = win_data
                else:
                    new_data = torch.cat((new_data, win_data), 0)
                new_label = window.nextD
            i += time_len
            self.content.append(DataWindow(new_data, new_label))

