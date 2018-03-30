
import torch.nn as nn
import numpy as np

KERNEL_SIZE = 3


class BasiCNN(nn.Module):

    def __init__(self, config):
        super(BasiCNN, self).__init__()
        input_size = int(config['input_size'])
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=KERNEL_SIZE),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            # nn.MaxPool1d(2),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=KERNEL_SIZE),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            # nn.MaxPool1d(2),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=KERNEL_SIZE),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            # nn.MaxPool1d(2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=KERNEL_SIZE),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.MaxPool1d(2),
        )
        # 16 channels
        self.pred_input_size = self.cal_size(input_size) * 4
        self.pred_output_size = int(config['output_size'])
        self.pred = nn.Sequential(
            nn.Linear(self.pred_input_size, int(self.pred_input_size / 2)),
            nn.BatchNorm1d(int(self.pred_input_size / 2)),
            nn.ReLU(),
            nn.Linear(int(self.pred_input_size / 2), int(self.pred_input_size / 4)),
            nn.BatchNorm1d(int(self.pred_input_size / 4)),
            nn.ReLU(),
            nn.Linear(int(self.pred_input_size / 4), int(self.pred_input_size / 8)),
            nn.BatchNorm1d(int(self.pred_input_size / 8)),
            nn.ReLU(),
            nn.Linear(int(self.pred_input_size / 8), self.pred_output_size),
        )
        # self.apply(weight_init)

    def forward(self, x):
        conved = self.conv(x).view(-1, self.pred_input_size)
        result = self.pred(conved)
        return result

    def cal_size(self, input_size):
        input_size = input_size - KERNEL_SIZE + 1
        input_size = input_size - KERNEL_SIZE + 1
        # input_size = int(input_size / 2)
        input_size = input_size - KERNEL_SIZE + 1
        input_size = input_size - KERNEL_SIZE + 1
        # input_size = int(input_size / 2)
        return input_size

def weight_init(m):
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, np.math.sqrt(2. / n))
    elif isinstance(m, nn.Linear):
        n = m.in_features * m.out_features
        m.weight.data.normal_(0, np.math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
