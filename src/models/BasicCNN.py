
import torch.nn as nn
import numpy as np

KERNEL_SIZE = 3


class BasiCNN(nn.Module):

    """
        a stupid way of construction in order to retain internal results
    """
    def __init__(self, config):
        super(BasiCNN, self).__init__()
        input_size = int(config['input_size'])
        drop_out_ratio = config['drop_out_ratio']
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=KERNEL_SIZE)
        self.dropOut1 = nn.Dropout(drop_out_ratio)
        self.BN1 = nn.BatchNorm1d(2)
        self.ReLU1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=KERNEL_SIZE)
        self.dropOut2 = nn.Dropout(drop_out_ratio)
        self.BN2 = nn.BatchNorm1d(4)
        self.ReLU2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=KERNEL_SIZE)
        self.dropOut3 = nn.Dropout(drop_out_ratio)
        self.BN3 = nn.BatchNorm1d(8)
        self.ReLU3 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=KERNEL_SIZE)
        self.dropOut4 = nn.Dropout(drop_out_ratio)
        self.BN4 = nn.BatchNorm1d(16)
        self.ReLU4 = nn.ReLU()

        self.pred_input_size = self.cal_size(input_size)
        self.pred_output_size = int(config['output_size'])

        self.Linear1 = nn.Linear(self.pred_input_size, int(self.pred_input_size / 2))
        self.linDropOut1 = nn.Dropout(drop_out_ratio)
        self.LinBN1 = nn.BatchNorm1d(int(self.pred_input_size / 2))
        self.LinReLu1 = nn.ReLU()

        self.Linear2 = nn.Linear(int(self.pred_input_size / 2), int(self.pred_input_size / 4))
        self.linDropOut2 = nn.Dropout(drop_out_ratio)
        self.LinBN2 = nn.BatchNorm1d(int(self.pred_input_size / 4))
        self.LinReLu2 = nn.ReLU()

        self.Linear3 = nn.Linear(int(self.pred_input_size / 4), int(self.pred_input_size / 8))
        self.linDropOut3 = nn.Dropout(drop_out_ratio)
        self.LinBN3 = nn.BatchNorm1d(int(self.pred_input_size / 8))
        self.LinReLu3 = nn.ReLU()

        self.output = nn.Linear(int(self.pred_input_size / 8), self.pred_output_size)

        self.apply(weight_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropOut1(x)
        x = self.BN1(x)
        x = self.ReLU1(x)

        x = self.conv2(x)
        x = self.dropOut2(x)
        x = self.BN2(x)
        x = self.ReLU2(x)

        x = self.conv3(x)
        x = self.dropOut3(x)
        x = self.BN3(x)
        x = self.ReLU3(x)

        x = self.conv4(x)
        x = self.dropOut4(x)
        x = self.BN4(x)
        x = self.ReLU4(x)

        x = x.view(x.size(0), -1)

        x = self.Linear1(x)
        x = self.linDropOut1(x)
        x = self.LinBN1(x)
        x = self.LinReLu1(x)

        x = self.Linear2(x)
        x = self.linDropOut2(x)
        x = self.LinBN2(x)
        x = self.LinReLu2(x)

        x = self.Linear3(x)
        x = self.linDropOut3(x)
        x = self.LinBN3(x)
        x = self.LinReLu3(x)

        x = self.output(x)

        return x

    def cal_size(self, input_size):
        input_size = input_size - KERNEL_SIZE + 1
        input_size = input_size - KERNEL_SIZE + 1
        # input_size = int(input_size / 2)
        input_size = input_size - KERNEL_SIZE + 1
        input_size = input_size - KERNEL_SIZE + 1
        # input_size = int(input_size / 2)
        return input_size * 16


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
