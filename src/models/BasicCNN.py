import torch.utils.data as data_util
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pylab as plt
import numpy as np

from data.PickleDataLoader import PickleDataLoader
from data.WindowDataSet import WindowDataSet

KERNEL_SIZE = 5


class BasiCNN(nn.Module):

    def __init__(self, config):
        super(BasiCNN, self).__init__()
        input_size = int(config['input_size'])
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=KERNEL_SIZE),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # 16 channels
        self.pred_input_size = self.cal_size(input_size) * 16
        self.pred_output_size = int(config['output_size'])
        self.pred = nn.Sequential(
            nn.Linear(self.pred_input_size, int(self.pred_input_size / 2)),
            nn.ReLU(),
            nn.Linear(int(self.pred_input_size / 2), int(self.pred_input_size / 4)),
            nn.ReLU(),
            nn.Linear(int(self.pred_input_size / 4), int(self.pred_input_size / 8)),
            nn.ReLU(),
            nn.Linear(int(self.pred_input_size / 8), self.pred_output_size),
        )

    def forward(self, x):
        conved = self.conv(x).view(-1, self.pred_input_size)
        result = self.pred(conved)
        return result

    def cal_size(self, input_size):
        for i in range(4):
            input_size = input_size - KERNEL_SIZE + 1
            input_size = int(input_size / 2)
        return input_size


def eval_net(net, data_loader, criterion):
    data = data_loader.__iter__().next()
    net_input = Variable(data['data'])
    target = Variable(data['label'], requires_grad=False).squeeze()
    output = net(net_input)

    test_loss = criterion(output, target)

    return test_loss


def train_net(train_net, data_loader, train_epoch, learn_rate):
    opt = optim.Adam(train_net.parameters(), lr=learn_rate)
    # train
    loss_list = []
    for epoch in range(train_epoch):
        for i, data in enumerate(data_loader, 0):
            train_input = Variable(data['data'])
            target = Variable(data['label'], requires_grad=False).squeeze()
            output = train_net(train_input)

            opt.zero_grad()
            train_loss = criterion(output, target)
            train_loss.backward()

            opt.step()
        loss_list.append(train_loss.data[0])

    plt.figure(0)
    plt.plot(np.arange(0, train_epoch), np.array(loss_list))
    plt.ylim([0, 0.002])


if __name__ == '__main__':
    # configs
    input_size = 128
    output_size = 8
    data_stride = 8
    train_ratio = 0.9
    max_epoch = 1000
    learn_rate = 1e-4
    batch_size = 10
    criterion = nn.MSELoss()
    data_file = '/home/cdx4838/PycharmProjects/exp2/exp2/data/cnn1.log.cache'
    config = {'input_size': input_size, 'output_size': output_size}

    # initializations
    loader = PickleDataLoader(data_file, input_size, output_size,
                              data_stride)
    train_set, test_set = loader.load(train_ratio)
    train_set = WindowDataSet(train_set)
    test_set = WindowDataSet(test_set)
    net = BasiCNN(config)

    train_loader = data_util.DataLoader(train_set, batch_size=10,
                                        shuffle=True, num_workers=2)
    test_loader = data_util.DataLoader(test_set, batch_size=test_set.__len__(),
                                       shuffle=True, num_workers=2)

    # train
    train_net(net, train_loader, max_epoch, learn_rate)
    # test
    loss = eval_net(net, test_loader, criterion)
    print("Test loss : %f" % loss)
    plt.savefig('basicCNN_%s_%s_%s.png' % (max_epoch, learn_rate, loss.data[0]))
