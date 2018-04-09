import torch.utils.data as data_util
from torch import nn
from torch.autograd import Variable

from data.PickleDataReader import PickleDataReader
from data.RNNDataSet import RNNDataSet
from models.RNN import RNN

import torch.optim as optim
import matplotlib.pylab as plt
import time

import numpy as np

# configs
input_size = 16  # 16
timestamp = 4  # 4  -- 0.0006
output_size = 8  # 8
hidden_size = 1024  # 1024  large hidden_size ,small layer_num
layer_num = 1  # 1
data_stride = 8  # 8
train_ratio = 0.9  # 0.9
max_epoch = 1000  # 1000
learn_rate = 1e-4  # 1e-4
batch_size = 64  # 64
criterion = nn.MSELoss()
use_cuda = True
use_BN = True
thread_num = 6
data_file = '/home/jt/codes/exp2/data/cnn1.log.cache'
config = {'input_size': input_size, 'output_size': output_size, 'learn_rate': learn_rate,
          'batch_size': batch_size, 'hidden_size': hidden_size, 'layer_num': layer_num, 'use_BN': use_BN,
          'use_cuda': use_cuda
          }


def train_net(train_net, criterion, data_loader, train_epoch, learn_rate):
    curr_time = time.time()
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
    # plt.ylim([0, 0.002])
    # plt.show()
    print('time consumption : %f s, loss : %.10f ' % ((time.time() - curr_time), loss_list[-1]))


def eval_net(net, data_loader, criterion):
    data = data_loader.__iter__().next()
    net_input = Variable(data['data'])
    target = Variable(data['label'], requires_grad=False).squeeze()
    output = net(net_input)

    test_loss = criterion(output, target)

    return test_loss


if __name__ == '__main__':

    net = RNN(config)
    loader = PickleDataReader(data_file, input_size, output_size,
                              data_stride)
    train_set, test_set = loader.load(train_ratio, False)
    train_set = RNNDataSet(train_set, timestamp)
    test_set = RNNDataSet(test_set, timestamp)

    if use_cuda:
        thread_num = 1

    train_loader = data_util.DataLoader(train_set, batch_size=batch_size,
                                        shuffle=True, num_workers=thread_num)
    test_loader = data_util.DataLoader(test_set, batch_size=test_set.__len__(),
                                       shuffle=True, num_workers=thread_num)

    train_net(net, criterion, train_loader, max_epoch, learn_rate)
    loss = eval_net(net, test_loader, criterion)
    print("test loss : %.10f" % loss.data[0])
