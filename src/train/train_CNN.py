import time

import matplotlib.pylab as plt
import numpy as np
import torch.optim as optim
import torch.utils.data as data_util
from torch import nn
from torch.autograd import Variable

from data.PickleDataReader import PickleDataReader
from data.WindowDataSet import WindowDataSet
from models.BasicCNN import BasiCNN

# configs
input_size = 128
output_size = 8
data_stride = 8
train_ratio = 0.9
max_epoch = 1000
learn_rate = 1e-4
batch_size = 10
criterion = nn.MSELoss()
use_cuda = True
use_BN = True
drop_out_ratio = 0.5
thread_num = 6
data_file = '/home/jt/codes/exp2/data/cnn1.log.cache'
config = {'input_size': input_size, 'output_size': output_size, 'drop_out_ratio': drop_out_ratio, 'use_BN': use_BN}


def eval_net(net, data_loader, criterion, use_cuda=False):
    data = data_loader.__iter__().next()
    net_input = Variable(data['data'])
    target = Variable(data['label'], requires_grad=False).squeeze()
    if use_cuda:
        net_input = net_input.cuda()
        target = target.cuda()
    output = net(net_input)

    test_loss = criterion(output, target)

    return test_loss


def train_net(train_net, data_loader, train_epoch, learn_rate, use_cuda=False):
    t = time.time()
    opt = optim.Adam(train_net.parameters(), lr=learn_rate)
    # train
    loss_list = []
    for epoch in range(train_epoch):
        for i, data in enumerate(data_loader, 0):
            train_input = Variable(data['data'])
            target = Variable(data['label'], requires_grad=False).squeeze()
            if use_cuda:
                train_input = train_input.cuda()
                target = target.cuda()
            output = train_net(train_input)

            opt.zero_grad()
            train_loss = criterion(output, target)
            train_loss.backward()

            opt.step()
        loss_list.append(train_loss.data[0])

    plt.figure(0)
    plt.plot(np.arange(0, train_epoch), np.array(loss_list))
    print('time consumed %f s' % (time.time() - t))


if __name__ == '__main__':


    # initializations
    loader = PickleDataReader(data_file, input_size, output_size,
                              data_stride)
    train_set, test_set = loader.load(train_ratio)
    train_set = WindowDataSet(train_set)
    test_set = WindowDataSet(test_set)
    net = BasiCNN(config)
    if use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()
        thread_num = 1

    train_loader = data_util.DataLoader(train_set, batch_size=batch_size,
                                        shuffle=True, num_workers=thread_num)
    test_loader = data_util.DataLoader(test_set, batch_size=test_set.__len__(),
                                       shuffle=True, num_workers=thread_num)

    # train
    train_net(net, train_loader, max_epoch, learn_rate, use_cuda)
    # test
    loss = eval_net(net, test_loader, criterion, use_cuda)
    print("Test loss : %f" % loss)
    # plt.savefig('basicCNN_%s_%s_%s.png' % (max_epoch, learn_rate, loss.data[0]))
