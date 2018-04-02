import time
import torch
import torch.optim as optim
import torch.utils.data as data_util
from torch.autograd import Variable

from data.PickleDataReader import PickleDataReader
from data.RNNDataSet import RNNDataSet
from models.GAN import GAN
import matplotlib.pylab as plt

import numpy as np

# configs
input_size = 16  # 16
timestamp = 4  # 4  -- 0.0006
output_size = 8  # 8
hidden_size = 256  # 1024  large hidden_size ,small layer_num
layer_num = 1  # 1
data_stride = 16  # 8
train_ratio = 0.9  # 0.9
pre_train_epoch = 10
max_epoch = 200  # 1000
learn_rate = 5e-4  # 1e-4
batch_size = 64  # 64
dis_gen_train_ratio = 5
criterion = torch.nn.MSELoss()
data_file = '/home/cdx4838/PycharmProjects/exp2/exp2/data/cnn2.log.cache'
config = {'input_size': input_size, 'output_size': output_size, 'learn_rate': learn_rate,
          'batch_size': batch_size, 'hidden_size': hidden_size, 'layer_num': layer_num,
          'timestamp': timestamp
          }


def pre_train_net(train_net, criterion, data_loader, train_epoch, learn_rate):
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

    # plt.figure(0)
    # plt.plot(np.arange(0, train_epoch), np.array(loss_list))
    # plt.ylim([0, 0.002])
    # plt.show()
    print('pre train time consumption : %f s, loss : %.10f ' % ((time.time() - curr_time), loss_list[-1]))


def train_net(train_net, data_loader, train_epoch, learn_rate):
    curr_time = time.time()
    gen_opt = optim.Adam(train_net.generator.nn.parameters(), lr=learn_rate)
    dis_opt = optim.Adam(train_net.discriminator.nn.parameters(), lr=learn_rate)
    criterion = torch.nn.MSELoss()
    # train
    gen_loss_list = []
    dis_loss_list = []
    dif_list = []
    for epoch in range(train_epoch):
        for i, sample in enumerate(data_loader, 0):
            data = sample['data']
            label = sample['label']
            gen_input = Variable(data)
            gen_output = train_net.generator.gen(gen_input)

            gen_input = gen_input.view(gen_input.size(0), -1)
            dis_gen_input = torch.cat((gen_input, gen_output), 1)
            dis_gen_output = train_net.discriminator.discriminate(dis_gen_input.unsqueeze(1))

            dis_label_input = Variable(
                torch.cat((data.view(gen_input.size(0), -1), label.view(gen_input.size(0), -1)), 1))
            dis_label_output = train_net.discriminator.discriminate(dis_label_input.unsqueeze(1))

            gen_loss = torch.mean(torch.log(1.0 - dis_gen_output))
            dis_loss = - torch.mean(torch.log(dis_label_output) + torch.log(1.0 - dis_gen_output))

            dis_opt.zero_grad()
            dis_loss.backward(retain_graph=True)
            dis_opt.step()

            if (i + 1) % dis_gen_train_ratio == 0:
                gen_opt.zero_grad()
                gen_loss.backward()
                gen_opt.step()

        gen_loss_list.append(gen_loss.data[0])
        dis_loss_list.append(dis_loss.data[0])
        dif_list.append(criterion(dis_gen_input, dis_label_input).data[0])

    print("time consumption : %f s" % (time.time() - curr_time))

    plt.figure()
    plt.plot(np.arange(0, train_epoch), np.array(gen_loss_list))
    plt.title("generator")

    plt.figure()
    plt.plot(np.arange(0, train_epoch), np.array(dis_loss_list))
    plt.title("discriminator")

    plt.figure()
    plt.plot(np.arange(0, train_epoch), np.array(dif_list))
    plt.title("difference")

    plt.show()


if __name__ == '__main__':
    # initializations
    net = GAN(config)
    loader = PickleDataReader(data_file, input_size, output_size,
                              data_stride)
    train_set, test_set = loader.load(train_ratio, False)
    train_set = RNNDataSet(train_set, timestamp)
    test_set = RNNDataSet(test_set, timestamp)

    train_loader = data_util.DataLoader(train_set, batch_size=batch_size,
                                        shuffle=True, num_workers=2)
    test_loader = data_util.DataLoader(test_set, batch_size=test_set.__len__(),
                                       shuffle=True, num_workers=2)

    pre_train_net(net.generator.nn, criterion, train_loader, pre_train_epoch, learn_rate)
    train_net(net, train_loader, max_epoch, learn_rate)
