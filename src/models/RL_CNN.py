from torch import nn

import models.BasicCNN as BasicCNN
from torch.autograd import Variable
import torch.optim as optim

from data.PickleDataLoader import PickleDataLoader
from data.WindowDataSet import WindowDataSet


class RL_CNN:
    def __init__(self, config, criterion) -> None:
        super().__init__()
        self.cnn = BasicCNN(config)
        self.opt = optim.Adam(self.cnn, lr=config['learn_rate'])
        self.criterion = criterion
        self.loss_list = []
        self.pred_reward = 0.0
        self.last_loss = 0.0

    def forward(self, data):
        net_out = self.cnn(data)
        return net_out

    def find_best(self, data):
        pred_vals = self.forward(data)
        reward = Variable(-10000)
        best_buy_index = None
        best_sell_index = None
        for i in range(len(pred_vals)):
            buy_point = pred_vals[i]
            for j in range(i + 1, len(pred_vals)):
                sell_point = pred_vals[j]
                if sell_point - buy_point > reward:
                    reward = sell_point - buy_point
                    best_buy_index = i
                    best_sell_index = j
        self.pred_reward = reward
        return reward, best_buy_index, best_sell_index

    def learn(self, true_reward):
        if true_reward is not None:
            loss = self.criterion(self.pred_reward, true_reward)
            loss.backward()
            self.last_loss = loss

    def update(self):
        self.loss_list.append(self.last_loss.data[0])
        self.opt.step()
        self.opt.zero_grad()


class Environment:

    def __init__(self, data_set) -> None:
        super().__init__()
        self.data_set = data_set
        self.index = 0
        self.limit = len(data_set)

    """
    :return
        current data sample from the data set
    """
    def get_curr_state(self):
        return self.data_set[self.index]['data']

    def step(self, buy_point, sell_point):
        reward = None
        if buy_point is not None and sell_point is not None:
            assert buy_point < sell_point
            real_prices = self.data_set[self.index]['label']
            reward = real_prices[sell_point] - real_prices[buy_point]
        self.index = self.index + 1
        return reward

    def is_over(self):
        return self.index < self.limit

    def reset(self):
        self.index = 0


def train_net(net, env, train_epoch, batch_size):
    for i in range(train_epoch):
        env.reset()
        j = 0
        while not env.is_over():
            j = j + 1
            curr_state = env.get_curr_state()
            _, best_buy_point, best_sell_point = net.find_best(curr_state)
            true_reward = env.step(best_buy_point, best_sell_point)
            net.learn(true_reward)

            if j % batch_size == 0:
                net.update()
        if j % batch_size != 0:
            net.update()


def eval_decision(net, env, criterion):
    env.reset()
    total_loss = 0.0
    cnt = 0
    while not env.is_over():
        curr_state = env.get_curr_state()
        reward, best_buy_point, best_sell_point = net.find_best(curr_state)
        true_reward = env.step(best_buy_point, best_sell_point)
        loss = criterion(true_reward, reward)
        total_loss += loss.data[0]
        cnt = cnt + 1
    return cnt, total_loss


def eval_net():
    pass


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
    config = {'input_size': input_size, 'output_size': output_size, 'learn_rate': learn_rate}

    # initializations
    loader = PickleDataLoader(data_file, input_size, output_size,
                              data_stride)
    train_set, test_set = loader.load(train_ratio)
    train_set = WindowDataSet(train_set)
    test_set = WindowDataSet(test_set)

    net = RL_CNN(config, criterion)
    env = Environment(train_set)

    train_net(net, env, max_epoch, batch_size)