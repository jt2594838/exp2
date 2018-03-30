import time

import torch
from torch import nn
from torch.autograd import Variable

from data.PickleDataReader import PickleDataReader
from data.WindowDataSet import WindowDataSet
from models.DQN import DQN, Environment
from models.Transition import Transition


def train_net(dqn, env, train_epoch, decay):
    total_time = 0
    for i in range(train_epoch):
        curr_time = time.time()
        env.reset()
        while not env.is_over():
            """
                let the eval net take an action and store the transition
            """
            curr_state = env.get_curr_state()
            prd_reward, act = dqn.take_action(Variable(curr_state), use_greedy=True)
            reward = env.step(act)
            if env.is_over():
                break
            next_state = env.get_curr_state()
            transition = Transition(curr_state, [act], next_state, [reward])
            dqn.store_transition(transition)
            dqn.learn()
        elapsed_time = time.time() - curr_time
        total_time += elapsed_time
        print('epoch %d : consumed %d ms, total %d ms, loss %.10f asset %f' % (
            i, elapsed_time * 1000, total_time * 1000, dqn.total_loss, env.curr_asset))
        dqn.greedy = dqn.greedy + (1 - dqn.greedy) * (1 - decay)
        dqn.total_loss = 0.0


def eval_decision(net, env, criterion):
    env.reset()
    total_loss = 0.0
    total_reward = 0.0
    cnt = 0
    while not env.is_over():
        curr_state = env.get_curr_state()
        reward, act = net.take_action(Variable(curr_state), use_greedy=True)
        true_reward = Variable(torch.FloatTensor([env.step(act)]), requires_grad=False)
        loss = criterion(reward, true_reward)
        total_loss += loss.data[0]
        total_reward += true_reward
        cnt = cnt + 1
    return cnt, total_loss, total_reward


if __name__ == '__main__':
    # configs
    input_size = 128
    output_size = 2
    data_stride = 8
    train_ratio = 1.0
    max_epoch = 100
    learn_rate = 1e-3
    batch_size = 64
    gamma = 0.9
    transition_mem_size = 5000
    transition_threshold = batch_size
    update_period = 10
    greedy = 0.9
    non_greedy_decay = 0.9
    criterion = nn.MSELoss()
    data_file = '/home/cdx4838/PycharmProjects/exp2/exp2/data/cnn1.log.cache'
    config = {'input_size': input_size, 'output_size': output_size, 'learn_rate': learn_rate,
              'batch_size': batch_size, 'gamma': gamma, 'transition_mem_size': transition_mem_size,
              'transition_threshold': transition_threshold, 'criterion': criterion, 'update_period': update_period,
              'greedy': greedy}

    # initializations
    loader = PickleDataReader(data_file, input_size, output_size,
                              data_stride)
    train_set, test_set = loader.load(train_ratio, False)
    train_set = WindowDataSet(train_set)

    net = DQN(config)
    env = Environment(train_set)

    train_net(net, env, max_epoch, non_greedy_decay)
    cnt, loss, reward = eval_decision(net, env, criterion)
    print("step : %d, loss : %f, reward : %f" % (cnt, loss, reward))
