import time
import torch
import random
from torch import nn

import models.BasicCNN as BasicCNN
from torch.autograd import Variable
import torch.optim as optim

from data.PickleDataReader import PickleDataReader
from data.WindowDataSet import WindowDataSet
from models.Transition import Transition, TransitionContainer

BUY = 0
SELL = 1

"""
    This agent is given a window of data, and its mission is to decide to buy with all its money or sell all its share.
    The output 1*2 vector represent the reward of buy and sell respectively.
"""
class DQN:
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.eval_net = BasicCNN.BasiCNN(config)
        print(self.eval_net)
        self.target_net = BasicCNN.BasiCNN(config)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.opt = optim.Adam(self.eval_net.parameters(), lr=config['learn_rate'])
        self.criterion = config['criterion']
        self.transition_mem = TransitionContainer(config['transition_mem_size'])
        self.transition_threshold = config['transition_threshold']
        self.update_period = config['update_period']
        self.update_counter = 0
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.greedy = config['greedy']
        self.total_loss = 0.0

    def take_action(self, data, use_greedy=False):
        if use_greedy and random.uniform(0, 1) > self.greedy:
            return 0, random.randint(0, 1)
        else:
            prd_val = self.eval_net(data)
            if prd_val.data[0, BUY] > prd_val.data[0, SELL]:
                return prd_val[0, BUY], BUY
            else:
                return prd_val[0, SELL], SELL

    def store_transition(self, transition):
        self.transition_mem.put(transition)

    def enough_transistion(self):
        return self.transition_mem.size() >= self.transition_threshold

    def learn(self):
        if not self.enough_transistion():
            return
        transitions = self.transition_mem.sample(self.batch_size)
        # concat the transitions as a minibatch
        curr_state_batch = transitions[0].prev_stat
        prd_act_batch = transitions[0].prd_act
        next_state_batch = transitions[0].next_stat
        real_reward_batch = transitions[0].real_val
        for i in range(1, len(transitions)):
            curr_state_batch = torch.cat((curr_state_batch, transitions[i].prev_stat), 0)
            prd_act_batch = torch.cat((prd_act_batch, transitions[i].prd_act), 0)
            next_state_batch = torch.cat((next_state_batch, transitions[i].next_stat), 0)
            real_reward_batch = torch.cat((real_reward_batch, transitions[i].real_val), 0)
        curr_state_batch = Variable(curr_state_batch)
        next_state_batch = Variable(next_state_batch)
        real_reward_batch = Variable(real_reward_batch).squeeze()
        # train the eval net
        ## calculate the rewards given by eval net
        eval_value = self.eval_net(curr_state_batch)
        eval_reward = Variable(torch.zeros(self.batch_size))
        for i in range(self.batch_size):
            act = prd_act_batch[i, 0, 0]
            eval_reward[i] = eval_value[i, act]
        ## calculate the rewards given by a combination of taeget net and real reward
        target_reward = real_reward_batch
        for i in range(self.batch_size):
            reward, _ = self.take_action(torch.unsqueeze(next_state_batch[i, :, :], 1))
            target_reward[i] = target_reward[i] + self.gamma * reward
        ## caculate the loss
        loss = self.criterion(eval_reward, target_reward.detach())
        self.total_loss += loss.data[0]
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.update_counter = self.update_counter + 1
        # update target net if necessary
        if self.update_counter % self.update_period == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())


class Environment:

    def __init__(self, data_set) -> None:
        super().__init__()
        self.data_set = data_set
        self.index = 0
        self.limit = len(data_set)
        self.fund = 1.0
        self.share = 0.0
        self.curr_price = 0.0
        self.curr_asset = 1.0
        self.prev_asset = 1.0

    """
    :return
        current data sample from the data set
    """

    def get_curr_state(self):
        return self.data_set[self.index]['data']

    def step(self, act):
        self.prev_asset = self.curr_asset
        price_list = torch.squeeze(self.data_set[self.index]['data'])
        self.curr_price = price_list[len(price_list) - 1]
        if act == BUY and self.fund > 0:
            self.share += self.fund / self.curr_price
            self.fund = 0.0
            # print("buy at %f" % self.curr_price)
        elif act == SELL and self.share > 0:
            self.fund += self.share * self.curr_price
            self.share = 0.0
            # print("sell at %f" % self.curr_price)
        self.index += 1
        self.curr_asset = self.fund + self.share * self.curr_price
        return self.get_reward()

    def get_reward(self):
        return self.curr_asset - self.prev_asset

    def is_over(self):
        return self.index >= self.limit

    def reset(self):
        self.index = 0
        self.fund = 1.0
        self.share = 0.0
        self.curr_price = 0.0
        self.curr_asset = 1.0
        self.prev_asset = 1.0



