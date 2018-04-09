import torch

from models.BasicCNN import BasiCNN
from models.RNN import RNN


class Generator:
    def __init__(self, config) -> None:
        super().__init__()
        self.nn = RNN(config)

    def gen(self, x):
        output = self.nn(x)
        return output


class Discriminator:
    def __init__(self, config) -> None:
        super().__init__()
        cnn_input_size = config['input_size'] * config['timestamp'] + config['output_size']
        cnn_output_size = 1
        self.nn = BasiCNN({'input_size': cnn_input_size, 'output_size': cnn_output_size,
                            'drop_out_ratio': config['drop_out_ratio'], 'use_BN': config['use_BN']})

    def discriminate(self, x):
        output = self.nn(x)
        output = torch.sigmoid(output)
        return output


class GAN_RNN:

    def __init__(self, config) -> None:
        super().__init__()
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)

    def gen(self, x):
        return self.generator.gen(x)

    def discriminate(self, x):
        return self.discriminator.discriminate(x)
