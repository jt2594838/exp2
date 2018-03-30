import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.layer_num = config['layer_num']
        self.output_size = config['output_size']

        self.nn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.layer_num,
            batch_first=True
        )

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        r_out, (h_n, h_c) = self.nn(x, None)
        last_time = r_out.size(1) - 1
        out = self.out(r_out[:, last_time, :])
        return out
