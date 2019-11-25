import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ts.utils.helper_funcs import BLOCK_TYPE


def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p < 10, "thetas_dim is too big."
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor([np.cos(2 * np.pi * i * t) for i in range(p1)]).float()  # H/2-1
    s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
    S = torch.cat([s1, s2])
    thetas_p = thetas.view(thetas.shape[0] * thetas.shape[1], thetas.shape[2])
    mm_t = torch.mm(thetas_p, S.to(device))
    if thetas.shape[1] == 1:
        return mm_t.unsqueeze(1)
    else:
        return mm_t.view(thetas.shape[0], thetas.shape[1], -1)


def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= 4, "thetas_dim is too big."
    T = torch.tensor([t ** i for i in range(p)]).float()
    thetas_p = thetas.view(thetas.shape[0] * thetas.shape[1], thetas.shape[2])
    mm_t = torch.mm(thetas_p, T.to(device))
    if thetas.shape[1] == 1:
        return mm_t.unsqueeze(1)
    else:
        return mm_t.view(thetas.shape[0], thetas.shape[1], -1)


def linspace(backcast_length, forecast_length):
    lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class Block(nn.Module):

    def __init__(self, block_type, id, units, thetas_dim, dropout, device, backcast_length=10, forecast_length=5,
                 share_thetas=False):
        super(Block, self).__init__()
        self.block_type = block_type
        self.id = id
        self.units = units
        self.thetas_dim = thetas_dim
        self.dropout = dropout
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(backcast_length, units)
        self.d1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(units, units)
        self.d2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(units, units)
        self.d3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(units, units)
        self.device = device
        self.backcast_linspace, self.forecast_linspace = linspace(backcast_length, forecast_length)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim)
            self.theta_f_fc = nn.Linear(units, thetas_dim)
        self.backcasts = []
        self.forecasts = []

    def forward(self, x):
        x = F.relu(self.fc1(x.to(self.device)))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = self.d2(x)
        x = F.relu(self.fc3(x))
        x = self.d3(x)
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f"{block_type}(id={self.id}, units={self.units}, thetas_dim={self.thetas_dim}, dropout={self.dropout}, " \
               f"backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, " \
               f"share_thetas={self.share_thetas}) at @{id(self)}"


class SeasonalityBlock(Block):

    def __init__(self, block_type, id, units, thetas_dim, dropout, device, backcast_length=10, forecast_length=5):
        super(SeasonalityBlock, self).__init__(block_type, id, units, thetas_dim, dropout, device, backcast_length,
                                               forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        self.backcasts.append(backcast)
        self.forecasts.append(forecast)
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, block_type, id, units, thetas_dim, dropout, device, backcast_length=10, forecast_length=5):
        super(TrendBlock, self).__init__(block_type, id, units, thetas_dim, dropout, device, backcast_length,
                                         forecast_length, share_thetas=True)

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        # print("Adding backcast/forecast for block:{}".format(self.id))
        self.backcasts.append(backcast)
        self.forecasts.append(forecast)
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, block_type, id, units, thetas_dim, dropout, device, backcast_length=10, forecast_length=5):
        super(GenericBlock, self).__init__(block_type, id, units, thetas_dim, dropout, device, backcast_length,
                                           forecast_length)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        # no constraint for generic arch.
        x = super(GenericBlock, self).forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        x = F.dropout(x, 0.2)
        theta_f = F.relu(self.theta_f_fc(x))

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.
        self.backcasts.append(backcast)
        self.forecasts.append(forecast)

        return backcast, forecast


class NBeatsNet(nn.Module):

    def __init__(self,
                 device,
                 stack_types=[BLOCK_TYPE.TREND, BLOCK_TYPE.SEASONALITY],
                 nb_blocks_per_stack=3,
                 forecast_length=5,
                 backcast_length=10,
                 thetas_dims=[4, 8],
                 share_weights_in_stack=False,
                 hidden_layer_units=256,
                 dropout=0, ):
        super(NBeatsNet, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dims
        self.parameters = []
        self.dropout = dropout
        self.device = device
        print(f"| N-Beats")
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(f"| --  Stack {stack_type.value} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})")
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one to make the stack
            else:
                block_id_str = str(stack_id) + "_" + str(block_id)
                print("Creating block:{}".format(block_id_str))
                block = block_init(stack_type, block_id_str, self.hidden_layer_units, self.thetas_dim[stack_id],
                                   self.dropout, self.device, self.backcast_length, self.forecast_length)
                self.parameters.extend(block.parameters())
            print(f"     | -- {block}")
            blocks.append(block)
        return blocks

    def get_block(self, stack_id, block_id):
        if stack_id > len(self.stacks):
            print("Invalid stack id!!!")
            return None
        if block_id > len(self.stacks[stack_id]):
            print("Invalid block id!!!")
            return None
        return self.stacks[stack_id][block_id]

    @staticmethod
    def select_block(block_type):
        if block_type == BLOCK_TYPE.SEASONALITY:
            return SeasonalityBlock
        elif block_type == BLOCK_TYPE.TREND:
            return TrendBlock
        else:
            return GenericBlock

    def forward(self, backcast):
        forecast = torch.zeros(
            # size=(backcast.shape[0], self.forecast_length,))  # maybe batch size here.
            # size=(backcast.shape[1], backcast.shape[0], self.forecast_length,))  # maybe batch size here.
            size=(backcast.shape[0], backcast.shape[1], self.forecast_length,))  # maybe batch size here.
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast.to(self.device) - b
                forecast = forecast.to(self.device) + f
        return backcast, forecast
