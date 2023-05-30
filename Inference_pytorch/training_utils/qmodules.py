import torch
from training_utils.Functions import QuantFunction
from torch import nn

quant = QuantFunction.apply
_moving_momentum = 0.9

class NModule(nn.Module):
    def set_noise(self, dev_var):
        self.noise = torch.randn_like(self.op.weight) * dev_var
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.op.weight)
    
    def push_S_device(self):
        self.noise = self.noise.to(self.op.weight.device)
        try:
            self.input_range = self.input_range.to(self.op.weight.device)
        except:
            pass
    
    def normalize(self):
        if self.original_w is None:
            self.original_w = self.op.weight.data
        if (self.original_b is None) and (self.op.bias is not None):
            self.original_b = self.op.bias.data
        scale = self.op.weight.data.abs().max().item()
        self.scale = scale
        self.op.weight.data = self.op.weight.data / scale


class NModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def set_noise(self, dev_var):
        for mo in self.modules():
            if isinstance(mo, NModule):
                mo.set_noise(dev_var)

    def clear_noise(self):
        for m in self.modules():
            if isinstance(m, NModule):
                m.clear_noise()
    
    def push_S_device(self):
        for m in self.modules():
            if isinstance(m, NModule):
                m.push_S_device()
    
    def de_normalize(self):
        for mo in self.modules():
            if isinstance(mo, NModule):
                if mo.original_w is None:
                    raise Exception("no original weight")
                else:
                    mo.scale = 1.0
                    mo.op.weight.data = mo.original_w
                    mo.original_w = None
                    if mo.original_b is not None:
                        mo.op.bias.data = mo.original_b
                        mo.original_b = None
    
    def normalize(self):
        for mo in self.modules():
            if isinstance(mo, NModule):
                mo.normalize()
    
    def unpack_flattern(self, x):
        return x.view(-1, self.num_flat_features(x))
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class QNLinear(NModule):
    def __init__(self, N, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.noise = torch.zeros_like(self.op.weight)
        self.function = nn.functional.linear
        self.N = N
        self.scale = 1.0
        self.register_buffer('input_range', torch.zeros(1))

    def forward(self, x):
        # x = x = self.function(x, quant(self.N,self.op.weight) + self.noise, None)
        x = x = self.function(x, quant(self.N,self.op.weight) + self.noise, None)
        x = x * self.scale
        if self.op.bias is not None:
            x += self.op.bias
        # x = self.function(x, (quant(self.N, self.op.weight) + self.noise)  * self.scale , quant(self.N, self.op.bias))
        if self.training:
            this_max = x.abs().max().item()
            if self.input_range == 0:
                self.input_range += this_max
            else:
                self.input_range = self.input_range * _moving_momentum + this_max * (1-_moving_momentum)
        return quant(self.N, x, self.input_range)

class QNConv2d(NModule):
    def __init__(self, N, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.noise = torch.zeros_like(self.op.weight)
        self.function = nn.functional.conv2d
        self.N = N
        self.scale = 1.0
        self.register_buffer('input_range', torch.zeros(1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.zero_()

    def forward(self, x):
        # x = self.function(x, quant(self.N, self.op.weight) + self.noise, None, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        x = self.function(x, quant(self.N, self.op.weight) + self.noise, None, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        x = x * self.scale
        if self.op.bias is not None:
            x += quant(self.N, self.op.bias).reshape(1,-1,1,1).expand_as(x)
        if self.training:
            this_max = x.abs().max().item()
            if self.input_range == 0:
                self.input_range += this_max
            else:
                self.input_range = self.input_range * _moving_momentum + this_max * (1-_moving_momentum)
        return quant(self.N, x, self.input_range)
