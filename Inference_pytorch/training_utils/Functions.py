import torch
from torch import autograd

class QuantFunction(autograd.Function):
    @staticmethod
    def forward(ctx, N, input, input_range=None):
        integer_range = pow(2, N) - 1
        if input_range is None:
            det = input.abs().max() / integer_range
        else:
            det = input_range / integer_range
        if det == 0:
            return input
        else:
            return (input/det).round().clamp(-integer_range, integer_range) * det

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output, None

def is_nan(x):
    return torch.isnan(x).sum() != 0

def nan_print(x):
    x = x.tolist()
    for i in x:
        print(i)

def test_nan(exp, exp_sum, g_input, g_inputS, ratio):
    if is_nan(g_input) or is_nan(g_inputS):
        torch.save([exp.cpu().numpy(), exp_sum.cpu().numpy()], "debug.pt")
        print(is_nan(g_input), is_nan(g_inputS))
        raise Exception