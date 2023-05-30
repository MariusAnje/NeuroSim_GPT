import torch
from torch import nn

system_content = "You are an expert in the field of neural architecture search."

user_input = '''Your task is to assist me in selecting the best channel numbers for a given model architecture. The model will be trained and tested on CIFAR10, and your objective will be to maximize the model's performance on CIFAR10. 

The model architecture will be defined as the following.
{
    layer1: nn.Conv2d(in_channels=3,             out_channels=rollout[0][0], kernel_size=rollout[0][1], padding=rollout[0][1]//2),
    layer2: nn.ReLU(),
    layer3: nn.Conv2d(in_channels=rollout[0][0], out_channels=rollout[1][0], kernel_size=rollout[1][1], padding=rollout[1][1]//2),
    layer4: nn.ReLU(),
    layer5: nn.MaxPool2d(2),
    layer6: nn.Conv2d(in_channels=rollout[1][0], out_channels=rollout[2][0], kernel_size=rollout[2][1], padding=rollout[2][1]//2),
    layer7: nn.ReLU(),
    layer8: nn.Conv2d(in_channels=rollout[2][0], out_channels=rollout[3][0], kernel_size=rollout[3][1], padding=rollout[3][1]//2),
    layer9: nn.ReLU(),
    layer10: nn.MaxPool2d(2),
    layer11: nn.Conv2d(in_channels=rollout[3][0], out_channels=rollout[4][0], kernel_size=rollout[4][1], padding=rollout[4][1]//2),
    layer12: nn.ReLU(),
    layer13: nn.Conv2d(in_channels=rollout[4][0], out_channels=rollout[5][0], kernel_size=rollout[5][1], padding=rollout[5][1]//2),
    layer14: nn.ReLU(),
    layer15: nn.MaxPool2d(2),
    layer16: nn.Linear(in_features=rollout[5][0]*16, out_features=1024),
    layer17: nn.ReLU(),
    layer18: nn.Linear(in_features=1024, out_features=10),
}

For the `rollout` variable, the available number for each index would be:
{
    rollout[0][0]: [32, 64, 128, 256, 512],
    rollout[1][0]: [32, 64, 128, 256, 512],
    rollout[2][0]: [32, 64, 128, 256, 512],
    rollout[3][0]: [32, 64, 128, 256, 512],
    rollout[4][0]: [32, 64, 128, 256, 512],
    rollout[5][0]: [32, 64, 128, 256, 512],
    rollout[0][1]: [3, 5, 7],
    rollout[1][1]: [3, 5, 7],
    rollout[2][1]: [3, 5, 7],
    rollout[3][1]: [3, 5, 7],
    rollout[4][1]: [3, 5, 7],
    rollout[5][1]: [3, 5, 7],
}

Your objective is to define the optimal number of rollout for each layer based on the given options above to maximize the model's performance on CIFAR10. 
The model's performance is a combination of hardware performance and model accuracy. If the hardware is invalid (e.g., too large in area), the performance I give you will be -1. After you give me a rollout list, I will give you the model's performance I calculated.
Your response should be the a rollout list consisting of 6 numbers pairs(e.g. [[32,3],[32,3],[64,3],[64,3],[128,3],[128,3]]).
'''

experiments_prompt = lambda arch_list, acc_list : '''Here are some experimental results that you can use as a reference:
{}
Please suggest a channel list that can improve the model's performance on CIFAR10 beyond the experimental results provided above.
'''.format(''.join(['{} gives an performance of {:.4f}\n'.format(arch, acc) for arch, acc in zip(arch_list, acc_list)]))

suffix = '''Please do not include anything else other than the channel list in your response.'''
