import torch
from torch import nn
from training_utils.utils import get_dataset, get_model_cfg, prepare_model, NEachEval, NTrain, str2bool
import nas_utils
import argparse
import time

def get_cfg_from_rollout(rollout):
    rollout = nas_utils.rollout_str_to_list(rollout)
    cfg = nas_utils.generate_cfg(rollout)
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epoch', action='store', type=int, default=1,
            help='# of epochs of training')
    parser.add_argument('--dev_var', action='store', type=float, default=0.3,
            help='device variation [std] before write and verify')
    parser.add_argument('--device', action='store', default="cuda:0",
            help='device used')
    parser.add_argument('--verbose', action='store', type=str2bool, default=False,
            help='see training process')
    parser.add_argument('--model', action='store', default="vgg8",
            help='model to use')
    parser.add_argument('--div', type=int, action='store', default=1,
            help='deprecated')
    parser.add_argument('--wl_weight', type=int, action='store', default=4,
            help='weight / activation quantization')
    parser.add_argument('--rollout', action='store', default="[[2,0],[2,0],[3,0],[3,0],[4,0],[4,0]]",
            help='weight / activation quantization')
    parser.add_argument('--use_tqdm', action='store',type=str2bool, default=False,
            help='whether to use tqdm')
    args = parser.parse_args()

    header = time.time()
    device = torch.device("cuda:0")
    BS = 128
    NW = 4
    trainloader, secondloader, testloader = get_dataset(args, BS, NW)
    rollout = args.rollout
    cfg = get_cfg_from_rollout(rollout)
    model = get_model_cfg(cfg, args)
    model, optimizer, warm_optimizer, scheduler = prepare_model(model, device, args)
    criteriaF = nn.CrossEntropyLoss()
    model_group = model, criteriaF, optimizer, scheduler, device, trainloader, testloader
    NTrain(model_group, args.train_epoch, header, args.dev_var, verbose=True)
    state_dict = torch.load(f"tmp_best_{header}.pt")
    model.load_state_dict(state_dict)
    performance = NEachEval(model_group, args.dev_var)
    print(f"No mask noise acc: {performance:.4f}")