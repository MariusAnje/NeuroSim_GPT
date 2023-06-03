import numpy as np
import os
import subprocess
import openai
import prompts
import nas_utils
import argparse
from training_utils.utils import str2bool

def get_float(line, start, offset=None, back=None):
    start = line.find(start) + len(start)
    if offset is not None:
        return float(line[start:start+offset])
    else:
        return float(line[start:-back])

def parse_neurosim_res(res):
    res = res.splitlines()
    flag = False
    for l in res:
        l = str(l)
        if "------------------------------ Summary --------------------------------" in l:
            flag = True
        if flag:
            if "ChipArea :" in l:
                area = get_float(l, "ChipArea :", None, 5)
            if "Chip layer-by-layer readLatency (per image) is: " in l: # ns
                latency = get_float(l, "Chip layer-by-layer readLatency (per image) is: ", None, 3)
            if "Chip total readDynamicEnergy is:" in l: # pJ
                d_energy = get_float(l, "Chip total readDynamicEnergy is:", None, 3)
            if "Chip total leakage Energy is:" in l: # pJ
                l_energy = get_float(l, "Chip total leakage Energy is:", None, 3)
    energy = d_energy + l_energy
    power = energy / latency # mW
    fps = 1 / (latency/(1e9))
    hw_p_dict = {"area":area,"latency":latency,"energy":energy,"power":power,"fps":fps}
    return hw_p_dict

def parse_ntrain_res(res):
    res = res.splitlines()
    flag = False
    for l in res:
        l = str(l)
        if "No mask noise acc: " in l:
            acc = get_float(l, "No mask noise acc: ", None, 1)
    return acc

def tradeoff_peformance(accuracy, hw_performance):
    return accuracy + hw_performance["fps"]/1600

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epoch', action='store', type=int, default=100,
            help='# of epochs of training')
    parser.add_argument('--rollout_id', action='store', type=int, default=-1,
            help='which rollout to use')
    parser.add_argument('--dev_var', action='store', type=float, default=0.04,
            help='device variation [std] before write and verify')
    parser.add_argument('--device', action='store', default="cuda:0",
            help='device used')
    parser.add_argument('--wl_weight', type=int, action='store', default=5,
            help='weight / activation quantization')
    parser.add_argument('--use_tqdm', action='store',type=str2bool, default=False,
            help='whether to use tqdm')
    args = parser.parse_args()

    dev_var = args.dev_var
    train_epoch = args.train_epoch
    openai.api_key = "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"
    system_content, user_input, experiments_prompt, suffix = prompts.system_content, prompts.user_input, prompts.experiments_prompt, prompts.suffix
    arch_list = []
    acc_list = []

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input + suffix},
    ]

    result_dict = {}
    for _ in range(1):
        # rep = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=messages, temperature=0, n=1)['choices'][0]['message']
        # print(messages)
        # rollout = parse_openai_reply(rep)
        # print(rep)
        # print(messages[:100])
        # rollout = []
        # for _ in range(6):
        #     p = [np.random.randint(5), np.random.randint(3)]
        #     rollout.append(p)
        # # rollout = [[2,0],[2,0],[3,0],[3,0],[4,0],[4,0]]
        rollout_l =  [
                [[32, 3], [64, 3], [128, 3], [256, 3], [512, 3], [512, 3]],
                [[32, 3], [64, 3], [128, 5], [256, 3], [512, 3], [512, 5]],
                [[32, 3], [64, 3], [128, 5], [256, 5], [512, 3], [512, 7]],
                [[32, 5], [64, 5], [128, 5], [256, 5], [512, 7], [512, 7]],
                [[64, 3], [128, 3], [256, 3], [256, 5], [512, 5], [512, 5]],
                [[64, 3], [128, 3], [256, 5], [256, 5], [512, 5], [512, 7]],
                [[64, 3], [128, 5], [256, 5], [256, 5], [512, 7], [512, 7]],
                [[64, 3], [128, 3], [256, 5], [512, 5], [512, 7], [256, 7]],
                [[64, 3], [128, 5], [256, 7], [512, 5], [512, 7], [128, 7]],
                [[64, 5], [128, 5], [256, 5], [256, 7], [512, 7], [128, 7]],
                [[64, 3], [128, 5], [256, 7], [256, 7], [512, 5], [256, 5]],
                [[64, 5], [128, 3], [256, 7], [256, 5], [512, 7], [256, 3]],
                [[64, 5], [128, 5], [256, 3], [256, 5], [512, 5], [256, 3]],
                [[64, 3], [128, 5], [256, 3], [512, 7], [512, 3], [256, 5]],
                [[64, 5], [128, 3], [256, 7], [512, 3], [512, 5], [256, 7]],
                [[64, 5], [128, 3], [256, 5], [512, 3], [512, 5], [256, 5]],
                [[64, 3], [128, 5], [256, 3], [256, 7], [512, 5], [512, 7]]
            ]
        rollout = rollout_l[args.rollout_id]
        rollout = nas_utils.gpt_to_rollout(rollout)
        rollout = str(rollout)
        if rollout in result_dict:
            hw_performance, accuracy = result_dict[rollout]
            print(f"rollout: {rollout}  ", hw_performance)
            performance = tradeoff_peformance(accuracy, hw_performance)
        else:
            try:
                hw_res = subprocess.check_output([f"python", f"hw_ana.py", f"--rollout", f"{rollout}"], stderr=subprocess.DEVNULL)
                hw_performance= parse_neurosim_res(hw_res)
                acc_res = subprocess.check_output([f"python", f"noise_train.py", f"--rollout", f"{rollout}", f"--dev_var", f"{dev_var}", f"--train_epoch", f"{train_epoch}", f"--device", f"{args.device}"], stderr=subprocess.DEVNULL)
                accuracy = parse_ntrain_res(acc_res)
                result_dict[rollout] = (hw_performance, accuracy)
                performance = tradeoff_peformance(accuracy, hw_performance)
                hw_performance["accuracy"] = accuracy
                hw_performance["performance"] = performance
                print(f"rollout: {rollout}  ", hw_performance)
            except:
                print(f"{rollout}: Invalid design!")
                performance = -1

        arch_list.append(rollout)
        acc_list.append(performance)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input + experiments_prompt(arch_list, acc_list) + suffix},
        ]

if __name__ == "__main__":
    main()
