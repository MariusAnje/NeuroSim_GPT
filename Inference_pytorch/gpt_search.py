import numpy as np
import os
import subprocess
import openai
import prompts
import nas_utils

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
    return accuracy + np.sqrt(hw_performance["fps"]/1600)

def main():
    dev_var = 0.04
    train_epoch = 100
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
                [[32, 5], [64, 5], [128, 5], [256, 5], [512, 7], [512, 7]]
            ]
        rollout = rollout_l[0]
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
                acc_res = subprocess.check_output([f"python", f"noise_train.py", f"--rollout", f"{rollout}", f"--dev_var", f"{dev_var}", f"--train_epoch", f"{train_epoch}"], stderr=subprocess.DEVNULL)
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
