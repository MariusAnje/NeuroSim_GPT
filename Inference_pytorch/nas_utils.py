import ast
from models.VGG import vgg8
# cfg_list = {
#     'cifar10': [('C', 128, 3, 'same', 2.0),
#                 ('C', 128, 3, 'same', 16.0),
#                 ('M', 2, 2),
#                 ('C', 256, 3, 'same', 16.0),
#                 ('C', 256, 3, 'same', 16.0),
#                 ('M', 2, 2),
#                 ('C', 512, 3, 'same', 16.0),
#                 ('C', 512, 3, 'same', 32.0),
#                 ('M', 2, 2)]
# }

_channel_list = [32, 64, 128, 256, 512]
_kernel_list =  [3,5,7]

def gpt_to_rollout(rollout):
    new_rollout = []
    for c_s, k_s in rollout:
        i = _channel_list.index(c_s)
        j = _kernel_list.index(k_s)
        new_rollout.append([i,j])
    return new_rollout

def generate_cfg(rollout):
    idk = [2., 16., 16., 16., 16., 32.]
    cfg = [()] * 9
    for i in range(2,9,3):
        cfg[i] = ('M', 2, 2)
    j = -1
    for i in range(6):
        if i %2 == 0:
            j += 1
        k = i + j
        c_out = _channel_list[rollout[i][0]]
        k_s   = _kernel_list[rollout[i][1]]
        cfg[k] = ('C', c_out, k_s, 'same', idk[i])
    return cfg

def generate_csv_content(rollout):
    FM = [32, 32, 16, 16, 8, 8]
    csv = ""
    c_in = 3
    for i in range(6):
        c_out = _channel_list[rollout[i][0]]
        k_s   = _kernel_list[rollout[i][1]]
        csv += f"{FM[i]},{FM[i]},{c_in},{k_s},{k_s},{c_out},{i%2},{1}\n"
        c_in = c_out
    csv += f"1,1,{c_in*4*4},1,1,1024,0,1\n"
    csv += f"1,1,1024,1,1,10,0,1\n"
    return csv

def write_csv(rollout, file_path='./NeuroSIM/NetWork_'+'VGG8'+'_nas.csv'):
    if isinstance(rollout, str):
        rollout = rollout_str_to_list(rollout)
    csv_contenct = generate_csv_content(rollout)
    f = open(file_path, "w+")
    f.write(csv_contenct)
    f.close()

def rollout_str_to_list(rollout_str):
    return ast.literal_eval(rollout_str)

def get_model_from_rollout(rollout, args, logger):
    rollout = rollout_str_to_list(rollout)
    cfg = generate_cfg(rollout)
    model = vgg8(cfg, args, logger)
    return model


if __name__ == "__main__":
    import numpy as np
    import ast
    a = [[2,0],[2,0],[3,0],[3,0],[4,0],[4,0]]
    print(write_csv(a))
