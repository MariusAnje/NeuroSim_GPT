import prompts
import pyperclip

system_content, user_input, experiments_prompt, suffix = prompts.system_content, prompts.user_input, prompts.experiments_prompt, prompts.suffix
arch_list = [[[32, 3], [64, 3], [128, 3], [256, 3], [512, 3], [512, 3]], 
             [[32, 3], [64, 3], [128, 5], [256, 3], [512, 3], [512, 5]],
             [[32, 3], [64, 3], [128, 5], [256, 5], [512, 3], [512, 7]],
             [[32, 5], [64, 5], [128, 5], [256, 5], [512, 7], [512, 7]]]
acc_list = [1.5915106271899613, 1.524998472077137, 1.5412971415508383]

messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input + experiments_prompt(arch_list, acc_list) + suffix},
        ]

output = messages[0]["content"] + "\n" + messages[1]["content"]
print(output)
pyperclip.copy(output)