import prompts
import pyperclip

system_content, user_input, experiments_prompt, suffix = prompts.system_content, prompts.user_input, prompts.experiments_prompt, prompts.suffix
arch_list = [[[32, 3], [64, 3], [128, 3], [256, 3], [512, 3], [512, 3]]]
acc_list = [1.321410627189961]

messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input + experiments_prompt(arch_list, acc_list) + suffix},
        ]

output = messages[0]["content"] + "\n" + messages[1]["content"]
print(output)
pyperclip.copy(output)