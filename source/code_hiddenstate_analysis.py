from utils.modelUtils import *
from utils.utils import *
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import kurtosis
from casper import nethook
import os
from datasets import load_dataset
from lllm.questions_loaders import WikiData, Sciq, MathematicalProblems

model_name = "/root/autodl-tmp/project/codemodel"
mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=(torch.float16 if "20b" in model_name else None),
    device = 'cuda:0,1'
)
mt.model

data_name = 'mbpp'
path = "/root/autodl-tmp/project/codemodel_results/{}_hiddenstates/".format(data_name)
if not os.path.exists(path):
    os.makedirs(path)
dataset = load_dataset('json', data_files='/root/autodl-tmp/project/data/mbpp/mbpp_prompts.jsonl')['train']
test_prompts = dataset['prompt']

# data_name = 'humaneval'
# path = "/root/autodl-tmp/project/codemodel_results/{}_hiddenstates/".format(data_name)
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = load_dataset('parquet', data_files='/root/autodl-tmp/project/data/humaneval/test-00000-of-00001.parquet')['train']
# test_prompts = dataset['prompt']

# data_name = 'pmc'
# path = "/root/autodl-tmp/project/codemodel_results/{}_5000_hiddenstates/".format(data_name)
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = load_dataset('json', data_files='/root/autodl-tmp/project/data/pmc/release.json')['train']
# test_prompts = []
# sample_num = 5000
# samples = random.sample(list(range(len(dataset))), sample_num)
# for i in samples:
#     test_prompt = dataset['instruction'][i] + dataset['input'][i]
#     test_prompts.append(test_prompt)
# np.save(path + "samples.npy", samples)

# data_name = 'menta'
# path = "/root/autodl-tmp/project/codemodel_results/{}_hiddenstates/".format(data_name)
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = load_dataset('json', data_files='/root/autodl-tmp/project/data/menta/Alexander_Street_shareGPT_2.0.json')['train']
# test_prompts = []
# sample_num = 5000
# samples = random.sample(list(range(len(dataset))), sample_num)
# for i in samples:
#     test_prompt = dataset['instruction'][i] + dataset['input'][i]
#     test_prompts.append(test_prompt)
# np.save(path + "samples.npy", samples)

# data_name = 'fin'
# path = "/root/autodl-tmp/project/codemodel_results/{}_hiddenstates/".format(data_name)
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = load_dataset('parquet', data_files='/root/autodl-tmp/project/data/fin/train-00000-of-00001-dabab110260ac909.parquet')['train']
# test_prompts = []
# sample_num = 5000
# samples = random.sample(list(range(len(dataset))), sample_num)
# for i in samples:
#     test_prompt = dataset['instruction'][i] + dataset['input'][i]
#     test_prompts.append(test_prompt)
# np.save(path + "samples.npy", samples)

# data_name = 'wikidata'
# path = "/root/autodl-tmp/project/codemodel_results/{}_hiddenstates/".format(data_name)
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = WikiData()
# test_prompts = dataset["question"]

# data_name = 'sciq'
# path = "/root/autodl-tmp/project/codemodel_results/{}_hiddenstates/".format(data_name)
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = Sciq()
# test_prompts = dataset["question"]

# data_name = 'mathematicalproblems'
# path = "/root/autodl-tmp/project/codemodel_results/{}_hiddenstates/".format(data_name)
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = MathematicalProblems()
# test_prompts = dataset["question"]

test_num = min(500, len(test_prompts))
hidden_states = []
for i in range(test_num):
    test_prompt = test_prompts[i]

    predict_token(
        mt,
        [test_prompt],
        return_p=True,
    )

    hidden_state = generate_hidden_states(test_prompt, mt,layer_range=range(-18, -23, -1))
    hidden_states.append(hidden_state.to('cpu').numpy())

np.save(path + "hidden_states.npy", hidden_states)