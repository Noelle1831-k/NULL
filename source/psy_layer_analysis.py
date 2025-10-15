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

def trace_with_patch_layer(
    model,
    inp,
    states_to_patch,
    answers_t,
):
    prng = np.random.RandomState(1)
    layers = [states_to_patch[0], states_to_patch[1]]

    inter_results = {}

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    def patch_rep(x, layer):
        if layer not in layers:
            return x

        if layer == layers[0]:
            if isinstance(x, tuple):
                inter_results["hidden_states"] = x[0].detach().clone().cpu()
            else:
                inter_results["hidden_states"] = x.detach().clone().cpu()
            return x
        elif layer == layers[1]:
            device = x[0].device if isinstance(x, tuple) else x.device
            saved_hidden = inter_results["hidden_states"].to(device)

            if isinstance(x, tuple):
                new_output = (saved_hidden,) + x[1:]
                return new_output
            else:
                return saved_hidden
            
    with torch.no_grad(), nethook.TraceDict(
        model,
        layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs

def analyse_based_on_layer(prompt,):
    inp = make_inputs(mt.tokenizer,[prompt]*2)
    with torch.no_grad():
        answer_t, logits = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    model = mt.model
    result_prob = []
    for layer in range(mt.num_layers-1):
        layers = [layername(model, layer),layername(model, layer + 1)]
        prob =  trace_with_patch_layer(model, inp, layers,answer_t)
        result_prob.append(prob)
    data_on_cpu = [abs(x.item() - logits.item()) for x in result_prob]
        
    return logits.item() ,data_on_cpu

model_name = "/root/autodl-tmp/project/psymodel"
mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=(torch.float16 if "20b" in model_name else None),
    device = 'cuda:0,1'
)
mt.model

# data_name = 'mbpp'
# path = "/root/autodl-tmp/project/psymodel_results/{}_layer/".format(data_name)
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = load_dataset('json', data_files='/root/autodl-tmp/project/data/mbpp/mbpp_prompts.jsonl')['train']
# test_prompts = dataset['prompt']

# data_name = 'humaneval'
# path = "/root/autodl-tmp/project/psymodel_results/{}_layer/".format(data_name)
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = load_dataset('parquet', data_files='/root/autodl-tmp/project/data/humaneval/test-00000-of-00001.parquet')['train']
# test_prompts = dataset['prompt']

# data_name = 'pmc'
# path = "/root/autodl-tmp/project/psymodel_results/{}_5000_layer/".format(data_name)
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

data_name = 'menta'
path = "/root/autodl-tmp/project/psymodel_results/{}_layer/".format(data_name)
if not os.path.exists(path):
    os.makedirs(path)
dataset = load_dataset('json', data_files='/root/autodl-tmp/project/data/menta/Alexander_Street_shareGPT_2.0.json')['train']
test_prompts = []
sample_num = 5000
samples = random.sample(list(range(len(dataset))), sample_num)
for i in samples:
    test_prompt = dataset['instruction'][i] + dataset['input'][i]
    test_prompts.append(test_prompt)
np.save(path + "samples.npy", samples)

# data_name = 'fin'
# path = "/root/autodl-tmp/project/psymodel_results/{}_layer/".format(data_name)
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
# path = "/root/autodl-tmp/project/psymodel_results/{}_layer/".format(data_name)
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = WikiData()
# test_prompts = dataset["question"]

# data_name = 'sciq'
# path = "/root/autodl-tmp/project/psymodel_results/{}_layer/".format(data_name)
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = Sciq()
# test_prompts = dataset["question"]

# data_name = 'mathematicalproblems'
# path = "/root/autodl-tmp/project/psymodel_results/{}_layer/".format(data_name)
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = MathematicalProblems()
# test_prompts = dataset["question"]

test_num = min(500, len(test_prompts))
kurts = []
layerAIEs = []
for i in range(test_num):
    test_prompt = test_prompts[i]

    predict_token(
        mt,
        [test_prompt],
        return_p=True,
    )

    output = generate_outputs(test_prompt,mt,)
    logits, layerAIE = analyse_based_on_layer(test_prompt)

    seq = layerAIE
    layerAIEs.append(seq)
    logits = logits
    kurt = kurtosis(seq, fisher=False)
    print(kurt)
    kurts.append(kurt)

    sns.set_theme()
    plt.figure(figsize=(9,6))
    sns.scatterplot(x=range(1, len(seq)+1), y=seq, color='b')
    plt.title('Prompt: ' + test_prompts[i][:50]) #apps
    plt.figtext(0.3, 0.03, f'Logits: {logits:.4f}', ha='center', va='center')
    plt.figtext(0.7, 0.03, f'Kurtosis: {kurt:.4f}', ha='center', va='center')
    plt.savefig(path + "{}.png".format(i),bbox_inches="tight")
    plt.close()
np.save(path + "kurts.npy", kurts)
np.save(path + "layerAIE.npy", layerAIEs)