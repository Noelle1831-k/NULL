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
from sentence_transformers import SentenceTransformer
from SPARQLWrapper import SPARQLWrapper, JSON
import json
from sklearn.metrics.pairwise import cosine_similarity
import requests
import pandas as pd
import ast

model_name = "/root/autodl-tmp/project/llama"
mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=(torch.float16 if "20b" in model_name else None),
    device = 'cuda:0,1'
)
mt.model

retriever = SentenceTransformer('/root/autodl-tmp/project/pubmed')

def get_spoke_api_resp(base_uri, end_point, params=None):
    uri = base_uri + end_point
    if params:
        return requests.get(uri, params=params)
    else:
        return requests.get(uri)

def query_spoke(disease_entity: str) -> list:
    base_uri = "https://spoke.rbvi.ucsf.edu"
    type_end_point = "/api/v1/types"
    result = get_spoke_api_resp(base_uri, type_end_point)
    data_spoke_types = result.json()
    node_types = list(data_spoke_types["nodes"].keys())
    edge_types = list(data_spoke_types["edges"].keys())
    node_types_to_remove = ["DatabaseTimestamp", "Version"]
    filtered_node_types = [node_type for node_type in node_types if node_type not in node_types_to_remove]

    api_params = {
        'node_filters': filtered_node_types,
        'edge_filters': edge_types,
        'cutoff_Compound_max_phase': 3,
        'cutoff_Protein_source': ['SwissProt'],
        'cutoff_DaG_diseases_sources': ['knowledge', 'experiments'],
        'cutoff_DaG_textmining': 3,
        'cutoff_CtD_phase': 3,
        'cutoff_PiP_confidence': 0.7,
        'cutoff_ACTeG_level': ['Low', 'Medium', 'High'],
        'cutoff_DpL_average_prevalence': 0.001,
        'depth': 1
    }
    
    node_type = "Disease"
    attribute = "name"
    nbr_end_point = f"/api/v1/neighborhood/{node_type}/{attribute}/{disease_entity}"
    url = f"{base_uri}{nbr_end_point}"
    
    try:
        response = requests.get(url, params=api_params)
        response.raise_for_status()
        node_context = response.json()
        
        nbr_nodes = []
        nbr_edges = []
        
        for item in node_context:
            if "_" not in item["data"]["neo4j_type"]:
                try:
                    if item["data"]["neo4j_type"] == "Protein":
                        name = item["data"]["properties"].get("description", "Unknown Protein")
                    else:
                        name = item["data"]["properties"].get("name", "Unknown Entity")
                except:
                    name = item["data"]["properties"].get("identifier", "Unknown")
                nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], name))
            
            else:
                try:
                    provenance = ", ".join(item["data"]["properties"].get("sources", []))
                except:
                    try:
                        provenance = item["data"]["properties"].get("source", "SPOKE-KG")
                        if isinstance(provenance, list):
                            provenance = ", ".join(provenance)
                    except:
                        try:
                            preprint_list = ast.literal_eval(item["data"]["properties"].get("preprint_list", "[]"))
                            if preprint_list:
                                provenance = ", ".join(preprint_list)
                            else:
                                pmid_list = ast.literal_eval(item["data"]["properties"].get("pmid_list", "[]"))
                                if pmid_list:
                                    provenance = ", ".join([f"pubmedId:{pmid}" for pmid in pmid_list])
                                else:
                                    provenance = "Institute For Systems Biology (ISB)"
                        except:
                            provenance = "SPOKE-KG"
                nbr_edges.append((item["data"]["source"], item["data"]["neo4j_type"], item["data"]["target"], provenance))
        
        nbr_nodes_df = pd.DataFrame(nbr_nodes, columns=["node_type", "node_id", "node_name"])
        nbr_edges_df = pd.DataFrame(nbr_edges, columns=["source", "edge_type", "target", "provenance"])

        merge_1 = pd.merge(nbr_edges_df, nbr_nodes_df, left_on="source", right_on="node_id").drop("node_id", axis=1)
        merge_1["source_name"] = merge_1["node_type"] + " " + merge_1["node_name"]
        merge_1 = merge_1.drop(columns=["source", "node_type", "node_name"])
        
        merge_2 = pd.merge(merge_1, nbr_nodes_df, left_on="target", right_on="node_id").drop("node_id", axis=1)
        merge_2["target_name"] = merge_2["node_type"] + " " + merge_2["node_name"]
        merge_2 = merge_2.drop(columns=["target", "node_type", "node_name"])

        merge_2["predicate"] = merge_2["edge_type"].apply(lambda x: x.split("_")[0])
        merge_2["triple"] = merge_2.apply(
            lambda row: f"{row['source_name']} → {row['predicate']} → {row['target_name']} (Source: {row['provenance']})", 
            axis=1
        )
        
        return merge_2["triple"].tolist()
    
    except Exception as e:
        print(f"SPOKE API failed: {str(e)}")
        return []

def extract_disease_entity(question: str) -> str:
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a biomedical entity recognition expert. Extract ONLY disease names from the user's Question.
Return your response in strict JSON format with a single key "diseases" containing a list of disease names.
Example: {{"diseases": ["XX disease"]}}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Question: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```json
{{"diseases": ["""
    inputs = mt.tokenizer(prompt, return_tensors="pt", truncation=True, add_special_tokens=False).to(mt.model.device)
    outputs = mt.model.generate(**inputs, max_new_tokens=100, temperature=0.01, do_sample=False, pad_token_id=mt.tokenizer.eos_token_id, eos_token_id=mt.tokenizer.eos_token_id, return_dict_in_generate=True)
    full_response = mt.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return parse_llama3_response(full_response)

def parse_llama3_response(full_response: str) -> str:
    if '```json' in full_response:
        json_blocks = re.findall(r'```json\n(.*?)\n```', full_response, re.DOTALL)
        
        if json_blocks:
            json_str = json_blocks[-1].strip()
            
            if not json_str.startswith('{'):
                json_str = '{' + json_str
            if not json_str.endswith('}'):
                json_str += '}'
            
            try:
                entity_data = json.loads(json_str)
                if "diseases" in entity_data and entity_data["diseases"]:
                    return entity_data["diseases"][0]
            except json.JSONDecodeError:
                pass
    
    if 'assistant' in full_response:
        assistant_part = full_response.split('assistant')[-1]
        
        json_like = re.search(r'\{.*\}', assistant_part, re.DOTALL)
        if json_like:
            try:
                json_str = json_like.group(0)
                
                json_str = json_str.replace('\n', '').replace(' ', '')
                json_str = re.sub(r',\s*]', ']', json_str)
                json_str = re.sub(r',\s*\}', '}', json_str)
                
                entity_data = json.loads(json_str)
                if "diseases" in entity_data and entity_data["diseases"]:
                    return entity_data["diseases"][0]
            except json.JSONDecodeError:
                pass
    

    disease_patterns = [
        r'["\']?diseases["\']?\s*:\s*\[([^\]]+)\]',
        r'["\']?diseases["\']?\s*:\s*"([^"]+)"',
    ]
    
    for pattern in disease_patterns:
        match = re.search(pattern, full_response)
        if match:
            disease_str = match.group(1)
            if '"' in disease_str or "'" in disease_str:
                diseases = re.findall(r'["\']([^"\']+)["\']', disease_str)
                if diseases:
                    return diseases[0]
            else:
                return disease_str.split(',')[0].strip()
    
    return ""

def link_to_spoke(question: str) -> list:
    disease = extract_disease_entity(question)
    if not disease or disease == "XX disease":
        print(f"Can't find the entity: {question}")
        return []
    triples = query_spoke(disease)
    return triples

def build_prompt(question: str, top_k: int = 5) -> str:
    triples = link_to_spoke(question)

    context = ""
    if triples:
        q_embed = retriever.encode([question])
        triple_embeds = retriever.encode(triples)
        sim_scores = cosine_similarity(q_embed, triple_embeds)[0]
        top_indices = np.argsort(sim_scores)[-top_k:][::-1]
        context = "\n".join([f"- {triples[i]}" for i in top_indices])

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a biomedical expert. Answer the user's question based solely on the provided knowledge from the SPOKE knowledge graph.
Do not make up information or speculate beyond what is provided.
Knowledge Context:
{context}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Question: {question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# data_name = 'mbpp'
# path = "/root/autodl-tmp/project/kgragmodel_results/{}_hiddenstates/".format(data_name)
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = load_dataset('json', data_files='/root/autodl-tmp/project/data/mbpp/mbpp_prompts.jsonl')['train']
# test_prompts = dataset['prompt']

# data_name = 'humaneval'
# path = "/root/autodl-tmp/project/kgragmodel_results/{}_hiddenstates/".format(data_name)
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = load_dataset('parquet', data_files='/root/autodl-tmp/project/data/humaneval/test-00000-of-00001.parquet')['train']
# test_prompts = dataset['prompt']

data_name = 'pmc'
path = "/root/autodl-tmp/project/kgragmodel_results/{}_5000_hiddenstates/".format(data_name)
if not os.path.exists(path):
    os.makedirs(path)
dataset = load_dataset('json', data_files='/root/autodl-tmp/project/data/pmc/release.json')['train']
test_prompts = []
sample_num = 5000
samples = random.sample(list(range(len(dataset))), sample_num)
for i in samples:
    test_prompt = dataset['instruction'][i] + dataset['input'][i]
    test_prompts.append(test_prompt)
np.save(path + "samples.npy", samples)

# data_name = 'menta'
# path = "/root/autodl-tmp/project/kgragmodel_results/{}_hiddenstates/".format(data_name)
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
# path = "/root/autodl-tmp/project/kgragmodel_results/{}_hiddenstates/".format(data_name)
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
# path = "/root/autodl-tmp/project/kgragmodel_results/{}_hiddenstates/".format(data_name)
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = WikiData()
# test_prompts = dataset["question"]

# data_name = 'sciq'
# path = "/root/autodl-tmp/project/kgragmodel_results/{}_hiddenstates/".format(data_name)
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = Sciq()
# test_prompts = dataset["question"]

# data_name = 'mathematicalproblems'
# path = "/root/autodl-tmp/project/kgragmodel_results/{}_hiddenstates/".format(data_name)
# if not os.path.exists(path):
#     os.makedirs(path)
# dataset = MathematicalProblems()
# test_prompts = dataset["question"]

test_num = min(500, len(test_prompts))
hidden_states = []
for i in range(test_num):
    test_prompt = build_prompt(test_prompts[i])

    predict_token(
        mt,
        [test_prompt],
        return_p=True,
    )

    hidden_state = generate_hidden_states(test_prompt, mt,layer_range=range(-18, -23, -1))
    hidden_states.append(hidden_state.to('cpu').numpy())

np.save(path + "hidden_states.npy", hidden_states)