from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, GenerationConfig, LlamaConfig
from peft import PeftModel
from utils import conv_templates
import torch
import re
import os
import argparse
from model import GraphEncoder
import numpy as np
from circuitsvis.attention import attention_patterns
from matplotlib import pyplot as plt
import pandas as pd
from torch_geometric.data import Data, Batch
from model import InstructGLM


df = pd.read_json('./instruction/pubmed/case_study.json')
idx = 0
instruction = df.iloc[idx]

parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])
args.gnn_type = 'GraphSAGE'
args.gnn_output = 4096
args.num_token = 5
args.gnn_input = 128
args.gt_layers = 2
args.att_d_model = 2048
args.graph_pooling = 'sum'
args.edge_dim = None
args.dropout = 0.5
args.graph_unsup = False
args.prefix = 'graphsage_1000tp_5token_512_neg0_arxiv_linear_2_3400'
args.dataset = 'arxiv'
args.mask_token_list = None


model_name = '/home/zuographgroup/zhr/model/LLM-Research/Meta-Llama-3-8B-Instruct'
model_path_basename = os.path.basename(os.path.normpath(model_name))
print(model_path_basename)

device = torch.device('cuda:0')
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
tokenizer.add_special_tokens({"pad_token":"<pad>"})
special={'additional_special_tokens': ['<Node {}>'.format(i) for i in range(1, 21)]}
tokenizer.add_special_tokens(special)
print("loading tokenizer finished")


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32, 
    ).to(device)
model_type = str(type(model)).lower()
# model.config.pad_token_id = tokenizer.pad_token_id
# model.model.embed_tokens.weight.data = torch.cat([model.model.embed_tokens.weight.data, torch.zeros(1, 4096).to(device=device, dtype=torch.float16)],dim=0)
# model.model.embed_tokens.padding_idx = tokenizer.pad_token_id
print(model_type)
llama_embeds = model.model.embed_tokens.weight.data
node_token=torch.zeros(110, llama_embeds.shape[1]).to(device=device, dtype=llama_embeds.dtype)
llama_embeds=torch.cat([llama_embeds, node_token],dim=0)

# first model
first_model_path = './saved_model/first_model/{}_fm_{}_epoch{}_{}.pth'
first_model = GraphEncoder(args, llama_embed=llama_embeds).to(device, dtype=torch.bfloat16)
first_model.load_state_dict(torch.load(first_model_path.format(args.prefix, args.dataset, 0, 'end')))

conv = conv_templates["llama3"].copy()
roles = ["user", "assistant"]
# Apply prompt templates
conversations = []
# conv.append_message(roles[0], "What is the meaning of AI?")
# conv.append_message(roles[0], """Given a paper with the following information:
# Title: Cell-mediated immunity and biological response modifiers in insulin-dependent diabetes mellitus complicated by end-stage renal disease. 
#  Question: Which diabetes does this paper involve? Please directly give the most likely answer from the following diabetes: "Type 1 diabetes", "Type 2 diabetes", "Experimentally induced diabetes".""")
conv.append_message(roles[0], """Given a representation of a paper: <Node 1> <Node 2> <Node 3> <Node 4> <Node 5>, with the following information:
Title: Cell-mediated immunity and biological response modifiers in insulin-dependent diabetes mellitus complicated by end-stage renal disease. 
 Question: Which diabetes does this paper involve? Please directly give the most likely answer from the following diabetes: "Type 1 diabetes", "Type 2 diabetes", "Experimentally induced diabetes".""")

conv.append_message(roles[1], None)
conversations.append(conv.get_prompt())
print(conversations)

input_ids = tokenizer(
    conversations,
    return_tensors="pt",
    padding="max_length",
    max_length=200,
).input_ids
attention_mask=input_ids.ne(tokenizer.pad_token_id)


graph = Data()
graph.edge_index = torch.LongTensor(instruction['edge_index'])
graph.edge_attr = None
node_list = instruction['node_set']
graph.x = torch.tensor(instruction['x'], dtype=torch.bfloat16)
graph.lp = False

is_node = (input_ids >= 128257)
extra_num = is_node.sum()
print(extra_num)

graph = Batch.from_data_list([graph])

# first model
embeds = first_model(
    input_ids=input_ids.to(device),
    is_node=is_node.to(device),
    graph=graph.to(device),
    use_llm=False
)

# embeds = llama_embeds[input_ids]
print(embeds)
# print(new_embeds)
# print(attention_mask)
# print(embeds[is_node])


# generate
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
# embeds = torch.cat([embeds, embeds], dim=0)
# attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
print(embeds.shape)
print(attention_mask.shape)

with torch.no_grad():
    output_ids = model.generate(
        # input_ids=input_ids.to(device),
        inputs_embeds=embeds.to(device),
        attention_mask=attention_mask.to(device),
        max_new_tokens=500,
        eos_token_id=terminators,
        pad_token_id=128256,
    )
print(output_ids)
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
print(outputs)