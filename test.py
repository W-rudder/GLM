from transformers import default_data_collator, LlamaTokenizerFast, LlamaConfig, AutoTokenizer, LlamaTokenizer
import torch
import dgl
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from utils import *
from transformers.trainer_pt_utils import LabelSmoother


tokenizer = LlamaTokenizer.from_pretrained('/home/zuographgroup/zhr/model/vicuna-7b-v1.5')
tokenizer.pad_token=tokenizer.unk_token
df = pd.read_json("./instruction/cora_simple/cora_simple_dataset_test.json")
# df = pd.read_parquet("./instruction/BACE/BACE.parquet")
# dt = {}
# for label in df['output']:
#     if label not in dt.keys():
#         dt[label] = 1
#     else:
#         dt[label] += 1
# p = 0
# for k in dt.keys():
#     print(dt[k] / len(df))
#     p += dt[k] / len(df)
# print(p)
# print(dt.keys())
# print(len(dt.keys()))

# print(len(df))
# df = df.sample(n=10, ignore_index=True)
# print(df.columns)
# print(df.loc[0, 'prompt'])
# assert df.loc[0, 'prompt'].startswith('Given a representation of a paper: <Node 1>, with the following information:')
# df.loc[0, 'prompt'] = 'Given a paper with the following information:' + df.loc[0, 'prompt'][76:]
# print(df.loc[0, 'prompt'])
# print(df.loc[11, 'prompt'])
# # print(df.loc[6, 'output'])
# print(len(df.loc[11, 'x']))
# print(df.loc[11, 'edge_index'])
# print(len(df))
# cnt = 0
# for data in df['edge_index']:
#     if data != []:
#         cnt += 1
# print(cnt)
# def get_edge_num(x):
#     return len(x['edge_index'])
# df['edge_num'] = df.apply(get_edge_num, axis=1)
# df = df[df['edge_num'] != 0]
# print(len(df))
# edge_index = torch.tensor(df.loc[0, 'edge_index'])
# print(edge_index)
# print(df.loc[0, 'node_set'])
# row, col = edge_index[0], edge_index[1]
# g = dgl.graph((row, col))
# bg = dgl.to_bidirected(g)
# col, row = bg.edges()
# edge_index1 = df.loc[0, 'edge_index'][0]
# edge_index1.sort()
# print(edge_index1)
# print(torch.stack([row, col], dim=0))
# for i in range(10):
#     for f in df.loc[i, 'x']:
#         if len(f) != 128:
#             print(1)

pbar = tqdm(total=len(df['prompt']))
max_length = 0
sec_length = 0
ori_length = 0
row = 0
len_ls = []
ls = []
for i, (text, label) in enumerate(zip(df['prompt'], df['output'])):
    prompt = text.split('Abstract: ')[0] + 'Title: ' + text.split('Title: ')[1]
    prompt = text
    # if data == 'arxiv':
    #     prompt = text.split('Abstract: ')[0] + 'Title: ' + text.split('Title: ')[1]
    # elif data in ['pubmed', 'cora']:
    #     prompt = 'Given a representation of a paper: <Node 1>, with the following information:\nTitle: ' + text.split('Title: ')[1].split('And the other representation of a paper')[0] + 'And the other representation of a paper: <Node 1>, with the following information:\nTitle: ' + text.split('Title: ')[2]
    # prompt = 'Given a electronic product with the following information:' + text[len('Given a representation of a electronic product: <Node 1>, with the following information:'):]
    # prompt = 'Given a book with the following information:' + text[len('Given a representation of a book: <Node 1>, with the following information:'):]
    # prompt = 'Given a electronic product with the following information:' + text[len('Given a representation of a electronic product: <Node 1>, with the following information:'):]
    # prompt = f"""Here is a representation of a molecule:\n\n<Node 1>\n\nAnd here is the SMILES formula of the molecule:\n\n[START_I_SMILES]{smiles}[END_I_SMILES]\n\nQuestion: {text}. Is this molecule effective to this assay?\n\nAnswer:"""
    # question = text[4]
    # prompt = question.split('Is this molecule effective to the assay?')[0] + f"Based on the SMILES of the molecule:\"{graph}\". " + "Is this molecule effective to the assay? Please give an answer of \"yes\" or\"no\"."
    # print(prompt, label)
    length = len(tokenizer.encode(prompt)) + len(tokenizer.encode(label))
    len_ls.append(length)
    ls.append(len(prompt.split(' ')) + len(label.split(' ')))
    if length > max_length:
        sec_length = max_length
        max_length = length
        ori_length = len(prompt.split(' ')) + len(label.split(' '))
        row = i
    pbar.update()
print(max_length, row, ori_length)
print(sec_length)
print(df.loc[row])
print(df.loc[row]['prompt'])
print(np.percentile(len_ls, 95))
print(np.percentile(ls, 95))

