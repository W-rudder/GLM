from platform import node
import re
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from ogb.graphproppred import PygGraphPropPredDataset
from transformers import LlamaTokenizerFast, LlamaForCausalLM, LlamaTokenizer
import argparse
from utils import *


def get_instructions(data_path):
    df = pd.read_json(data_path)
    df['edge_num'] = df.apply(get_edge_num, axis=1)
    df = df[df['edge_num'] != 0]
    df = df.reset_index(drop=True)
    # df = df.sample(n=50, ignore_index=True)

    return df


def get_graph_data(name, mode):
    dataset = PygGraphPropPredDataset(name = name,root = './data')
    filter_idx = []
    if mode.split('_')[-1] == 'val':
        split_idx = dataset.get_idx_split()['valid']
    else:
        split_idx = dataset.get_idx_split()[mode]
    for idx in split_idx:
        data = dataset[idx]
        if data.num_nodes <= 70:
            filter_idx.append(int(idx))
    if mode == "train":
        filter_idx = random.sample(filter_idx, len(filter_idx))
        filter_idx.sort()
    else:
        filter_idx = random.sample(filter_idx, len(filter_idx) // 40)
        filter_idx.sort()
    return dataset[filter_idx], filter_idx


class HIVDataset(Dataset):
    def __init__(self, tokenizer, args, mode='train') -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode

        self.graph_data, self.filter_idx = get_graph_data("ogbg-molhiv", self.mode)
        self.instructions = get_instructions("./instruction/HIV/HIV_{}_data.csv", self.mode, self.filter_idx)
        args.gnn_input = self.graph_data[0].x.shape[1]
        
        if mode.split('_')[-1] == 'val':
            self.len_transductive=len(self.graph_data)

    def __len__(self):
        assert len(self.instructions) == len(self.graph_data)
        return len(self.instructions)

    def __getitem__(self, idx):
        out_dict = {}
        out_dict['args'] = self.args

        instruction = self.instructions.iloc[idx]
        graph = self.graph_data[idx]

        source_text = instruction['prompt']
        target_text = instruction['output']
        
        input_ids = torch.tensor(self.tokenizer.encode(source_text))
        target_ids = torch.tensor(self.tokenizer.encode(target_text, padding=True, truncation=True, max_length=self.args.gen_max_length))
        
        is_node = (input_ids >= 32000)

        extra_num = is_node.sum()
        assert extra_num == graph.num_nodes

        out_dict['is_node'] = is_node
        out_dict['node_features'] = graph.x
        out_dict['edge_index'] = graph.edge_index
        out_dict['input_ids'] = input_ids
        out_dict['input_length'] = len(input_ids)
        out_dict['target_ids'] = target_ids
        out_dict['target_length'] = len(target_ids)
        out_dict['source_text'] = source_text
        out_dict['target_text'] = target_text

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}
        B = len(batch)
        if self.mode.split('_')[0] == 'train':
            if self.args.max_text_length:
                S_W_L = self.args.max_text_length
            else:
                S_W_L = max(entry['input_length']+entry['target_length']+1 for entry in batch)
        else:
            S_W_L = max(entry['input_length'] for entry in batch)  
        target_ids = torch.ones(B, S_W_L, dtype=torch.long) * (-100)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        is_node = torch.empty((B, S_W_L), dtype=torch.bool).fill_(False)

        source_text = []
        target_text = []
        node_features = []
        edge_index = []
        
        # Llama is decoder-only model, so we input source sentence + target sentence together during training.
        # And only input source sentence during validation/ inference.
        # Notably, Llama is left padding.
        for i, entry in enumerate(batch):
            if self.mode.split('_')[0] == 'train':     
                # The '[2]' indicates the EOS token in Llama.
                input_ids[i, -(entry['input_length']+entry['target_length']+1):] = torch.cat([entry['input_ids'], entry['target_ids'], torch.tensor([2])], dim=-1).long()
                is_node[i, -(entry['input_length']+entry['target_length']+1):-(entry['target_length']+1)] = entry['is_node']
            else:
                input_ids[i, -(entry['input_length']):] = torch.LongTensor(entry['input_ids'])
                is_node[i, -(entry['input_length']):] = entry['is_node']
            target_ids[i, -(entry['target_length']):] = torch.cat([entry['target_ids'][1:], torch.tensor([2])], dim=-1).long()

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
            if 'target_text' in entry:
                target_text.append(entry['target_text'])
            if 'node_features' in entry:
                node_features.append(entry['node_features'])
            if 'edge_index' in entry:
                edge_index.append(entry['edge_index'])

        attn_mask = input_ids.ne(self.tokenizer.pad_token_id).to(dtype=input_ids.dtype, device=input_ids.device)   # attention mask

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text    # For accuracy calculation.
        batch_entry['input_ids'] = input_ids # tensor
        batch_entry['target_ids'] = target_ids # tensor
        batch_entry['attn_mask']= attn_mask # tensor
        batch_entry['is_node'] = is_node # tensor
        batch_entry['node_features'] = node_features # list
        batch_entry['edge_index'] = edge_index # list

        return batch_entry      # Real batch data.


class InstructionDataset(Dataset):
    def __init__(self, tokenizer, args, mode='train') -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode

        if args.zero_shot:
            self.instructions = get_instructions(f"../TAGNN/instruction_new/{args.test_dataset}/{args.test_dataset}_dataset_{self.mode}.json")
            # self.instructions = get_instructions(f"./instruction/{args.test_dataset}/{args.test_dataset}_dataset_{self.mode}.json")
        else:
            self.instructions = get_instructions(f"../TAGNN/instruction_new/{args.dataset}/{args.dataset}_dataset_{self.mode}.json")
            # self.instructions = get_instructions(f"./instruction/{args.dataset}/{args.dataset}_dataset_{self.mode}.json")
            # self.instructions = get_instructions(f"./instruction/{args.dataset}/{args.dataset}_dataset_train.json")
        # args.gnn_input = len(self.instructions.loc[0, 'x'][0])
        args.gnn_input = len(self.instructions.loc[0, 'x'][0]) + len(self.instructions.loc[0, 'distance_list'][0])
        

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        out_dict = {}
        out_dict['args'] = self.args

        instruction = self.instructions.iloc[idx]

        source_text = instruction['prompt']
        source_text = "USER: What is the meaning of AI? ASSISTANT:"
        target_text = instruction['output']
        node_list = instruction['node_set']
        
        input_ids = torch.tensor(self.tokenizer.encode(source_text))
        target_ids = torch.tensor(self.tokenizer.encode(target_text, padding=True, truncation=True, max_length=self.args.gen_max_length))
        
        is_node = (input_ids >= 32000)
        edge_index = get_undirected_graph(instruction['edge_index'])

        extra_num = is_node.sum()
        # if self.args.no_graph:
        #     assert extra_num == 0
        # else:
        #     assert extra_num == len(node_list) + 1, f'extra_num: {extra_num}  node_list: {len(node_list)}'
        assert len(instruction['x']) == len(node_list)

        out_dict['is_node'] = is_node
        out_dict['node_features'] = torch.cat(
            [torch.tensor(instruction['x']), torch.tensor(instruction['distance_list'])],
            dim=-1
            )
        # out_dict['node_features'] = torch.tensor(instruction['x'])
        out_dict['edge_index'] = edge_index
        out_dict['input_ids'] = input_ids
        out_dict['input_length'] = len(input_ids)
        out_dict['target_ids'] = target_ids
        out_dict['target_length'] = len(target_ids)
        out_dict['source_text'] = source_text
        out_dict['target_text'] = target_text
        out_dict['mapping'] = [i for i in range(len(node_list))] + [0]

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}
        B = len(batch)
        if self.mode == 'train':
            if self.args.max_text_length:
                S_W_L = self.args.max_text_length
            else:
                S_W_L = max(entry['input_length']+entry['target_length']+1 for entry in batch)
        else:
            S_W_L = max(entry['input_length'] for entry in batch)
            S_W_L = 20
        target_ids = torch.ones(B, S_W_L, dtype=torch.long) * (-100)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        is_node = torch.empty((B, S_W_L), dtype=torch.bool).fill_(False)

        source_text = []
        target_text = []
        node_features = []
        edge_index = []
        mapping = []
        cum = 0
        
        # Llama is decoder-only model, so we input source sentence + target sentence together during training.
        # And only input source sentence during validation/ inference.
        # Notably, Llama is left padding.
        for i, entry in enumerate(batch):
            if self.mode == 'train':     
                # The '[2]' indicates the EOS token in Llama.
                input_ids[i, -(entry['input_length']+entry['target_length']+1):] = torch.cat([entry['input_ids'], entry['target_ids'], torch.tensor([2])], dim=-1).long()
                is_node[i, -(entry['input_length']+entry['target_length']+1):-(entry['target_length']+1)] = entry['is_node']
            else:
                input_ids[i, -(entry['input_length']):] = torch.LongTensor(entry['input_ids'])
                is_node[i, -(entry['input_length']):] = entry['is_node']
            target_ids[i, -(entry['target_length']):] = torch.cat([entry['target_ids'][1:], torch.tensor([2])], dim=-1).long()

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
            if 'target_text' in entry:
                target_text.append(entry['target_text'])
            if 'node_features' in entry:
                node_features.append(entry['node_features'])
            if 'edge_index' in entry:
                edge_index.append(entry['edge_index'])
            if 'mapping' in entry:
                mapping.extend([i + cum for i in entry['mapping']])
            cum += entry['node_features'].shape[0]

        attn_mask = input_ids.ne(self.tokenizer.pad_token_id).to(dtype=input_ids.dtype, device=input_ids.device)   # attention mask

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text    # For accuracy calculation.
        batch_entry['input_ids'] = input_ids # tensor
        batch_entry['target_ids'] = target_ids # tensor
        batch_entry['attn_mask']= attn_mask # tensor
        batch_entry['is_node'] = is_node # tensor
        batch_entry['node_features'] = node_features # list
        batch_entry['edge_index'] = edge_index # list
        batch_entry['mapping'] = mapping # list

        return batch_entry      # Real batch data.


if __name__ == '__main__':
    model=LlamaForCausalLM.from_pretrained('/home/zuographgroup/zhr/model/vicuna-7b-v1.5/Xorbits/vicuna-7b-v1.5')
    node_token=torch.zeros(100,4096) # Freezed Llama-v1-7b word embedding.
    llama_embeds=torch.cat([model.model.embed_tokens.weight.data, node_token],dim=0)

    save_pickle(llama_embeds,'vicuna_embeds.pkl') # Freezed Llama-v1-7b word embedding.
    # parser = argparse.ArgumentParser(description="GraphLLM")
    # args = parser.parse_args()
    # args.backbone = './7B'
    # args.distributed = False
    # args.losses = 'classification'
    # args.gen_max_length = 64
    # dl = get_HIV_loader(args)
    # m = 0
    # for entry in dl.dataset:
    #     if (entry['input_length']+entry['target_length']+1) > m :
    #         m = entry['input_length']+entry['target_length']+1
    # print(m)
    # for i, batch in enumerate(dl):
    #     print(i)
    #     if i == 12:
    #         print(batch['input_ids'].shape)
    #     break
    # df = get_instructions("./instruction/HIV/HIV_{}_data.csv", "train", filter_idx)

