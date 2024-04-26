from platform import node
import re
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data, Batch
from ogb.graphproppred import PygGraphPropPredDataset
import transformers
from transformers import LlamaTokenizerFast, LlamaForCausalLM, LlamaTokenizer
import argparse
from .utils import *
from .conversation import conv_templates, SeparatorStyle
from .dataloader import NodeNegativeLoader

IGNORE_TOKEN_ID = -100
# conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]


def get_instructions(data_path):
    df = pd.read_json(data_path)
    df['edge_num'] = df.apply(get_edge_num, axis=1)
    df['gpt'] = df['output']
    # df = df[df['edge_num'] != 0]
    df = df.reset_index(drop=True)
    # df = df.sample(n=50, ignore_index=True)

    return df

def get_chembl_instructions(data_path, frac=1):
    df = pd.read_json(data_path)
    df['gpt'] = df['label']
    df = df.reset_index(drop=True)
    if frac != 1:
        df = df.sample(frac=frac, ignore_index=True)
    # df = df.sample(n=50, ignore_index=True)

    return df


def preprocess_opt(instruction, tokenizer, max_length, mode='train'):
    prompt = instruction['prompt']
    if mode == 'train':
        prompt = f"{prompt} {instruction['gpt']}</s>"

    tokenizer.padding_side = 'right' if mode == 'train' else 'left'

    # Apply prompt templates
    conversations = []
    conversations.append(prompt)

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    ).input_ids
    
    if mode != 'train':
        targets = tokenizer(
            [instruction["gpt"]+"</s>"],
            return_tensors="pt",
            padding="max_length",
            max_length=200,
            truncation=True,
        ).input_ids
    else:
        targets = input_ids.clone()

        question = prompt.split('Answer:')[0]
        answer = prompt.split('Answer:')[1]
        targets[:, :len(tokenizer(question + 'Answer:').input_ids)] = IGNORE_TOKEN_ID
        targets[:, len(tokenizer(question + 'Answer:').input_ids) + len(tokenizer(answer).input_ids):] = IGNORE_TOKEN_ID

    return dict(
        input_ids=input_ids,
        target_ids=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        text=conversations[0]
    )


def preprocess(
    instruction,
    tokenizer: transformers.PreTrainedTokenizer,
    max_length,
    mode='train'
) -> Dict:
    conv = conv_templates["vicuna_v1_1"].copy()
    assert conv.sep_style == SeparatorStyle.TWO

    roles = ["USER", "ASSISTANT"]
    tokenizer.padding_side = 'right' if mode == 'train' else 'left'

    # Apply prompt templates
    conversations = []
    conv.append_message(roles[0], instruction["prompt"])
    # conv.append_message(roles[0], "What is the meaning of AI?")
    if mode == 'train':
        conv.append_message(roles[1], instruction["gpt"])
    else:
        conv.append_message(roles[1], None)
    conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    ).input_ids
    
    if mode != 'train':
        targets = tokenizer(
            [instruction["gpt"]+"</s>"],
            return_tensors="pt",
            padding="max_length",
            max_length=200,
            truncation=True,
        ).input_ids
    else:
        targets = input_ids.clone()

        # Mask targets. Only compute loss on the assistant outputs.
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct. the first label is not _, but _label
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    instruction_len -= 1

                # Ignore the user instructions
                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                cur_len += turn_len

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    cur_len -= 1

            target[cur_len:] = IGNORE_TOKEN_ID


            if cur_len < max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    rank0_print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" #turn = {len(turns) - 1}. (ignored)"
                    )

    return dict(
        input_ids=input_ids,
        target_ids=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        text=conversations[0]
    )


class InstructionDataset(Dataset):
    def __init__(self, tokenizer, args, mode='train') -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode

        if args.zero_shot:
            self.instructions = get_instructions(f"./instruction/{args.test_dataset}/{args.test_dataset}_dataset_{self.mode}.json")
            # self.paper = torch.load(f"./data/graph_data_all.pt")
            # self.pubmed = self.paper['pubmed']
        else:
            if args.dataset == 'product':
                self.graph_data = torch.load(f"./data/graph_data_pd.pt")['product']
            else:
                self.graph_data = torch.load(f"./data/graph_data_ecommercial.pt")
                self.paper = torch.load(f"./data/graph_data_paper.pt")
                self.arxiv = self.paper['arxiv']
                self.pubmed = self.paper['pubmed']
                self.cora = self.paper['cora']
                self.cora_simple = self.paper['cora_simple']
                self.children = self.graph_data['book_children']
                self.history = self.graph_data['book_history']
                self.computer = self.graph_data['computer']
                self.photo = self.graph_data['photo']
                self.sports = self.graph_data['sports']
            self.instructions = get_instructions(f"./instruction/{args.dataset}/{args.dataset}_dataset_{self.mode}.json")
        # args.gnn_input = len(self.instructions.loc[0, 'x'][0])
        args.gnn_input = 128
        args.edge_dim = None
        
    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        # instruction = self.instructions.iloc[idx]
        raw = self.instructions.iloc[idx]
        instruction = raw.copy()
        tokens = " ".join([f"<Node {i}>" for i in range (1, 1 + self.args.num_token)])
        if not self.args.inference:
            if self.args.dataset == 'arxiv':  
                instruction['prompt'] = (raw['prompt'].split('Abstract: ')[0] + 'Title: ' + raw['prompt'].split('Title: ')[1]).replace("<Node 1>", tokens)
            elif self.args.dataset == 'paper_LP':
                if instruction['data'] == 'arxiv':
                    instruction['prompt'] = (raw['prompt'].split('Abstract: ')[0] + 'Title: ' + raw['prompt'].split('Title: ')[1]).replace("<Node 1>", tokens)
                elif instruction['data'] in ['pubmed', 'cora']:
                    instruction['prompt'] = ('Given a representation of a paper: <Node 1>, with the following information:\nTitle: ' + raw['prompt'].split('Title: ')[1].split('And the other representation of a paper')[0] + 'And the other representation of a paper: <Node 1>, with the following information:\nTitle: ' + raw['prompt'].split('Title: ')[2]).replace("<Node 1>", tokens)
            elif self.args.dataset == 'cross_domain':
                if instruction['data'] == 'arxiv':
                    instruction['prompt'] = (raw['prompt'].split('Abstract: ')[0] + 'Title: ' + raw['prompt'].split('Title: ')[1]).replace("<Node 1>", tokens)
                else:
                    instruction['prompt'] = raw['prompt'].replace("<Node 1>", tokens)
            else:
                instruction['prompt'] = raw['prompt'].replace("<Node 1>", tokens)
        else:
            if self.args.test_dataset in ['arxiv', 'pubmed', 'cora', 'cora_simple']:  
                instruction['prompt'] = (raw['prompt'].split('Abstract: ')[0] + 'Title: ' + raw['prompt'].split('Title: ')[1]).replace("<Node 1>", tokens)
                # instruction['prompt'] = 'Given a paper with the following information:\n' + 'Title: ' + raw['prompt'].split('Title: ')[1]
            # elif self.args.test_dataset == 'cora_simple':
            #     instruction['prompt'] = raw['prompt'].replace("<Node 1>", tokens)
                # instruction['prompt'] = 'Given a paper with the following information:\n' + raw['prompt'].split('with the following information: \n')[1]
            else:
                instruction['prompt'] = raw['prompt'].replace("<Node 1>", tokens)
        # instruction['prompt'] = (raw['prompt'].split('Description: ')[0] + 'Title: ' + raw['prompt'].split('Title: ')[1]).replace("<Node 1>", tokens)

        # assert instruction['prompt'].startswith('Given a representation of a paper: <Node 1> <Node 2> <Node 3> <Node 4> <Node 5>, with the following information:'), f"{raw['prompt']}"

        # ablation
        # ab_instruction = instruction.copy()
        # if self.args.test_dataset in ['computer', 'photo']:
        #     ab_instruction['prompt'] = 'Given a electronic product with the following information:' + instruction['prompt'][len('Given a representation of a electronic product: <Node 1>, with the following information:'):]
        # elif self.args.test_dataset == 'sports':
        #     ab_instruction['prompt'] = 'Given a fitness-related item with the following information:' + instruction['prompt'][len('Given a representation of a fitness-related item: <Node 1>, with the following information:'):]
        # elif self.args.test_dataset in ['book_children', 'book_history']:
        #     ab_instruction['prompt'] = 'Given a book with the following information:' + instruction['prompt'][len('Given a representation of a book: <Node 1>, with the following information:'):]
        # else:
        #     ab_instruction['prompt'] = 'Given a paper with the following information:' + instruction['prompt'][76:]

        out_dict = preprocess(instruction, self.tokenizer, self.args.max_text_length, self.mode)
        
        # graph data
        graph = Data()
        graph.edge_index = torch.LongTensor(instruction['edge_index'])
        if not self.args.zero_shot:
            if self.args.dataset == 'product':
                node_list = torch.LongTensor(instruction['node_set'])
                graph.x = self.graph_data.x[node_list].to(dtype=torch.bfloat16)
                graph.lp = False
            elif self.args.dataset == 'children_computer':
                node_list = torch.LongTensor(instruction['node_set'])
                if instruction['data'] == 'children':
                    graph.x = self.children.x[node_list].to(dtype=torch.bfloat16)
                else:
                    graph.x = self.computer.x[node_list].to(dtype=torch.bfloat16)
                graph.lp = False
            elif self.args.dataset.endswith('LP') or self.args.dataset == 'cross_domain':
                node_list = torch.LongTensor(instruction['node_set'])
                if instruction['data'] == 'arxiv':
                    graph.x = self.arxiv.x[node_list].to(dtype=torch.bfloat16)
                elif instruction['data'] == 'pubmed':
                    graph.x = self.pubmed.x[node_list].to(dtype=torch.bfloat16)
                if instruction['data'] == 'cora':
                    graph.x = self.cora.x[node_list].to(dtype=torch.bfloat16)
                elif instruction['data'] == 'children':
                    graph.x = self.children.x[node_list].to(dtype=torch.bfloat16)
                elif instruction['data'] == 'history':
                    graph.x = self.history.x[node_list].to(dtype=torch.bfloat16)
                elif instruction['data'] == 'computer':
                    graph.x = self.computer.x[node_list].to(dtype=torch.bfloat16)
                elif instruction['data'] == 'photo':
                    graph.x = self.photo.x[node_list].to(dtype=torch.bfloat16)
                elif instruction['data'] == 'sports':
                    graph.x = self.sports.x[node_list].to(dtype=torch.bfloat16)
                
                if instruction['task'] == 'lp':
                    graph.lp = True
                elif instruction['task'] == 'nc':
                    graph.lp = False
            else:
                node_list = instruction['node_set']
                graph.x = torch.tensor(instruction['x'], dtype=torch.bfloat16)
                graph.lp = False
                assert len(instruction['x']) == len(node_list)
        else:
            node_list = instruction['node_set']
            graph.x = torch.tensor(instruction['x'], dtype=torch.bfloat16)
            graph.lp = False
            assert len(instruction['x']) == len(node_list)
        graph.edge_attr = None
        # test
        # graph.y = self.pubmed.y[node_list][0].reshape(-1, 1).to(dtype=torch.bfloat16)
        
        is_node = (out_dict['input_ids'] >= 32000)
        extra_num = is_node.sum()
        if self.args.no_graph:
            assert extra_num == 0
        else:
            # ablation == 0
            assert extra_num in [self.args.num_token, self.args.num_token * 2], f'extra_num: {extra_num}'
            # assert extra_num == 0, f'extra_num: {extra_num}'

        out_dict['is_node'] = is_node
        out_dict['graph'] = graph

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}
        input_ids = []
        target_ids = []
        attention_mask = []
        is_node = []
        graph = []
        
        for i, entry in enumerate(batch):
            input_ids.append(entry['input_ids'])
            target_ids.append(entry['target_ids'])
            attention_mask.append(entry['attention_mask'])
            is_node.append(entry['is_node'])
            graph.append(entry['graph'])
        
        batch_entry['input_ids'] = torch.cat(input_ids, dim=0) # tensor
        batch_entry['target_ids'] = torch.cat(target_ids, dim=0) # tensor
        batch_entry['attn_mask']= torch.cat(attention_mask, dim=0) # tensor
        batch_entry['is_node'] = torch.cat(is_node, dim=0) # tensor
        batch_entry['graph'] = Batch.from_data_list(graph)

        return batch_entry      # Real batch data.


class ChemblDataset(Dataset):
    def __init__(self, tokenizer, args, mode='train') -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode

        if args.zero_shot:
            self.instructions = get_chembl_instructions(f"./instruction/{args.test_dataset}/{args.test_dataset}_dataset_{self.mode}.json")
            self.features = torch.load(f"./instruction/{args.test_dataset}/{args.test_dataset}_features_{self.mode}.pt")
        else:
            if mode == 'train':
                frac = 0.2
                self.instructions = get_chembl_instructions(f"./instruction/{args.dataset}/{args.dataset}_dataset_{self.mode}.json", frac)
                self.features = torch.load(f"./instruction/{args.dataset}/{args.dataset}_features_{self.mode}.pt")
            else:
                frac = 1
                self.instructions = get_chembl_instructions(f"./instruction/{args.dataset}/{args.dataset}_dataset_{self.mode}.json", frac)
                self.features = torch.load(f"./instruction/{args.dataset}/{args.dataset}_features_{self.mode}.pt")
        args.gnn_input = 128
        args.edge_dim = 128
        
    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        raw = self.instructions.iloc[idx]
        smiles = raw['graph']
        text_data = raw['text']
        if isinstance(text_data, list):
            text_data = text_data[0]

        if self.args.llm_type != 'opt':
            if self.args.zero_shot:
                # assert text_data.endswith('. Is this molecule effective to the assay?')
                # text = text_data.split('. Is this molecule effective to the assay?')[0]
                assert text_data.endswith('. Is this molecule effective to this assay?')
                text = text_data.split('. Is this molecule effective to this assay?')[0]
            else:
                assert text_data.endswith(' . Is the molecule effective to this assay?')
                text = text_data.split(' . Is the molecule effective to this assay?')[0]
        else:
            text = text_data

        instruction = raw.copy()

        if self.args.llm_type == 'opt':
            prompt = f"""Here is a representation of a molecule:\n\n<Node 1>\n\nAnd here is the SMILES formula of the molecule:\n\n[START_I_SMILES]{smiles}[END_I_SMILES]\n\nQuestion: {text}\n\nAnswer:"""
            tokens = " ".join([f"<Node {i}>" for i in range (1, 101)])
            instruction['prompt'] = prompt.replace("<Node 1>", tokens)
            # ablation
            # text = 'Will the chemical compound inhibit HIV replication?'
            # prompt = f"""Here is the SMILES formula of the molecule:\n\n[START_I_SMILES]{smiles}[END_I_SMILES]\n\nQuestion: {text}\n\nAnswer:"""
            # instruction['prompt'] = prompt
            out_dict = preprocess_opt(instruction, self.tokenizer, self.args.max_text_length, self.mode)
        else:
            prompt = f"""Given a representation of a molecule: <Node 1>, with the following information:\nAssay: {text}.\nMolecule: The simplified molecular input line entry specification of this molecule is "{smiles}".\nQuestion: Is this molecule effective to this assay? Please give an answer from "Yes" or "No". You don't need to give reasons."""
            tokens = " ".join([f"<Node {i}>" for i in range (1, 101)])
            instruction['prompt'] = prompt.replace("<Node 1>", tokens)
            # ablation
            # prompt = f"""Given a representation of a molecule with the following information:\nAssay: {text}.\nMolecule: The simplified molecular input line entry specification of this molecule is "{smiles}".\nQuestion: Is this molecule effective to this assay? Please give an answer from "Yes" or "No". You don't need to give reasons."""
            # instruction['prompt'] = prompt
            out_dict = preprocess(instruction, self.tokenizer, self.args.max_text_length, self.mode)
        
        # graph data
        if self.args.zero_shot:
            chemid = int(raw['molecule_index'])
        else:
            chemid = raw['chemid']

        graph = self.features[chemid]
        is_node = (out_dict['input_ids'] >= self.tokenizer.vocab_size)

        extra_num = is_node.sum()
        if self.args.no_graph:
            assert extra_num == 0
        else:
            assert extra_num == 100, f'extra_num: {extra_num}'
        assert graph.x.shape[0] == graph.num_nodes

        out_dict['is_node'] = is_node
        out_dict['graph'] = graph

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}
        input_ids = []
        target_ids = []
        attention_mask = []
        is_node = []
        graph = []
        
        for i, entry in enumerate(batch):
            input_ids.append(entry['input_ids'])
            target_ids.append(entry['target_ids'])
            attention_mask.append(entry['attention_mask'])
            is_node.append(entry['is_node'])
            graph.append(entry['graph'])
        
        batch_entry['input_ids'] = torch.cat(input_ids, dim=0) # tensor
        batch_entry['target_ids'] = torch.cat(target_ids, dim=0) # tensor
        batch_entry['attn_mask']= torch.cat(attention_mask, dim=0) # tensor
        batch_entry['is_node'] = torch.cat(is_node, dim=0) # tensor
        batch_entry['graph'] = Batch.from_data_list(graph)

        return batch_entry      # Real batch data.


class TestDataset(Dataset):
    def __init__(self, tokenizer, args, mode='train') -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode

        self.instructions = pd.read_parquet(f"./instruction/{args.test_dataset}/{args.test_dataset}.parquet")
        self.instructions = self.instructions[self.instructions['task_index'] == 'CHEMBL1613914']
        args.gnn_input = 128
        
    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        raw = self.instructions.iloc[idx]

        # ablation
        ab_instruction = raw.copy()
        smiles = ab_instruction['graph']
        question = ab_instruction['text'][0]
        assert question.endswith(' . Is the molecule effective to this assay?')
        ab_instruction['prompt'] = question.split(' . Is the molecule effective to this assay?')[0] + f". Based on the SMILES of the molecule: \"{smiles}\". " + """Is this molecule effective to this assay? Please choose an answer of "yes" or "no"."""
        ab_instruction['gpt'] = ab_instruction['label']

        out_dict = preprocess(ab_instruction, self.tokenizer, self.args.max_text_length, self.mode)
        
        # graph data
        node_list = [0]
        edge_index = torch.LongTensor([[0], [0]])
        is_node = (out_dict['input_ids'] >= 32000)

        extra_num = is_node.sum()
        if self.args.no_graph:
            assert extra_num == 0
        else:
            # ablation == 0
            assert extra_num == 0, f'extra_num: {extra_num}  node_list: {len(node_list)}'
        # assert len(instruction['x']) == len(node_list)

        out_dict['is_node'] = is_node
        # out_dict['node_features'] = torch.cat(
        #     [torch.tensor(instruction['x']), torch.tensor(instruction['distance_list'])],
        #     dim=-1
        #     )
        out_dict['node_features'] = torch.tensor([[1.0 for _ in range(128)]], dtype=torch.bfloat16)
        out_dict['edge_index'] = edge_index
        out_dict['mapping'] = [i for i in range(len(node_list))] + [0]

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}
        input_ids = []
        target_ids = []
        attention_mask = []
        is_node = []
        node_features = []
        edge_index = []
        mapping = []
        cum = 0
        
        for i, entry in enumerate(batch):
            input_ids.append(entry['input_ids'])
            target_ids.append(entry['target_ids'])
            attention_mask.append(entry['attention_mask'])

            is_node.append(entry['is_node'])
            node_features.append(entry['node_features'])
            edge_index.append(entry['edge_index'])
            mapping.extend([i + cum for i in entry['mapping']])
            cum += entry['node_features'].shape[0]

        
        batch_entry['input_ids'] = torch.cat(input_ids, dim=0) # tensor
        batch_entry['target_ids'] = torch.cat(target_ids, dim=0) # tensor
        batch_entry['attn_mask']= torch.cat(attention_mask, dim=0) # tensor
        batch_entry['is_node'] = torch.cat(is_node, dim=0) # tensor
        batch_entry['node_features'] = node_features # list
        batch_entry['edge_index'] = edge_index # list
        batch_entry['mapping'] = mapping # list

        return batch_entry      # Real batch data.


if __name__ == '__main__':
    model=LlamaForCausalLM.from_pretrained('/home/zuographgroup/zhr/model/vicuna-7b-v1.5')
    node_token=torch.zeros(100,4096) # Freezed Llama-v1-7b word embedding.
    llama_embeds=torch.cat([model.model.embed_tokens.weight.data, node_token],dim=0)

    save_pickle(llama_embeds,'vicuna_7B_embeds.pkl') # Freezed Llama-v1-7b word embedding.
    # parser = argparse.ArgumentParser(description="GraphLLM")
    # args = parser.parse_args()
    # args.backbone = '/home/zuographgroup/zhr/model/vicuna-13b-v1.5'
    # args.zero_shot = False
    # args.max_text_length = 1800
    # args.no_graph = False
    # args.dataset = 'arxiv'
    # tokenizer_class = LlamaTokenizer
    # tokenizer = tokenizer_class.from_pretrained(args.backbone)
    # tokenizer.pad_token=tokenizer.unk_token
    # special={'additional_special_tokens': ['<node {}>'.format(i) for i in range(1, 40)]}   # Add a new special token as place holder
    # tokenizer.add_special_tokens(special)
    # train_dataset = InstructionDataset(tokenizer, args, mode="train")
    # test_dataset = InstructionDataset(tokenizer, args, mode="test")
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, drop_last=True,
    #                                         pin_memory=True, shuffle=True, collate_fn=train_dataset.collate_fn)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, drop_last=False,
    #                                                 pin_memory=True, shuffle=False, collate_fn=test_dataset.collate_fn)
    # print(train_dataset[0]['input_ids'])
    # print(train_dataset[0]['target_ids'])
    # print(train_dataset[0]['attention_mask'])
    # print(test_dataset[0]['target_ids'][0, -50:])
    # print(test_dataset[0]['attention_mask'])
    # print(train_dataset[0]['text'])
    # print(test_dataset[0]['text'])
    # for batch in train_loader:
    #     print(batch['input_ids'].shape)
    #     print(batch['target_ids'].shape)
    #     print(batch['attn_mask'].shape)
    #     print(batch['is_node'].shape)
    #     break
    # for batch in test_loader:
    #     print(batch['input_ids'].shape)
    #     print(batch['target_ids'].shape)
    #     print(batch['attn_mask'].shape)
    #     print(batch['is_node'].shape)
    #     break
        

