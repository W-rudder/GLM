import torch
import tqdm
from config import *
import torch.nn.functional as F
from model import GraphEncoder, InstructGLM, Projector, LlamaEmbedding, GraphSageEncoder, OptGLM
from utils import InstructionDataset, TestDataset, ChemblDataset
from transformers import LlamaConfig, get_scheduler, AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer, OPTForCausalLM, LlamaTokenizer


first_model_path = './saved_model/first_model/{}_fm_{}_epoch{}_{}.pth'
cur_device = torch.device('cuda:0')
args = parse_args()
args.backbone = '/home/zuographgroup/zhr/model/vicuna-7b-v1.5'
args.zero_shot = True
args.best_epoch = 0
args.dataset = 'arxiv'
args.test_dataset = 'pubmed'
args.att_d_model = 2048
args.gnn_output = 4096
args.max_text_length = 700
args.dropout = 0.
# args.prefix = 'graphsage_5tp_5token_gnnarxiv_no_sim'
args.prefix = 'grace_5token_allarxiv'
args.batch_size = 1

tokenizer = LlamaTokenizer.from_pretrained(args.backbone)
tokenizer.pad_token=tokenizer.unk_token
special={'additional_special_tokens': ['<Node {}>'.format(i) for i in range(1, 110)]}
tokenizer.add_special_tokens(special)

test_dataset = InstructionDataset(tokenizer, args, mode="test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False,
                                        pin_memory=True, shuffle=False, collate_fn=test_dataset.collate_fn)

config_class = LlamaConfig
config = config_class.from_pretrained(args.backbone)
config.dropout_rate = args.dropout
config.dropout = args.dropout
config.attention_dropout = args.dropout
config.activation_dropout = args.dropout

model = InstructGLM.from_pretrained(
    args.backbone,
    config=config,
    torch_dtype=torch.bfloat16,
    # use_cache=True, 
    # low_cpu_mem_usage=True,
    device_map={"": cur_device}
)
llama_embeds = model.get_input_embeddings().weight.data
node_token=torch.zeros(110, llama_embeds.shape[1]).to(device=cur_device, dtype=llama_embeds.dtype)
llama_embeds=torch.cat([llama_embeds, node_token],dim=0)
first_model = GraphEncoder(args, llama_embed=llama_embeds).to(cur_device, dtype=torch.bfloat16)
first_model.load_state_dict(torch.load(first_model_path.format(args.prefix, args.dataset, args.best_epoch, 'end')))

first_model.eval()
model.eval()

max_sim = [0, 0, 0, 0, 0]
max_token = [0, 0, 0, 0, 0]
matrix = model.get_input_embeddings().weight.data

for batch in tqdm.tqdm(test_loader):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(cur_device)
        is_node = batch['is_node'].to(cur_device)
        attention_mask = batch['attn_mask'].to(cur_device)
        graph = batch['graph'].to(cur_device)

        embeds = first_model(
            input_ids=input_ids,
            is_node=is_node,
            graph=graph
        )
        vectors = embeds[is_node]
        assert vectors.shape[0] == 5

        for i in range(5):
            vector = vectors[i]
            cosine_similarity = torch.nn.functional.cosine_similarity(vector, matrix, dim=1)
            topk_values, topk_indices = torch.topk(cosine_similarity, 1)
            if topk_values.data.item() > max_sim[i]:
                max_sim[i] = topk_values.data.item()
                max_token[i] = tokenizer.convert_ids_to_tokens(topk_indices)

            # print("Top {} Similarities:".format(k))
            # print("Values:", topk_values)
            # print("Indices:", topk_indices)
            # print("Tokens:", tokenizer.convert_ids_to_tokens(topk_indices))
print(max_sim)
print(max_token)
        # results = model.g_step(in_embeds=embeds, attention_mask=attention_mask)



            
