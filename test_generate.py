from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import torch
import re
from utils import conv_templates, generate_stream, stream_output, load_pickle
import os


SEQUENCE_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_position_embeddings",
    "max_seq_len",
    "model_max_length",
]

def get_context_length(config):
    """Get the context length of a model from a huggingface model config."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling:
        rope_scaling_factor = config.rope_scaling["factor"]
    else:
        rope_scaling_factor = 1

    for key in SEQUENCE_LENGTH_KEYS:
        val = getattr(config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048


model_name = '/home/zuographgroup/zhr/model/vicuna-7b-v1.5'
# model_name = '/home/zuographgroup/zhr/model/Llama-2-7b-chat-hf'
model_path_basename = os.path.basename(os.path.normpath(model_name))
print(model_path_basename)
prompt = {
    "id": "pubmed_test_10", 
    "graph": {"node_idx": 10, "edge_index": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 14, 0, 15, 16, 17, 18, 19, 20], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]], 
    "node_list": [10, 3878, 15581, 8554, 12388, 3515, 11983, 16894, 3185, 10852, 10624, 19410, 19510, 8307, 11041, 6869, 4384, 6023, 15915, 12233, 304]}, 
    "conversations": [
        {
            "from": "human", 
            "value": "Given a citation graph: \n<graph>\nwhere the 0th node is the target paper, and other nodes are its one-hop or multi-hop neighbors, with the following information: \nAbstract: We examined the effects of long-term treatment with an aldose reductase inhibitor (ARI) fidarestat on functional, morphological and metabolic changes in the peripheral nerve of 15-month diabetic rats induced by streptozotocin (STZ). Slowed F-wave, motor nerve and sensory nerve conduction velocities were corrected dose-dependently in fidarestat-treated diabetic rats. Morphometric analysis of myelinated fibers demonstrated that frequencies of abnormal fibers such as paranodal demyelination and axonal degeneration were reduced to the extent of normal levels by fidarestat-treatment. Axonal atrophy, distorted axon circularity and reduction of myelin sheath thickness were also inhibited. These effects were associated with normalization of increased levels of sorbitol and fructose and decreased level of myo-inositol in the peripheral nerve by fidarestat. Thus, the results demonstrated that long-term treatment with fidarestat substantially inhibited the functional and structural progression of diabetic neuropathy with inhibition of increased polyol pathway flux in diabetic rats. \n Title: Effects of 15-month aldose reductase inhibition with fidarestat on the experimental diabetic neuropathy in rats. \n Question: Which case of Type 1 diabetes, Type 2 diabetes, or Experimentally induced diabetes does this paper involve? Please give one answer of either Type 1 diabetes, Type 2 diabetes, or Experimentally induced diabetes directly. "
        }, 
        {
            "from": "gpt", 
            "value": "Experimentally induced diabetes"
        }
        ]
    }

cot_prompt = {
    "id": "pubmed_test_10", 
    "graph": {"node_idx": 10, "edge_index": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 14, 0, 15, 16, 17, 18, 19, 20], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]], "node_list": [10, 3878, 15581, 8554, 12388, 3515, 11983, 16894, 3185, 10852, 10624, 19410, 19510, 8307, 11041, 6869, 4384, 6023, 15915, 12233, 304]}, 
    "conversations": [
        {
            "from": "human", 
            "value": "Given a citation graph: \n<graph>\nwhere the 0th node is the target paper, with the following information: \nAbstract: We examined the effects of long-term treatment with an aldose reductase inhibitor (ARI) fidarestat on functional, morphological and metabolic changes in the peripheral nerve of 15-month diabetic rats induced by streptozotocin (STZ). Slowed F-wave, motor nerve and sensory nerve conduction velocities were corrected dose-dependently in fidarestat-treated diabetic rats. Morphometric analysis of myelinated fibers demonstrated that frequencies of abnormal fibers such as paranodal demyelination and axonal degeneration were reduced to the extent of normal levels by fidarestat-treatment. Axonal atrophy, distorted axon circularity and reduction of myelin sheath thickness were also inhibited. These effects were associated with normalization of increased levels of sorbitol and fructose and decreased level of myo-inositol in the peripheral nerve by fidarestat. Thus, the results demonstrated that long-term treatment with fidarestat substantially inhibited the functional and structural progression of diabetic neuropathy with inhibition of increased polyol pathway flux in diabetic rats. \n Title: Effects of 15-month aldose reductase inhibition with fidarestat on the experimental diabetic neuropathy in rats. \n Question: Does the paper involve any cases of Type 1 diabetes, Type 2 diabetes, or Experimentally induced diabetes? Please give one or more answers of either Type 1 diabetes, Type 2 diabetes, or Experimentally induced diabetes; if multiple options apply, provide a comma-separated list ordered from most to least related. Please think about the categorization in a step by step manner and avoid making false associations. Then provide your reasoning for each choice. "
        }, 
        {
            "from": "gpt", 
            "value": "The paper involves experimentally induced diabetes.\n\nReasoning: The paper states that the experimental diabetic rats were induced by streptozotocin (STZ). STZ is commonly used to induce experimental diabetes in animal models. Therefore, the diabetes in this study is experimentally induced."
        }
        ]
    }
device = torch.device('cuda:0')
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
print("loading tokenizer finished")
model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, 
    use_cache=True, 
    low_cpu_mem_usage=True,
    device_map={"":device}
    )
model_type = str(type(model)).lower()
print(f"loading model: {model_type}")
print(model.config.is_encoder_decoder)
context_len = get_context_length(model.config)
print(context_len)

qs = prompt["conversations"][0]["value"]
pattern = r'<graph>'
qs = re.sub(pattern, '', qs)

conv_mode = "vicuna_v1_1"

conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], prompt["conversations"][1]["value"])
prompt = conv.get_prompt()

# prompt ="You are WangduoModel, a large language and graph-structral assistant trained by HKUDS Lab. You are able to understand the graph structures that the user provides, and assist the user with a variety of tasks using natural language. Follow the instructions carefully and explain your answers in detail. USER: What is your name? ASSISTANT:"
# prompt = "USER: What is the meaning of AI? ASSISTANT:"
# print(prompt)
# smiles = "COCCNC(=O)CCSc1nc(-c2ccc(OC)cc2)cc(C(F)(F)F)n1"
# smiles = "COc1ccc(OC)c(C(C)=NNc2ccc(C(=O)O)cc2)c1.O=S(=O)(O)O"
smiles = "c1csc(-c2nc3ccccc3s2)c1"

# smiles = "O=C(c1ccccn1)N1N=C2C(=Cc3ccccc3)CCCC2C1c1ccccc1"
ins = """&The assay is PUBCHEM_BIOASSAY: qHTS Assay for Inhibitors of Human Jumonji Domain Containing 2E (JMJD2E). (Class of assay: confirmatory)  , and it is Direct single protein target assigned . The assay has properties: assay category is confirmatory ; assay organism is Homo sapiens ; assay type description is Functional."""
qs = f"""{ins} Based on the SMILES of the molecule:"{smiles}". Is this molecule effective to this assay? Please choose an answer of "yes" or "no"."""
prompt = f"""USER: {qs} ASSISTANT:"""
print(tokenizer.tokenize(prompt))
input_ids = tokenizer(
    prompt,
    return_tensors="pt",
    padding="max_length",
    max_length=1000,
    truncation=True,
).input_ids
print(len(input_ids))
attention_mask=input_ids.ne(tokenizer.pad_token_id)

llama_embed = load_pickle('./vicuna_embeds.pkl')
inputs_embeds = llama_embed[input_ids]

output_ids = model.generate(
    # input_ids.to(device),
    inputs_embeds=inputs_embeds.to(device),
    attention_mask=attention_mask,
    do_sample=False,
    temperature=0.9,
    max_new_tokens=1024,
    top_p=0.6,
)
# output_ids = model.generate(
#     # input_ids.to(device),
#     inputs_embeds=inputs_embeds.to(device),
#     attention_mask=attention_mask,
#     do_sample=True,
#     temperature=0.7,
#     max_new_tokens=1024,
# )
# output_ids = output_ids[0][len(input_ids[0]) :]
output_ids = output_ids[0]
outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
print(outputs)

output = model(inputs_embeds=inputs_embeds.to(device), attention_mask=attention_mask)
print("*"*20)
gen_params = {
    "model": model_name,
    "prompt": prompt,
    "temperature": 0.9,
    "repetition_penalty": 1.0,
    "max_new_tokens": 1024,
    "stop": None,
    "stop_token_ids": None,
    "top_p": 0.6,
    "echo": False,
}
output_stream = generate_stream(
    model,
    tokenizer,
    gen_params,
    device,
    context_len=context_len
)

outputs = stream_output(output_stream)


# 16k有问题，用不了
# generate输入embeddings时输出就是答案，不要截断｜输入id时输出需要截断
# 训练时padding side=right｜预测时padding side=left



