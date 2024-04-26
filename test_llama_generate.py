from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from peft import PeftModel
import torch
import re
from utils import conv_templates, generate_stream, stream_output, load_pickle
import os


# model_name = '/home/zuographgroup/zhr/model/vicuna-7b-v1.5'
# model_name = '/home/zuographgroup/zhr/model/Llama-2-7b-chat-hf'
model_name = '/home/zuographgroup/zhr/model/galactica-6.7b'
lora_weights = '/home/zuographgroup/zhr/model/zjunlp'
model_path_basename = os.path.basename(os.path.normpath(model_name))
print(model_path_basename)
template = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:"
}

device = torch.device('cuda:0')

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
tokenizer.pad_token=tokenizer.unk_token
tokenizer.pad_token_id = 0
print("loading tokenizer finished")

model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, 
    load_in_8bit=True,
    use_cache=True,
    device_map={"":device}
    )
model_type = str(type(model)).lower()
# llama_embed = model.model.embed_tokens.weight.data

model = PeftModel.from_pretrained(
    model,
    lora_weights,
    device_map={"": device},
    use_cache=True,
    torch_dtype=torch.float16,
)
for n, p in model.named_parameters():
    print(n, p.dtype)
model.config.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2
print(f"loading model: {model_type}")
print(model.config.is_encoder_decoder)

input = 'The molecule is a natural product found in Picea abies, Citrus unshiu, and other organisms with data available.'
instruction = 'Create a molecule with the structure as the one described.'

instruction = "Provide a description of this molecule."
input = "[C][C@H1][C@@H1][Branch2][#Branch1][Ring2][C@H1][Branch2][=Branch1][#C][C@H1][Branch2][=Branch1][#Branch2][C@@H1][Branch1][Ring2][O][Ring1][=Branch1][O][C][C@@H1][C@H1][Branch2][Branch1][N][C@@H1][Branch2][Branch1][#Branch1][C@H1][Branch2][Branch1][C][C@@H1][Branch1][Ring2][O][Ring1][=Branch1][O][C][=C][Branch2][Ring1][Ring2][O][C][=C][C][=Branch1][=N][=C][C][=Branch1][Branch2][=C][Ring1][=Branch1][C][Ring1][#Branch2][=O][O][O][C][=C][C][=Branch1][=N][=C][Branch1][=Branch2][C][=Branch1][Ring2][=C][Ring1][=Branch1][O][C][O][O][O][O][O][O][O][O]"

input = 'c1csc(-c2nc3ccccc3s2)c1'
input = "COCCNC(=O)CCSc1nc(-c2ccc(OC)cc2)cc(C(F)(F)F)n1"
input = "COc1ccc(OC)c(C(C)=NNc2ccc(C(=O)O)cc2)c1.O=S(=O)(O)O"
input = "O=C(c1ccccn1)N1N=C2C(=Cc3ccccc3)CCCC2C1c1ccccc1"
instruction = 'The assay is PUBCHEM_BIOASSAY: qHTS Assay for Inhibitors of Human Jumonji Domain Containing 2E (JMJD2E). (Class of assay: confirmatory)  , and it is Direct single protein target assigned . The assay has properties: assay category is confirmatory ; assay organism is Homo sapiens ; assay type description is Functional. Is this molecule effective to this assay? Please choose an answer of "yes" or "no".'

input = 'O1CC[C@@H](NC(=O)[C@@H](Cc2cc3cc(ccc3nc2N)-c2ccccc2C)C)CC1(C)C'
instruction = 'BACE1 is an aspartic-acid protease important in the pathogenesis of Alzheimer\'s disease, and in the formation of myelin sheaths. It cleaves amyloid precursor protein (APP) to reveal the N-terminus of the beta-amyloid peptides. The beta-amyloid peptides are the major components of the amyloid plaques formed in the brain of patients with Alzheimer\'s disease (AD). Since BACE mediates one of the cleavages responsible for generation of AD, it is regarded as a potential target for pharmacological intervention in AD. BACE1 is a member of family of aspartic proteases. Same as other aspartic proteases, BACE1 is a bilobal enzyme, each lobe contributing a catalytic Asp residue, with an extended active site cleft localized between the two lobes of the molecule. Is this molecule effective to this assay? Please choose an answer of "yes" or "no".'

if input:
    res = template["prompt_input"].format(
        instruction=instruction, input=input
    )
else:
    res = template["prompt_no_input"].format(
        instruction=instruction
    )
print(res)

input_ids = tokenizer(
    [res],
    return_tensors="pt",
    padding="max_length",
    max_length=600,
).input_ids
attention_mask=input_ids.ne(tokenizer.pad_token_id)

llama_embed = model.model.model.embed_tokens.weight.data
inputs_embeds = llama_embed[input_ids]
print(inputs_embeds[-5:])

temperature=0.1
top_p=0.75
top_k=40
num_beams=4
repetition_penalty=1
max_new_tokens=128

generation_config = GenerationConfig(
    do_sample=False,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    num_beams=num_beams,
    repetition_penalty=repetition_penalty,
)
generation_config.bos_token_id = 1

with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids.to(device),
        # inputs_embeds=inputs_embeds.to(device),
        attention_mask=attention_mask,
        generation_config=generation_config,
        max_new_tokens=max_new_tokens,
    )
print(output_ids)
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
print(outputs)


# 16k有问题，用不了
# generate输入embeddings时输出就是答案，不要截断｜输入id时输出需要截断
# 训练时padding side=right｜预测时padding side=left



