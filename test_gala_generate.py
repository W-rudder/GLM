from transformers import AutoTokenizer, OPTForCausalLM, GenerationConfig
from peft import PeftModel
import torch
import re
from utils import conv_templates, generate_stream, stream_output, load_pickle
import os


model_name = '/home/zuographgroup/zhr/model/galactica-6.7b'
model_path_basename = os.path.basename(os.path.normpath(model_name))
print(model_path_basename)

device = torch.device('cuda:0')

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
tokenizer.bos_token_id = 0
tokenizer.pad_token_id = 1
tokenizer.eos_token_id = 2
tokenizer.unk_token_id = 3
print("loading tokenizer finished")

# model = OPTForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     device_map={"":device}
#     )
# model_type = str(type(model)).lower()

# print(f"loading model: {model_type}")
# print(model)
# print(model.config.is_encoder_decoder)

# # input_text = "Title: The benefits of deadlifting\n\n"
# input_text = "The Transformer architecture [START_REF]"

smiles = 'c1csc(-c2nc3ccccc3s2)c1'
# # smiles = "COCCNC(=O)CCSc1nc(-c2ccc(OC)cc2)cc(C(F)(F)F)n1"
# # smiles = "COc1ccc(OC)c(C(C)=NNc2ccc(C(=O)O)cc2)c1.O=S(=O)(O)O"
# # smiles = "O=C(c1ccccn1)N1N=C2C(=Cc3ccccc3)CCCC2C1c1ccccc1"
input_text = f"""Here is a SMILES formula:\n\n[START_I_SMILES]{smiles}[END_I_SMILES]\n\nQuestion: BACE1 is an aspartic-acid protease important in the pathogenesis of Alzheimer\'s disease, and in the formation of myelin sheaths. It cleaves amyloid precursor protein (APP) to reveal the N-terminus of the beta-amyloid peptides. The beta-amyloid peptides are the major components of the amyloid plaques formed in the brain of patients with Alzheimer\'s disease (AD). Since BACE mediates one of the cleavages responsible for generation of AD, it is regarded as a potential target for pharmacological intervention in AD. BACE1 is a member of family of aspartic proteases. Same as other aspartic proteases, BACE1 is a bilobal enzyme, each lobe contributing a catalytic Asp residue, with an extended active site cleft localized between the two lobes of the molecule. Is this molecule effective to this assay?\n\nAnswer: Yes"""

# input_ids = tokenizer(
#     input_text,
#     return_tensors="pt",
#     padding="max_length",
#     max_length=600,
# ).input_ids.to(device)
# print(input_ids)
# attention_mask=input_ids.ne(tokenizer.pad_token_id).to(device, dtype=torch.long)

# llama_embed = model.model.decoder.embed_tokens.weight.data
# inputs_embeds = llama_embed[input_ids].to(device)

# with torch.no_grad():
#     outputs = model.generate(
#         input_ids=input_ids,
#         # inputs_embeds=inputs_embeds,
#         attention_mask=attention_mask,
#         max_new_tokens=1000,
#         do_sample=True,
#         temperature=0.7,
#         top_k=25,
#         top_p=0.9,
#         no_repeat_ngram_size=10,
#         early_stopping=True
#         )
# print(outputs)
# outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# print(outputs)

print(tokenizer.tokenize(' Yes'))
print(tokenizer(' Yes').input_ids)
print(tokenizer.vocab_size)
special={'additional_special_tokens': ['<Node {}>'.format(i) for i in range(1, 110)]}   # Add a new special token as place holder
tokenizer.add_special_tokens(special)
print(tokenizer.vocab_size)


# 16k有问题，用不了
# generate输入embeddings时输出就是答案，不要截断｜输入id时输出需要截断
# 训练时padding side=right｜预测时padding side=left



