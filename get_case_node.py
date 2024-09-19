import json
import pandas as pd
import numpy as np
import re
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

df = pd.read_json('./instruction/pubmed/pubmed_dataset_test.json')

prefix = 'graphsage_1000tp_5token_512_neg0_arxiv_linear_2_3400'
with open(f'./results/pubmed/{prefix}_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open(f'./results/pubmed/{prefix}_model_results.txt', 'r') as f:
    eval_pred = json.load(f)

label_list = [
    'Experimentally induced diabetes',
    'Type 1 diabetes',
    'Type 2 diabetes'
]
l_label_list = [label.lower() for label in label_list]
label2idx = {k.lower(): v for v, k in enumerate(label_list)}
patterns = [label.lower() for label in label_list]

real_label = [label2idx[label.lower()] for label in df['output'].to_list()]

cnt = 0
y, x = [], []
glm_out = []
for label, pred in zip(eval_decode_label, eval_pred):
    ori = pred
    pred = pred.lower()
    label = label.lower()

    matches = []
    for pattern in patterns:
        match_label = re.findall(pattern, pred)
        if len(match_label) >= 1:
            matches.append(match_label[0])
    if len(matches) >= 1:
        pred = matches[0]
    else:
        pred = pred

    if pred not in label2idx.keys():
        # print("|"+ori+"|")
        cnt += 1
        # continue
        y.append(label2idx[label])
        x.append(2)
    else:
        y.append(label2idx[label])
        x.append(label2idx[pred])
    glm_out.append(ori)


prefix = 'graphsage_0tp_5token_512_neg0_arxiv_linear_2_batch'
with open(f'./results/pubmed/{prefix}_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open(f'./results/pubmed/{prefix}_model_results.txt', 'r') as f:
    eval_pred = json.load(f)

label_list = [
    'Experimentally induced diabetes',
    'Type 1 diabetes',
    'Type 2 diabetes'
]
label2idx = {k.lower(): v for v, k in enumerate(label_list)}

cnt = 0
y_b, x_b = [], []
llm_out = []
for label, pred in zip(eval_decode_label, eval_pred):
    ori = pred
    pred = pred.lower()
    label = label.lower()

    matches = []
    for pattern in patterns:
        match_label = re.findall(pattern, pred)
        if len(match_label) >= 1:
            matches.append(match_label[0])
    if len(matches) >= 1:
        pred = matches[0]
    else:
        pred = pred
        
    if pred not in label2idx.keys():
        # print(pred)
        cnt += 1
        # continue
        y_b.append(label2idx[label])
        x_b.append(75)
    else:
        y_b.append(label2idx[label])
        x_b.append(label2idx[pred])
    llm_out.append(ori)

assert real_label == y
assert real_label == y_b

glm_right = np.array(x) == np.array(y)
llm_right = np.array(x_b) == np.array(y_b)
glm_idx = glm_right.nonzero()
llm_idx = llm_right.nonzero()
diff_node = np.setdiff1d(glm_idx, llm_idx, assume_unique=False)
diff_node_ = np.setdiff1d(llm_idx, glm_idx, assume_unique=False)

print(len(diff_node), len(diff_node_))

new_df = df.loc[np.array(diff_node)]
new_df['glm_out'] = np.array(glm_out)[np.array(diff_node)]
new_df['llm_out'] = np.array(llm_out)[np.array(diff_node)]
new_df.to_json("./instruction/pubmed/case_study.json", force_ascii=False)


