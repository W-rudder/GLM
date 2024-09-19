import json
import pandas as pd
import numpy as np
import re
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score


df = pd.read_json('./instruction/pubmed/pubmed_dataset_test.json')

def get_ans(eval_decode_label, eval_pred):
    label_list = [
        'Experimentally induced diabetes',
        'Type 1 diabetes',
        'Type 2 diabetes'
    ]
    l_label_list = [label.lower() for label in label_list]
    label2idx = {k.lower(): v for v, k in enumerate(label_list)}
    patterns = [label.lower() for label in label_list]

    real_label = [label2idx[label.lower()] for label in df['output'].to_list()]

    y, x = [], []
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
            # print("|"+ori+"|")
            # continue
            y.append(label2idx[label])
            x.append(2)
        else:
            y.append(label2idx[label])
            x.append(label2idx[pred])
        llm_out.append(ori)
    assert real_label == y

    return y, x, llm_out

prefix = 'graphsage_1000tp_5token_512_neg0_arxiv_linear_1_3400'
with open(f'./results/pubmed/{prefix}_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open(f'./results/pubmed/{prefix}_model_results.txt', 'r') as f:
    eval_pred = json.load(f)
y, x, glm_out = get_ans(eval_decode_label, eval_pred)

prefix = 'graphsage_1000tp_5token_512_neg0_arxiv_linear_2_3400_mask2'
with open(f'./results/pubmed/{prefix}_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open(f'./results/pubmed/{prefix}_model_results.txt', 'r') as f:
    eval_pred = json.load(f)
y_n, x_n, glm_out_n = get_ans(eval_decode_label, eval_pred)

prefix = 'graphsage_0tp_5token_512_neg0_arxiv_linear_2_batch'
with open(f'./results/pubmed/{prefix}_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open(f'./results/pubmed/{prefix}_model_results.txt', 'r') as f:
    eval_pred = json.load(f)
y_b, x_b, llm_out = get_ans(eval_decode_label, eval_pred)

glm_right = np.array(x) == np.array(y)
glm_n_right = np.array(x_n) == np.array(y_n)
llm_right = np.array(x_b) == np.array(y_b)

grlr = np.intersect1d(glm_right.nonzero()[0], llm_right.nonzero()[0])
grlw = np.intersect1d(glm_right.nonzero()[0], (~llm_right).nonzero()[0])
gwlr = np.intersect1d((~glm_right).nonzero()[0], llm_right.nonzero()[0])
gwlw = np.intersect1d((~glm_right).nonzero()[0], (~llm_right).nonzero()[0])

grlr_n = np.intersect1d(glm_n_right.nonzero()[0], llm_right.nonzero()[0])
grlw_n = np.intersect1d(glm_n_right.nonzero()[0], (~llm_right).nonzero()[0])
gwlr_n = np.intersect1d((~glm_n_right).nonzero()[0], llm_right.nonzero()[0])
gwlw_n = np.intersect1d((~glm_n_right).nonzero()[0], (~llm_right).nonzero()[0])

df['mark'] = 0
df.loc[grlr, 'mark'] = 1
df.loc[grlw, 'mark'] = 2
df['glm_out'] = np.array(glm_out)
df['llm_out'] = np.array(llm_out)
print(df)
df.to_json("./instruction/pubmed/mark_ans_1.json", force_ascii=False)


