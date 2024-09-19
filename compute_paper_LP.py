import json
import re
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# with open('./instruction/pubmed/labels.txt', 'r') as f:
#     eval_decode_label = json.load(f)

# with open('./instruction/pubmed/results.txt', 'r') as f:
#     eval_pred = json.load(f)
data = "cora_LP"
prefix = 'graphsage_1000tp_5token_512_neg0_arxiv_linear_1_3400_LP_max'
# prefix = 'soft_prompt_3_LP_soft_prompt_1'

# prefix = 'graphsage_0tp_5token_512_neg0_arxiv_linear_2_batch'
with open(f'./results/{data}/{prefix}_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open(f'./results/{data}/{prefix}_model_results.txt', 'r') as f:
    eval_pred = json.load(f)

label_list = [
    'These two papers must have citation relationships',
    'These two papers may not have citation relationships',
]
label2idx = {k.lower(): v for v, k in enumerate(label_list)}
patterns = [label.lower() for label in label_list]

cnt = 0
y, x = [], []
for label, pred in zip(eval_decode_label, eval_pred):
    pred = pred.lower()
    label = label.lower()[:-1]
    if label == 'yes':
        label = 'these two papers must have citation relationships'
    else:
        label = 'these two papers may not have citation relationships'
    
    if pred.startswith('these two papers may have citation relationships') or pred.startswith('these two papers have citation relationships') :
        pred = 'these two papers must have citation relationships'
    elif pred.startswith('these two papers have no citation relationship'):
        pred = 'these two papers may not have citation relationships'

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
        print(pred)
        cnt += 1
        continue
        y.append(label2idx[label])
        x.append(75)
    else:
        y.append(label2idx[label])
        x.append(label2idx[pred])

acc = accuracy_score(y, x)
auc = roc_auc_score(y, x)
res = classification_report(y, x)

print(cnt)
print(acc)
print(auc)
print(res)
