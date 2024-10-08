import json
import re
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

# with open('./instruction/pubmed/labels.txt', 'r') as f:
#     eval_decode_label = json.load(f)

# with open('./instruction/pubmed/results.txt', 'r') as f:
#     eval_pred = json.load(f)
prefix = 'graphsage_0tp_5token_512_neg0_arxiv_linear_1_3400_baseline'
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
patterns = [label.lower() for label in label_list]

cnt = 0
y, x = [], []
s_y, s_x = [], []
for label, pred in zip(eval_decode_label, eval_pred):
    pred = pred.lower()
    label = label.lower()
    # if pred.endswith('.') or pred.endswith('?') or pred.endswith(','):
    #     pred = pred[:-1]

    # if pred.startswith('the paper involves '):
    #     pred = pred[19:]
    # elif pred.startswith('this paper involves '):
    #     pred = pred[20:]

    # if pred.split(' ')[2] == 'diabetes':
    #     pred = " ".join(pred.split(' ')[:3])

    # pred = pred.strip()
    
    # pred = pred.split('\n\n* ')[1][:5]
    # label = label[:5]
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
        # continue
        s_y.append(label2idx[label])
        s_x.append(75)
    else:
        y.append(label2idx[label])
        x.append(label2idx[pred])
        s_y.append(label2idx[label])
        s_x.append(label2idx[pred])

# acc = accuracy_score(y, x)
acc = accuracy_score(s_y, s_x)
r = recall_score(y, x, average="macro")
p = precision_score(y, x, average="macro")
f1 = f1_score(y, x, average="macro")

print(acc)
print(f1)
print(r)
print(p)
print(cnt / len(eval_decode_label))
