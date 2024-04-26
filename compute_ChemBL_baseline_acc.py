import json
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score

with open('./results/ChemBL/grace_5token_ans_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open('./results/ChemBL/grace_5token_ans_model_results.txt', 'r') as f:
    eval_pred = json.load(f)

label_list = [
    'yes',
    'no'
]
label2idx = {k.lower(): v for v, k in enumerate(label_list)}

cnt = 0
y, x = [], []
for label, pred in zip(eval_decode_label, eval_pred):
    pred = pred.lower()
    label = label.lower()
    if pred.endswith('.') or pred.endswith('?') or pred.endswith(','):
        pred = pred[:-1]

    pred = pred.strip()

    if pred.endswith('it is unlikely that this molecule would be effective in the assay') or pred.endswith('it is unlikely that the molecule would be effective in the assay'):
        pred = 'no'
    
    if '\n\n' in pred:
        pred = pred.split('\n\n')[1]
    
    # pred = pred.split('\n\n* ')[1][:5]
    # label = label[:5]
        
    if pred not in label2idx.keys():
        print("|"+pred+"|")
        cnt += 1
        # continue
        # y.append(label2idx[label])
        # x.append(75)
    else:
        y.append(label2idx[label])
        x.append(label2idx[pred])

acc = accuracy_score(y, x)
r = recall_score(y, x, average="macro")
p = precision_score(y, x, average="macro")
f1 = f1_score(y, x, average="macro")
auc = roc_auc_score(y, x)

print(f"Acc: {acc}")
print(f"F1: {f1}")
print(f"Precison: {p}")
print(f"Recall: {r}")
print(f"AUC: {auc}")
print(cnt / len(eval_decode_label))
