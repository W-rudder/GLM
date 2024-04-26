import json
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

with open('./results/photo/graphsage_5tp_5token_children_new_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open('./results/photo/graphsage_5tp_5token_children_new_model_results.txt', 'r') as f:
    eval_pred = json.load(f)

label_list = [
    'Film Photography',
    'Video',
    'Digital Cameras',
    'Accessories',
    'Binoculars & Scopes',
    'Lenses',
    'Bags & Cases',
    'Lighting & Studio',
    'Flashes',
    'Tripods & Monopods',
    'Underwater Photography',
    'Video Surveillance'
]
label2idx = {k: v for v, k in enumerate(label_list)}

cnt = 0
y, x = [], []
for label, pred in zip(eval_decode_label, eval_pred):
    if '\"' in pred:
        ls = pred.split('\"')
        for i in ls:
            if i.endswith('.') or i.endswith('?') or i.endswith(','):
                ans = i[:-1]
            else:
                ans = i
            if ans in label_list:
                pred = i
                break
        else:
            pass
    else:
        pred = pred
        
    
    pred = pred.strip()
    label = label

    if pred.endswith('.') or pred.endswith('?') or pred.endswith(','):
        pred = pred[:-1]

    if '\n\n' in pred:
        pred = pred.split('\n\n')[0]

    if pred.startswith('Category: '):
        pred = pred[len('Category: '):]
        
    if pred not in label2idx.keys():
        print("|"+pred+"|")
        cnt += 1
        # continue
        y.append(label2idx[label])
        x.append(75)
    else:
        y.append(label2idx[label])
        x.append(label2idx[pred])

acc = accuracy_score(y, x)
r = recall_score(y, x, average="weighted")
p = precision_score(y, x, average="weighted")
f1 = f1_score(y, x, average="weighted")

print(acc)
print(f1)
print(p)
print(r)
print(cnt / len(eval_decode_label))
