import json
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

with open('./results/sports/graphsage_1000tp_5token_256_neg1_computer_1_1432_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open('./results/sports/graphsage_1000tp_5token_256_neg1_computer_1_1432_model_results.txt', 'r') as f:
    eval_pred = json.load(f)

label_list = [
    'Other Sports',
    'Exercise & Fitness',
    'Hunting & Fishing',
    'Accessories',
    'Leisure Sports & Game Room',
    'Team Sports',
    'Boating & Sailing',
    'Swimming',
    'Tennis & Racquet Sports',
    'Golf',
    'Airsoft & Paintball',
    'Clothing',
    'Sports Medicine'
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
        
    if pred not in label2idx.keys():
        # print(pred)
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
