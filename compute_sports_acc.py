import json
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, classification_report

prefix = 'graphsage_0tp_5token_512_neg0_computer_2_3400_more_epoch'
# prefix = 'soft_promt_5'

with open(f'./results/sports/{prefix}_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open(f'./results/sports/{prefix}_model_results.txt', 'r') as f:
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
s_x, s_y = [], []
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
    
    # if 'Accessories' in pred:
    #     pred = 'Accessories'
        
    
    pred = pred.strip()
    label = label

    if pred.endswith('.') or pred.endswith('?') or pred.endswith(','):
        pred = pred[:-1]
        
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
print(p)
print(r)
print(cnt / len(eval_decode_label))
print(classification_report(y, x))
