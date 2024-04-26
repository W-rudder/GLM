import json
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

with open('./results/book_history/graphsage_1000tp_5token_256_neg1_computer_1_1432_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open('./results/book_history/graphsage_1000tp_5token_256_neg1_computer_1_1432_model_results.txt', 'r') as f:
    eval_pred = json.load(f)

label_list = [
    'World',
    'Americas',
    'Asia',
    'Military',
    'Europe',
    'Russia',
    'Africa',
    'Ancient Civilizations',
    'Middle East',
    'Historical Study & Educational Resources',
    'Australia & Oceania',
    'Arctic & Antarctica'
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
            # print(pred)
            pass
        # if len(ls) <= 3:
        #     pred = pred.split('\"')[1]
        # elif 3< len(ls) <= 5:
        #     pred = pred.split('\"')[3]
        # else:
        #     pred = pred.split('\"')[3]
    else:
        if 'Category: ' in pred:
            pred = pred.split('Category: ')[1]
        elif ', ' in pred:
            pred = pred.split(', ')[0]
        else:
            pred = pred
    
    pred = pred.strip()
    label = label

    if pred.endswith('.') or pred.endswith('?') or pred.endswith(','):
        pred = pred[:-1]
    
    if pred.startswith('European'):
        pred = 'Europe'
        
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

# r = recall_score(y, x, average="macro")
# p = precision_score(y, x, average="macro")
# f1 = f1_score(y, x, average="macro")

print(acc)
print(f1)
print(p)
print(r)
print(cnt / len(eval_decode_label))
