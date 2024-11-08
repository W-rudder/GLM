import json
import re
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, classification_report

prefix = 'graphsage_ae_5token_512_neg0_arxiv_linear_1_epoch_100'
# prefix = 'soft_prompt_5'

with open(f'./results/pubmed/{prefix}_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open(f'./results/pubmed/{prefix}_model_results.txt', 'r') as f:
    eval_pred = json.load(f)


def check_ans(ans):
    mark = False
    for a in label_list:
        if a.lower() in ans:
            mark = True
    return mark

def get_ans(ans):
    id1, id2, id3 = 10000, 10000, 10000
    if label_list[0].lower() in ans:
        id1 = ans.index(label_list[0].lower())
    if label_list[1].lower() in ans:
        id2 = ans.index(label_list[1].lower())
    if label_list[2].lower() in ans:
        id3 = ans.index(label_list[2].lower())
    
    m = min([id1, id2, id3])
    if m == id1:
        return label_list[0].lower()
    if m == id2:
        return label_list[1].lower()
    if m == id3:
        return label_list[2].lower()

label_list = [
    'Experimentally induced diabetes',
    'Type 1 diabetes',
    'Type 2 diabetes'
]
l_label_list = [label.lower() for label in label_list]
label2idx = {k.lower(): v for v, k in enumerate(label_list)}
patterns = [label.lower() for label in label_list]

cnt = 0
y, x = [], []
s_y, s_x = [], []
for label, pred in zip(eval_decode_label, eval_pred):
    pred = pred.lower()
    ori = pred
    label = label.lower()

    if '\"' in pred:
        ls = pred.split('\"')
        for i in ls:
            if i.endswith('.') or i.endswith('?') or i.endswith(','):
                ans = i[:-1]
            else:
                ans = i
            if ans in l_label_list:
                pred = i
                break
        else:
            pass
    else:
        pred = pred

    if pred.startswith('type 2'):
        pred = 'type 2 diabetes'
    elif pred.startswith('type 1'):
        pred = 'type 1 diabetes'
    elif pred.startswith('experimentally induced diabetes'):
        pred = 'experimentally induced diabetes'

    # matches = []
    # for pattern in patterns:
    #     match_label = re.findall(pattern, pred)
    #     if len(match_label) >= 1:
    #         matches.append(match_label[0])
    # if len(matches) >= 1:
    #     pred = matches[0]
    # else:
    #     pred = pred
            

    if pred.endswith('.') or pred.endswith('?') or pred.endswith(','):
        pred = pred[:-1]

    

    pred = pred.replace('\"', '')
    if ', as the' in pred:
        pred = pred.split(', as the')[0]
    elif ', as ' in pred:
        pred = pred.split(', as ')[0]

    if pred.startswith('this paper is related to '):
        pred = pred[len('this paper is related to '):]
    elif pred.startswith('this paper involves '):
        pred = pred[20:]
    elif pred.startswith('this paper is categorized as '):
        pred = pred[len('this paper is categorized as '):]
        if pred.split(' ')[2] == 'diabetes':
            pred = " ".join(pred.split(' ')[:3])

    if ',' in pred:
        pred = pred.split(',')[0]

    if pred.startswith('a case of '):
        pred = pred[10:]

    
    if pred.startswith("type 2 diabetes"):
        pred = "type 2 diabetes"

    if pred.split(' ')[-1] == 'diabetes':
        pred = " ".join(pred.split(' ')[-3:])

    pred = pred.strip()
        
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
# r = recall_score(y, x, average="weighted")
# p = precision_score(y, x, average="weighted")
# f1 = f1_score(y, x, average="weighted")

print(f"Acc: {acc}")
print(accuracy_score(y, x))
print(f"F1: {f1}")
print(f"Precison: {p}")
print(f"Recall: {r}")
print(cnt / len(eval_decode_label))
print(classification_report(y, x))
