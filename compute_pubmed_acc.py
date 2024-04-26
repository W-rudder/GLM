import json
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

with open('./results/pubmed/graphsage_1000tp_5token_512_neg0_arxiv_linear_3_es_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open('./results/pubmed/graphsage_1000tp_5token_512_neg0_arxiv_linear_3_es_model_results.txt', 'r') as f:
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

cnt = 0
y, x = [], []
for label, pred in zip(eval_decode_label, eval_pred):
    pred = pred.lower()
    label = label.lower()

    # if pred.startswith("the paper involves "):
    #     pred = pred[len("the paper involves "):].replace('\"', '')
    #     if ', ' in pred:
    #         pred = pred.split(', ')[0]
    #     if len(pred.split(' ')) > 3:
    #         pred = " ".join(pred.split(' ')[:3])
    # if pred.startswith("this paper involves "):
    #     pred = pred[len("this paper involves "):].replace('\"', '')
    #     if len(pred.split(' ')) > 3:
    #         pred = " ".join(pred.split(' ')[:3])
    # else:
    #     if 'involves' in pred:
    #         pred = pred.split('involves')[1]
    #         if pred.startswith(' is '):
    #             pred = pred.replace('\"', '')[4:]
    #             if len(pred.split(' ')) > 3:
    #                 pred = " ".join(pred.split(' ')[:3])
    #         elif pred.startswith(' are '):
    #             pred = pred.replace('\"', '')[5:]
    #             if len(pred.split(' ')) > 3:
    #                 pred = " ".join(pred.split(' ')[:3])
    #         elif pred.replace('\"', '')[1:].split(' ')[2] in ['diabetes', 'diabetes.', 'diabetes,']:
    #             pred = ' '.join(pred.replace('\"', '')[1:].split(' ')[:3])
    #         else:
    #             pass
    #     elif pred.split(' ')[2] not in ['diabetes', 'diabetes.', 'diabetes,']:
    #         if 'the most likely diabetes type for the paper is' in pred:
    #             pred = pred.split('\"')[1]
    #         elif '\"' in pred:
    #             if not check_ans(pred.split('\"')):
    #                 print(pred.split('\"'))
    #                 pass
    #             else:
    #                 pred = get_ans(pred.split('\"'))
    #             # print(pred.split('\"'))
    #             # if pred.split('\"')[1].split(' ')[2] == 'diabetes':
    #             #     pred = pred.split('\"')[1]
    #             # else:
    #             #     print(pred)
    #                 # pred = pred.split('\"')[3]
    #             # print(pred)
    #         else:
    #             if pred.split(' ')[-1] in ['diabetes', 'diabetes.', 'diabetes,']:
    #                 pred = ' '.join(pred.split(' ')[-3:])
    #             else:
    #                 # print(pred)
    #                 pass
    #     else:
    #         pred = ' '.join(pred.split(' ')[:3])
    #         # print(pred)

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

    if pred.startswith('type 2 diabetes'):
        pred = 'type 2 diabetes'
    elif pred.startswith('type 1 diabetes'):
        pred = 'type 1 diabetes'
    elif pred.startswith('experimentally induced diabetes'):
        pred = 'experimentally induced diabetes'
            

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
        print("|"+pred+"|")
        cnt += 1
        # continue
        y.append(label2idx[label])
        x.append(2)
    else:
        y.append(label2idx[label])
        x.append(label2idx[pred])

acc = accuracy_score(y, x)
r = recall_score(y, x, average="macro")
p = precision_score(y, x, average="macro")
f1 = f1_score(y, x, average="macro")
# r = recall_score(y, x, average="weighted")
# p = precision_score(y, x, average="weighted")
# f1 = f1_score(y, x, average="weighted")

print(f"Acc: {acc}")
print(f"F1: {f1}")
print(f"Precison: {p}")
print(f"Recall: {r}")
print(cnt / len(eval_decode_label))
