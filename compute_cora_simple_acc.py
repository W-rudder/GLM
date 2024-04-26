import json
import re
import random
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

with open('./results/cora_simple/graphsage_1000tp_5token_512_neg0_arxiv_linear_2_es_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open('./results/cora_simple/graphsage_1000tp_5token_512_neg0_arxiv_linear_2_es_model_results.txt', 'r') as f:
    eval_pred = json.load(f)

label_list = [
    "Case Based",
    "Genetic Algorithms",
    "Neural Networks",
    "Probabilistic Methods",
    "Reinforcement Learning",
    "Rule Learning",
    "Theory"
    ]
label2idx = {k: v for v, k in enumerate(label_list)}
patterns = [label for label in label_list]

cnt = 0
y, x = [], []
for label, pred in zip(eval_decode_label, eval_pred):
    # pred = pred.lower()
    # label = label.lower()
    
    matches = []
    for pattern in patterns:
        match_label = re.findall(pattern, pred)
        if len(match_label) >= 1:
            matches.append(match_label[0])
    if len(matches) >= 1:
        final_ans = matches[0]
    else:
        final_ans = pred
        
    if final_ans not in label2idx.keys():
        print("|"+final_ans+"|")
        cnt += 1
        true_label = label2idx[label]
        # 1.
        # random_number = random.randint(0, 69)
        # while random_number == true_label:
        #     random_number = random.randint(0, 69)
        # y.append(true_label)
        # x.append(random_number)
        # 2.
        y.append(true_label)
        x.append(10)
        # 3.
        # continue
    else:
        y.append(label2idx[label])
        x.append(label2idx[final_ans])

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
