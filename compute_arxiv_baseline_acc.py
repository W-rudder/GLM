import json
import re
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

prefix = 'graphsage_1000tp_5token_512_neg0_arxiv_linear_1_3400_arxiv_baseline2'
# prefix = 'graphsage_0tp_5token_512_neg0_arxiv_linear_1_3400_arxiv_baseline'
with open(f'./results/arxiv/{prefix}_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open(f'./results/arxiv/{prefix}_model_results.txt', 'r') as f:
    eval_pred = json.load(f)

# with open(f'./instruction/arxiv/labels.txt', 'r') as f:
#     eval_decode_label = json.load(f)

# with open(f'./instruction/arxiv/results.txt', 'r') as f:
#     eval_pred = json.load(f)


label_list = [
    'cs.AI, Artificial Intelligence', 
    'cs.CL, Computation and Language', 
    'cs.CC, Computational Complexity', 
    'cs.CE, Computational Engineering, Finance, and Science', 
    'cs.CG, Computational Geometry', 
    'cs.GT, Computer Science and Game Theory', 
    'cs.CV, Computer Vision and Pattern Recognition', 
    'cs.CY, Computers and Society',
    'cs.CR, Cryptography and Security', 
    'cs.DS, Data Structures and Algorithms', 
    'cs.DB, Databases', 
    'cs.DL, Digital Libraries', 
    'cs.DM, Discrete Mathematics', 
    'cs.DC, Distributed, Parallel, and Cluster Computing', 
    'cs.ET, Emerging Technologies', 
    'cs.FL, Formal Languages and Automata Theory', 
    'cs.GL, General Literature', 
    'cs.GR, Graphics', 
    'cs.AR, Hardware Architecture', 
    'cs.HC, Human-Computer Interaction', 
    'cs.IR, Information Retrieval', 
    'cs.IT, Information Theory', 
    'cs.LO, Logic in Computer Science', 
    'cs.LG, Machine Learning', 
    'cs.MS, Mathematical Software', 
    'cs.MA, Multiagent Systems', 
    'cs.MM, Multimedia', 
    'cs.NI, Networking and Internet Architecture', 
    'cs.NE, Neural and Evolutionary Computing', 
    'cs.NA, Numerical Analysis', 
    'cs.OS, Operating Systems', 
    'cs.OH, Other Computer Science',
    'cs.PF, Performance', 
    'cs.PL, Programming Languages', 
    'cs.RO, Robotics', 
    'cs.SI, Social and Information Networks', 
    'cs.SE, Software Engineering',
    'cs.SD, Sound',
    'cs.SC, Symbolic Computation', 
    'cs.SY, Systems and Control'
]
label2idx = {k[:5]: v for v, k in enumerate(label_list)}
patterns = [label[:5] for label in label_list]

cnt = 0
y, x = [], []
s_y, s_x = [], []
for label, pred in zip(eval_decode_label, eval_pred):
    label = label[:5]
    # if '\n\n' in pred:
    #     raw = pred
    #     pred = pred.split('\n\n')[1]
    #     if pred.startswith('*'):
    #         pred = pred[2:7]
    #     else:
    #         pred = pred[:5]    
    #         print(raw)
    # else:
    #     if '\"' in pred:
    #         pred = pred.split()

    #     pred = pred
    matches = []
    for pattern in patterns:
        match_label = re.findall(pattern, pred)
        if len(match_label) >= 1:
            matches.append(match_label[0])
    if len(matches) >= 1:
        pred = matches[0]
    else:
        pred = pred
    
    # pred = pred.strip()
    # label = label
        
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
