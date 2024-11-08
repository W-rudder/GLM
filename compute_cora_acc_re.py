import json
import re
import random
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, classification_report

prefix = 'graphsage_ae_5token_512_neg0_arxiv_linear_1_epoch_100'
# prefix = 'soft_prompt_5'

with open(f'./results/cora/{prefix}_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open(f'./results/cora/{prefix}_model_results.txt', 'r') as f:
    eval_pred = json.load(f)

# with open('./results/cora/graphsage_10tp_5token_ans_labels.txt', 'r') as f:
#     eval_decode_label = json.load(f)

# with open('./results/cora/graphsage_10tp_5token_ans_results.txt', 'r') as f:
#     eval_pred = json.load(f)

label_list = [
    'artificial intelligence, agents', 
    'artificial intelligence, data mining', 
    'artificial intelligence, expert systems', 
    'artificial intelligence, games and search', 
    'artificial intelligence, knowledge representation', 
    # 'artificial intelligence, machine learning, case-based', 
    # 'artificial intelligence, machine learning, genetic algorithms', 
    # 'artificial intelligence, machine learning, neural networks', 
    # 'artificial intelligence, machine learning, probabilistic methods', 
    # 'artificial intelligence, machine learning, reinforcement learning', 
    # 'artificial intelligence, machine learning, rule learning', 
    # 'artificial intelligence, machine learning, theory', 
    'machine learning, case-based', 
    'machine learning, genetic algorithms', 
    'machine learning, neural networks', 
    'machine learning, probabilistic methods', 
    'machine learning, reinforcement learning', 
    'machine learning, rule learning',
    'machine learning, theory',
    'artificial intelligence, nlp',
    'artificial intelligence, planning',
    'artificial intelligence, robotics', 
    'artificial intelligence, speech', 
    'artificial intelligence, theorem proving', 
    'artificial intelligence, vision and pattern recognition', 
    'data structures, algorithms and theory, computational complexity', 
    'data structures, algorithms and theory, computational geometry', 
    'data structures, algorithms and theory, formal languages', 
    'data structures, algorithms and theory, hashing', 
    'data structures, algorithms and theory, logic', 
    'data structures, algorithms and theory, parallel', 
    'data structures, algorithms and theory, quantum computing', 
    'data structures, algorithms and theory, randomized', 
    'data structures, algorithms and theory, sorting', 
    'databases, concurrency', 
    'databases, deductive', 
    'databases, object oriented', 
    'databases, performance', 
    'databases, query evaluation', 
    'databases, relational', 
    'databases, temporal', 
    'encryption and compression, compression', 
    'encryption and compression, encryption', 
    'encryption and compression, security', 
    'hardware and architecture, distributed architectures', 
    'hardware and architecture, high performance computing', 
    'hardware and architecture, input output and storage', 
    'hardware and architecture, logic design', 
    'hardware and architecture, memory structures', 
    'hardware and architecture, microprogramming', 
    'hardware and architecture, vlsi', 
    'human computer interaction, cooperative', 
    'human computer interaction, graphics and virtual reality', 
    'human computer interaction, interface design', 
    'human computer interaction, multimedia', 
    'human computer interaction, wearable computers', 
    'information retrieval, digital library', 
    'information retrieval, extraction', 
    'information retrieval, filtering', 
    'information retrieval, retrieval', 
    'networking, internet', 
    'networking, protocols', 
    'networking, routing', 
    'networking, wireless', 
    'operating systems, distributed', 
    'operating systems, fault tolerance', 
    'operating systems, memory management', 
    'operating systems, realtime', 
    'programming, compiler design', 
    'programming, debugging', 
    'programming, functional', 
    'programming, garbage collection', 
    'programming, java', 
    'programming, logic', 
    'programming, object oriented', 
    'programming, semantics', 
    'programming, software development']
label2idx = {k: v for v, k in enumerate(label_list)}
patterns = label_list

cnt = 0
y, x = [], []
s_x, s_y = [], []
for label, pred in zip(eval_decode_label, eval_pred):
    pred = pred.lower()
    label = label.lower()
    ori = pred
    if label.split(', ')[1] == 'machine learning':
        label = ", ".join(label.split(', ')[1:])
    elif label.startswith('data structures '):
        label = label.replace('data structures ', 'data structures,', 1)
    
    matches = []
    for pattern in patterns:
        if pattern == 'artificial intelligence, nlp':
            all_pattern = 'natural language processing' + '|nlp'
            match_label = re.findall(all_pattern, pred)
            if len(match_label) >= 1:
                matches.append('artificial intelligence, nlp')
        elif pattern.startswith('data structures'):
            mark = pattern.split(', ')[-1]
            all_pattern = 'data structures, algorithms and theory, ' + mark + '|data structures  algorithms and theory, ' + mark + '|data structures, algorithms, and theory, ' + mark
            match_label = re.findall(all_pattern, pred)
            if len(match_label) >= 1:
                matches.append('data structures, algorithms and theory, ' + mark)
        elif pattern.split(', ')[-1] in ['agents', 'data mining', 'expert systems', 'games and search', 'knowledge representation',
                'nlp', 'planning', 'robotics', 'speech', 'theorem proving', 'vision and pattern recognition']:
            mark = pattern.split(', ')[-1]
            all_pattern = 'artificial intelligence, ' + mark + '|machine learning, ' + mark
            match_label = re.findall(all_pattern, pred)
            if len(match_label) >= 1:
                matches.append('artificial intelligence, ' + mark)
        elif pattern.split(', ')[-1] in ['case-based', 'genetic algorithms', 'neural networks', 'probabilistic methods', 'reinforcement learning', 'rule learning', 'theory']:
            mark = pattern.split(', ')[-1]
            all_pattern = 'artificial intelligence, ' + mark + '|machine learning, ' + mark
            match_label = re.findall(all_pattern, pred)
            if len(match_label) >= 1:
                matches.append('machine learning, ' + mark)
        else:
            match_label = re.findall(pattern, pred)
            if len(match_label) >= 1:
                matches.append(match_label[0])
    if len(matches) >= 1:
        final_ans = matches[0]
    else:
        final_ans = pred
        
    if final_ans not in label2idx.keys():
        print(final_ans)
        cnt += 1
        # continue
        s_y.append(label2idx[label])
        s_x.append(75)
    else:
        y.append(label2idx[label])
        x.append(label2idx[final_ans])
        s_y.append(label2idx[label])
        s_x.append(label2idx[final_ans])

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