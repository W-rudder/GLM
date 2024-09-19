import json
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

# with open('./results/cora/graphsage_10tp_5token_ans_labels.txt', 'r') as f:
#     eval_decode_label = json.load(f)

# with open('./results/cora/graphsage_10tp_5token_ans_results.txt', 'r') as f:
#     eval_pred = json.load(f)

prefix = 'graphsage_1000tp_5token_512_neg0_arxiv_linear_1_3400_cora_baseline2'
with open(f'./results/cora/{prefix}_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open(f'./results/cora/{prefix}_model_results.txt', 'r') as f:
    eval_pred = json.load(f)

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

cnt = 0
y, x = [], []
s_x, s_y = [], []
for label, pred in zip(eval_decode_label, eval_pred):
    pred = pred.lower()
    label = label.lower()
    if label.split(', ')[1] == 'machine learning':
        label = ", ".join(label.split(', ')[1:])

    assert '\"' in pred, pred
    
    # if '\"' in pred:
    #     pred = pred.split('\"')
    #     if len(pred) >= 4:
    #         print(pred)
    #         pred = pred[3]
    #     else:
    #         pred = pred[1]

    # 长答案的，
    # computer architecture


    if pred.startswith('the most likely subcategory for the paper is'):
        pred = pred.split("\"")[1]
    elif pred.startswith('the most likely subcategory for the paper'):
        pred = pred.split(" is ")[1].replace('\"', '')
        if '. ' in pred:
            pred = pred.split('. ')[0]
    elif pred.startswith('the most likely subcategory for this paper is'):
        pred = pred.split("\"")[1]
    elif pred.startswith('the most likely subcategory for a paper on'):
        if len(pred.split("\"")) <= 3:
            pred = pred.split("\"")[1]
        else:
            pred = pred.split("\"")[3]
    else:
        if len(pred.split("\"")) <= 3:
            pred = pred.split("\"")[1]
        else:
            print(pred)
            pred = pred.split("\"")[3]
        
    
    # if pred.startswith(': '):
    #     pred = pred[2:]

    # if ':\n\n* ' in pred:
    #     pred = pred.split(':\n\n* ')[1]
    #     if '\n' in pred:
    #         pred = pred.split('\n')[0]

    # if '\"' in pred:
    #     pred = pred.split('\"')[1]
        
    if pred.startswith('. '):
        pred = pred[2:]
    elif pred.startswith('.'):
        pred = pred[1:]
    elif pred.endswith('.') or pred.endswith('?') or pred.endswith(','):
        pred = pred[:-1]
    
    pred = pred.strip()

    if label.startswith('data structures '):
        label = label.replace('data structures ', 'data structures,', 1)

    if pred.startswith('data structures '):
        pred = pred.replace('data structures ', 'data structures,', 1)
    if pred.startswith('data structures, algorithms,'):
        pred = pred.replace('data structures, algorithms,', 'data structures, algorithms', 1)
        
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

print(acc)
print(f1)
print(r)
print(p)
print(cnt / len(eval_decode_label))
