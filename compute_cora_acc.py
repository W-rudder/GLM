import json
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

with open('./results/cora/graphsage_0tp_5token_512_neg0_arxiv_linear_2_batch_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open('./results/cora/graphsage_0tp_5token_512_neg0_arxiv_linear_2_batch_model_results.txt', 'r') as f:
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
for label, pred in zip(eval_decode_label, eval_pred):
    pred = pred.lower()
    label = label.lower()
    ori = pred
    if label.split(', ')[1] == 'machine learning':
        label = ", ".join(label.split(', ')[1:])
    # if '\"' in pred:
    #     ori = pred
    #     pred = pred.split('\"')
    #     if len(pred) <=2:
    #         pred = pred[0]
    #     elif len(pred) < 4:
    #         pred = pred[1]
    #     elif 4 <= len(pred) < 6:
    #         if ori.startswith('the most likely category for this paper is'):
    #             if pred[3] == 'computational complexity' or pred[3] == 'computational geometry':
    #                 pred = ", ".join([pred[1], pred[3]])
    #             else:
    #                 pred = pred[3]
    #         elif ori.startswith('the subcategory of the paper is'):
                
    #             pred = pred[1]
    #         elif ori.startswith('the subcategory of the paper'):
                
    #             pred = pred[3]
    #         elif ori.startswith('the category') or ori.startswith('the subcategory'):
                
    #             pred = pred[1]
    #         elif ori.startswith('the paper belongs to'):
                
    #             pred = pred[1]
    #         else:
    #             pred = pred[3]
    #     else:
    #         if pred[0] == '':
                
    #             pred = pred[1]
    #         else:
    #             if pred[1] == 'data structures, algorithms, and theory' or pred[1] == 'data structures  algorithms and theory':
    #                 pred = ', '.join([pred[1], pred[3]])
                    
    #             else:
    #                 pred = pred[1]
    # else:
    #     if pred.startswith('the subcategory of '):
    #         pred = pred[len('the subcategory of '):]
    
    # if pred.startswith('the most likely subcategory for this paper is:\n\n'):
    #     pred = pred.split('\n\n')[1]
    #     if pred.startswith('* '):
    #         pred = pred[2:]
    # if '\n' in pred:
    #     pred = pred.split('\n')[0]


    # if '\"' in pred:
    #     if pred.startswith('the paper \"'):
    #         if '* ' in pred:
    #             pred = pred.split('* ')[1]
    #         else:
    #             if len(pred.split('\"')) > 3:
    #                 pred = pred.split('\"')[3] 
    #             else:
    #                 pred = pred.split('\"')[1] 
    #     else:
    #         pred = pred.split('\"')[1]    

    # if '\n' in pred:
    #     print(pred)
    #     pred = pred.split('\n')[0]


    # if '\"' in pred:
    #     if pred.startswith('the paper \"'):
    #         pred = pred.split('\"')[3]
    #     else:
    #         if pred.split('\"')[1] == '':
    #             pred = pred.split('\"')[0]
    #         else:
    #             pred = pred.split('\"')[1]
    
    # if '\n\n' in pred:
    #     pred = pred.split('\n\n')[1]
    #     if pred.startswith('* '):
    #         pred = pred[2:]

    # if '\n' in pred:
    #     pred = pred.split('\n')[0]
    if '\"' in pred:
        ls = pred.split('\"')
        for i in ls:
            if i.endswith('.') or i.endswith('?') or i.endswith(','):
                ans = i[:-1]
            # if 'pattern recognition' in i:
            #     ans = 'artificial intelligence, vision and pattern recognition'
            if i.startswith('data structures, algorithms,'):
                ans = i.replace('data structures, algorithms,', 'data structures, algorithms', 1)
            elif i.startswith('data structures '):
                ans = i.replace('data structures ', 'data structures,', 1)
            elif i == 'hardware and architecture, input/output and storage':
                ans = 'hardware and architecture, input output and storage'
            elif i in ['artificial intelligence, natural language processing(nlp)', 'artificial intelligence, natural language processing']:
                ans = 'artificial intelligence, nlp'
            else:
                ans = i

            if ans.split(', ')[-1] in ['agents', 'data mining', 'expert systems', 'games and search', 'knowledge representation',
                'nlp', 'planning', 'robotics', 'speech', 'theorem proving', 
                'vision and pattern recognition']:
                if len(ans.split(', ')) < 2 or ans.split(', ')[-2] in ['artificial intelligence, machine learning', 'artificial intelligence', 'machine learning']:
                    ans = 'artificial intelligence, ' + ans.split(', ')[-1]
            elif ans.split(', ')[-1] in ['case-based', 'genetic algorithms', 'neural networks', 'probabilistic methods', 'reinforcement learning', 'rule learning', 'theory']:
                if len(ans.split(', ')) < 2 or ans.split(', ')[-2] in ['artificial intelligence, machine learning', 'artificial intelligence', 'machine learning']:
                    ans = 'machine learning, ' + ans.split(', ')[-1]
            elif ans.split(', ')[0] in ['computational complexity', 'computational geometry']:
                ans = 'data structures, algorithms and theory, ' + ans.split(', ')[0]
            
            if ans in label_list:
                pred = ans
                break
        else:
            pass
    else:
        pred = pred
        
    if pred.endswith('natural language processing') or pred.endswith('(nlp)'):
        pred = 'artificial intelligence, nlp'
        
    if pred.startswith('. '):
        pred = pred[2:]
    elif pred.startswith('.'):
        pred = pred[1:]
    elif pred.endswith('.') or pred.endswith('?') or pred.endswith(','):
        pred = pred[:-1]
    
    pred = pred.strip()

    if label.startswith('data structures '):
        label = label.replace('data structures ', 'data structures,', 1)

    # if 'pattern recognition' in pred:
    #     pred = 'artificial intelligence, vision and pattern recognition'
    if pred.startswith('data structures '):
        pred = pred.replace('data structures ', 'data structures,', 1)
    if pred.startswith('data structures, algorithms,'):
        pred = pred.replace('data structures, algorithms,', 'data structures, algorithms', 1)
    if pred == 'hardware and architecture, input/output and storage':
        pred = 'hardware and architecture, input output and storage'
    if pred == 'artificial intelligence, natural language processing':
        pred = 'artificial intelligence, nlp'
    
    if pred.split(', ')[-1] in ['agents', 'data mining', 'expert systems', 'games and search', 'knowledge representation',
        'nlp', 'planning', 'robotics', 'speech', 'theorem proving', 
        'vision and pattern recognition']:
        if len(pred.split(', ')) < 2 or pred.split(', ')[-2] in ['artificial intelligence, machine learning', 'artificial intelligence', 'machine learning']:
            pred = 'artificial intelligence, ' + pred.split(', ')[-1]
    elif pred.split(', ')[-1] in ['case-based', 'genetic algorithms', 'neural networks', 'probabilistic methods', 'reinforcement learning', 'rule learning', 'theory']:
        if len(pred.split(', ')) < 2 or pred.split(', ')[-2] in ['artificial intelligence, machine learning', 'artificial intelligence', 'machine learning']:
            pred = 'machine learning, ' + pred.split(', ')[-1]
    elif pred.split(', ')[0] in ['computational complexity', 'computational geometry']:
        pred = 'data structures, algorithms and theory, ' + pred.split(', ')[0]

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
