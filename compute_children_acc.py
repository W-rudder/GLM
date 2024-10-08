import json
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, classification_report

prefix = 'graphsage_0tp_5token_512_neg0_computer_1_3400_more_epoch'
# prefix = 'soft_promt_5'

with open(f'./results/book_children/{prefix}_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open(f'./results/book_children/{prefix}_model_results.txt', 'r') as f:
    eval_pred = json.load(f)

label_list = [
    'Literature & Fiction',
    'Animals',
    'Growing Up & Facts of Life',
    'Humor',
    'Cars, Trains & Things That Go',
    'Fairy Tales, Folk Tales & Myths',
    'Activities, Crafts & Games',
    'Science Fiction & Fantasy',
    'Classics',
    'Mysteries & Detectives',
    'Action & Adventure',
    'Geography & Cultures',
    'Education & Reference',
    'Arts, Music & Photography',
    'Holidays & Celebrations',
    'Science, Nature & How It Works',
    'Early Learning',
    'Biographies',
    'History',
    "Children's Cookbooks",
    'Religions',
    'Sports & Outdoors',
    'Comics & Graphic Novels',
    'Computers & Technology'
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
            elif ans.startswith('Religion'):
                pred = 'Religions'
                break
            # elif ans == "Children's Books":
            #     pred = "Children's Cookbooks"
            #     break
        else:
            pass
    else:
        pred = pred
        
    
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
print(accuracy_score(y, x))
print(f1)
print(p)
print(r)
print(cnt / len(eval_decode_label))
print(classification_report(y, x))
