import json
import re
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# with open('./instruction/pubmed/labels.txt', 'r') as f:
#     eval_decode_label = json.load(f)

# with open('./instruction/pubmed/results.txt', 'r') as f:
#     eval_pred = json.load(f)
data = "photo_LP"
prefix = 'graphsage_1000tp_5token_512_neg0_computer_2_3400_more_epoch_LP_baseline'
# prefix = 'graphsage_0tp_5token_512_neg0_arxiv_linear_2_batch'
with open(f'./results/{data}/{prefix}_model_labels.txt', 'r') as f:
    eval_decode_label = json.load(f)

with open(f'./results/{data}/{prefix}_model_results.txt', 'r') as f:
    eval_pred = json.load(f)

label_list = [
    'Yes',
    'No',
]
label2idx = {k.lower(): v for v, k in enumerate(label_list)}
patterns = [label.lower() for label in label_list]

cnt = 0
y, x = [], []
for label, pred in zip(eval_decode_label, eval_pred):
    pred = pred.lower()
    label = label.lower()[:-1]
    
    # if pred.startswith('yes'):
    #     pred = 'these two books must have co-purchase relationships'
    # else:
    #     pred = 'these two books may not have co-purchase relationships'
    
    if pred.startswith('these two books have co-purchased or co-viewed relationships') or pred.startswith('these two products have co-purchased or co-viewed relationships') or pred.startswith('these two items have co-purchased or co-viewed relationships'):
        pred = 'yes'
    elif pred.startswith('these two books have co-purchased relationships') or pred.startswith('these two products have co-purchased relationships') or pred.startswith('these two items have co-purchased relationships'):
        pred = 'yes'
    elif pred.startswith('these two books may not have co-purchased or co-viewed relationships') or pred.startswith('these two products may not have co-purchased or co-viewed relationships') or pred.startswith('these two items may not have co-purchased or co-viewed relationships'):
        pred = 'no'
    elif 'it is likely that these two products have co-purchased' in pred or 'it is possible that these two products have co-purchased' in pred or 'it seems that these two products have co-purchased' in pred:
        pred = 'yes'
    elif 'it is unlikely that these two products have co-purchased' in pred:
        pred = 'no'
    elif 'it is not possible to determine the relationship between' in pred or 'it is difficult to determine the' in pred or 'it is not clear what the relationship' in pred:
        pred = 'yes'
    elif 'two products have a negative relationship' in pred:
        pred = 'no'
    elif 'two products have a positive relationship' in pred:
        pred = 'yes'
    else:
        pred = pred

    # matches = []
    # for pattern in patterns:
    #     match_label = re.findall(pattern, pred)
    #     if len(match_label) >= 1:
    #         matches.append(match_label[0])
    # if len(matches) >= 1:
    #     pred = matches[0]
    # else:
    #     pred = pred
        
    if pred not in label2idx.keys():
        print("|"+pred+"|")
        cnt += 1
        y.append(label2idx[label])
        x.append(0)
    else:
        y.append(label2idx[label])
        x.append(label2idx[pred])

acc = accuracy_score(y, x)
auc = roc_auc_score(y, x)
res = classification_report(y, x)

print(cnt)
print(acc)
print(auc)
print(res)
