import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
import random


paper_graph = torch.load('./data/graph_data_all.pt')

arxiv = paper_graph['arxiv']
pubmed = paper_graph['pubmed']
cora = paper_graph['cora']

edge_lists = arxiv['edge_index']
adj = [set() for _ in range(arxiv['x'].shape[0])]
adj_in = [set() for _ in range(arxiv['x'].shape[0])]
adj_out = [set() for _ in range(arxiv['x'].shape[0])]
for i in range(edge_lists.shape[1]):
    x, y = edge_lists[0, i].item(), edge_lists[1, i].item()
    adj[x].add(y)
    adj[y].add(x)
    adj_out[x].add(y)
    adj_in[y].add(x)
degree = [len(list(i)) for i in adj]
in_degree = [len(list(i)) for i in adj_in]
out_degree = [len(list(i)) for i in adj_out]

fig, axs = plt.subplots(1, 3)
sns.histplot(degree, ax=axs[0], bins=100)
# sns.histplot(in_degree, ax=axs[1])
# sns.histplot(out_degree, ax=axs[2])
plt.show()



