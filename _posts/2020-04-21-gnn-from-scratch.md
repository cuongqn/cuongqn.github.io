---
layout: post
title: Predicting Chemical Properties with Graph Neural Networks from Scratch with PyTorch
hidden: false
---

Drug discovery is an expensive process. Specifically, compounds proposed by medicinal chemists need to be tested. To reduce the cost, *in silico* methods have historically been used to prioritize experiments. The prime examples are ML models trained to predict physicochemical properties from fixed chemical descriptors such as the ECFP family. Advances in geometric deep learning have enabled representation learning graphs via the Graph Convolutional Networks (GCN) family. In the blog post I'll walkthrough the math of GCN, build one from scratch, and train a few predictive models.
### SMILES to graph
I'll be using the lipophilicity dataset for demonstration. The most common format to share compounds is SMILES. First we need to convert SMILES to graphs. In short, we can describe a graph as a tuple of (A, F) where A is the adjacency matrix and F is the node features matrix. The adjacency matrix describes the connectivity between nodes, while the node features matrix describes the content of each node. To convert SMILES to graph we'll use RDkit. First we need to create a RDMol object.
```python
mol = Chem.MolFromSmiles(smiles,sanitize=True)
```
Then to generate adjancecy matrix we'll use its `Chem.rdmolops.GetAdjacencyMatrix` function. 
```python
adjacency = Chem.rdmolops.GetAdjacencyMatrix(mol)
```
To get node features, we'll iterate through the atoms generate one hot encodings of their atomic numbers.
```python
ELEMENTS = [1, 5, 6, 7, 8, 9, 16, 17]
ELEMENTS_VOCAB = {char: i for i, char in enumerate(ELEMENTS)}
ELEMENTS_VOCAB["unk"] = max(ELEMENTS_VOCAB) + 1

node = [a.GetAtomicNum() for a in mol.GetAtoms()]
node = [ELEMENTS_VOCAB.get(a, ELEMENTS_VOCAB["unk"]) for a in node]
```

Combine everything together into a reusable function we get:
```python
from collections import namedtuple
import numpy as np
from functools import lru_cache

ELEMENTS = [1, 5, 6, 7, 8, 9, 16, 17]
ELEMENTS_VOCAB = {char: i for i, char in enumerate(ELEMENTS)}
ELEMENTS_VOCAB["unk"] = max(ELEMENTS_VOCAB)+1

Graph = namedtuple("Graph", ["adjacency", "node_features"])

@lru_cache(maxsize=None)
def smiles2graph(smiles: str) -> Graph:
    mol = Chem.MolFromSmiles(smiles,sanitize=True)
    adjacency = Chem.rdmolops.GetAdjacencyMatrix(mol)
    adjacency = adjacency + np.eye(len(adjacency))
    
    node = [a.GetAtomicNum() for a in mol.GetAtoms()]
    node = [ELEMENTS_VOCAB.get(a, ELEMENTS_VOCAB["unk"]) for a in node]
    node_features = np.zeros((len(node), len(ELEMENTS_VOCAB)))
    node_features[np.arange(len(node)),node] = 1
    
    graph = Graph(adjacency=adjacency, node_features=node_features)
    
    return graph
```

### The Dataset Object
PyTorch sampling is done by using its Dataset class. 
from torch.utils.data import Dataset
```python
class MoleculeDataset(Dataset):
    def __init__(self, smiles, labels, featurize_fn):
        self.smiles = smiles
        self.labels = labels
        self.featurize_fn = featurize_fn
        
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        graph = self.featurize_fn(self.smiles[idx])
        adj = torch.Tensor(graph.adjacency)
        node = torch.Tensor(graph.node_features)
        labels = torch.tensor(self.labels[idx])
        return {"adj": adj, "node": node, "labels": labels}
    

def pad_collate(batch):
    adj = [b["adj"] for b in batch]
    node = [b["node"] for b in batch]
    labels = [b["labels"] for b in batch]
    num_nodes = [len(n) for n in node]
    difference = [int(max(num_nodes) - n) for n in num_nodes]
    
    adj = [
        nn.functional.pad(tensor,(0,diff,0,diff)) 
        for tensor, diff in zip(adj, difference)
    ]
    adj = torch.stack(adj)
    
    node = [
        nn.functional.pad(tensor,(0,0,0,diff)) 
        for tensor, diff in zip(node, difference)
    ]
    node = torch.stack(node)
    
    labels = torch.stack(labels)
    
    return {"adj": adj, "node": node, "labels": labels}
```
### Nevergrad for gradient-free optimization

