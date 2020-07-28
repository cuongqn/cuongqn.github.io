---
layout: post
title: Predicting Chemical Properties with Graph Neural Networks from Scratch with PyTorch
hidden: True
---

Recently, there are many new graph neural network architectures being published and various open-source graph frameworks. In the blog post I'll go to back to the basics and build a Graph Convolutional Network from scratch in PyTorch, and use it for molecular property predictions.
### SMILES To Graph
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
ELEMENTS_VOCAB["unk"] = max(ELEMENTS_VOCAB) + 1

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

```python
from torch.utils.data import Dataset

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
```
Each instance returned from `MoleculeDataset` will have different dimensions. We could pre-emptively pad all graphs to the max number of atoms, however that will take up too much memory. A more reasonable is padding by batch. Let's write the custom collate fn!
```python
def pad_collate(batch: List[dict]):
    # Unroll elements in list
    adj = [b["adj"] for b in batch]
    node = [b["node"] for b in batch]
    labels = [b["labels"] for b in batch]

    # Calculate padding dimension
    num_nodes = [len(n) for n in node]
    difference = [int(max(num_nodes) - n) for n in num_nodes]
    
    # Pad adjacency and node features matrices
    adj = [
        nn.functional.pad(tensor,(0,diff,0,diff)) 
        for tensor, diff in zip(adj, difference)
    ]
    node = [
        nn.functional.pad(tensor,(0,0,0,diff)) 
        for tensor, diff in zip(node, difference)
    ]

    # Stack them batch-first
    adj = torch.stack(adj)
    node = torch.stack(node)
    labels = torch.stack(labels)
    
    return {"adj": adj, "node": node, "labels": labels}
```
### Graph Convolutional Network Implementation


