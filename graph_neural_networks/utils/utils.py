import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf

def load_dataset(target, self_loop=True):
    """load_dataset
    
    Load built-in dataset
    
    Parameters:
        target:
            str, 'cora'
        self_loop:
            boolean, whether or not to add self loop to the graph.
    
    Return:
        X:
            feature numpy array, shape: (1, F)
        y:
            one-hot encoded label numpy array, shape (1, num_classes)
        A:
            sparse adjancency matrix, shape: (1, nodes)
        label:
            list of node labels.
        idx:
            dictionary of index mapping.
    """
    target = target.lower()
    assert target in ['cora']
    if target == 'cora':
        nodes = pd.read_csv('./dataset/cora/cora.content', sep='\t', index_col=0, header=None)
        node_index = nodes.index
        X = nodes.iloc[:, :-1].reset_index(drop=True).astype(np.float32)
        #X = (X.T - X.mean(axis=1)) / X.std(axis=1).T
        y = pd.get_dummies(nodes.iloc[:, -1]).reset_index(drop=True).astype(np.float32)
        label = y.columns
        
        idx = pd.Series(np.arange(len(node_index)), index=node_index)
        
        edges = pd.read_csv('./dataset/cora/cora.cites', sep='\t')
        edges.columns = ['source', 'target']
        edges['source'] = edges['source'].apply(lambda x: idx[x])
        edges['target'] = edges['target'].apply(lambda x: idx[x])
        graph = nx.from_pandas_edgelist(edges)
        A = nx.to_scipy_sparse_matrix(graph, nodelist=X.index).tolil().astype(np.float32)
        if self_loop:
            A.setdiag(1.0)

    return X.values, y.values, A, list(label), idx.to_dict()