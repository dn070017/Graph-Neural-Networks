import networkx as nx
import pandas as pd

def load_dataset(target):
    target = target.lower()
    assert target in ['cora']
    if target == 'cora':
        edges = pd.read_csv('./dataset/cora/cora.cites', sep='\t')
        edges.columns = ['source', 'target']
        edges = nx.from_pandas_edgelist(edges)
        nodes = pd.read_csv('./dataset/cora/cora.content', sep='\t', index_col=0, header=None)
        X = nodes.iloc[:, :-1]
        y = pd.get_dummies(nodes.iloc[:, -1])

    return X, y, edges