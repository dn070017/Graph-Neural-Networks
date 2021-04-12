import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf

def load_dataset(target):
    """load_dataset
    
    Load built-in dataset
    
    Parameters:
       target: 'cora'
    
    Return:
        dataset:
            Tensorflow dataset with field:
                1. x: feature tensor, shape: (1, F)
                2. y: one-hot encoded label tensor, shape (1, num_classes)
                3. edge: adjancy matrix, shape: (1, nodes)
                4. idx: reindex used to extract edge information, int.
        label:
            Original label for each class. 
        idx:
            Original index for each node.
    """
    
    def data_generator():
        for feature, label, edge, i in zip(X.values, y.values, A, np.arange(len(y))):
            yield {'x': feature, 'y': label, 'edge': edge, 'idx': i}
        
    target = target.lower()
    assert target in ['cora']
    if target == 'cora':
        nodes = pd.read_csv('./dataset/cora/cora.content', sep='\t', index_col=0, header=None)
        X = nodes.iloc[:, :-1]
        
        y = pd.get_dummies(nodes.iloc[:, -1])
        label = y.columns
        
        idx = pd.Series(np.arange(len(y)), index=y.index)
        
        edges = pd.read_csv('./dataset/cora/cora.cites', sep='\t')
        edges.columns = ['source', 'target']
        edges['source'] = edges['source'].apply(lambda x: idx[x])
        edges['target'] = edges['target'].apply(lambda x: idx[x])
        A = nx.to_numpy_array(nx.from_pandas_edgelist(edges))
    
    dataset = tf.data.Dataset.from_generator(data_generator, output_types={'x': tf.float32, 'y': tf.float32, 'edge': tf.float32, 'idx': tf.int32})
    idx = pd.Series(y.index, index=np.arange(len(y)))
    
    return dataset, label, idx