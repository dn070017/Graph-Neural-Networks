import tensorflow as tf

class GraphAttentionLayer(tf.keras.layers.Layer):
    """GraphAttentionLayer
    
    Tensorflow implementation for Graph Attention Network.
    
    Parameters:
        output_dim: int, required.
            Output dimensions of GraphAttentionLayer. (F')
            
        num_attn_heads: int, default: 1, optional
            Number of attention heads (K in eq. 5)
        
        mask: boolean, default: False, optional
            Whether or not to build the connection between nodes using mask (faster computation with
            more memory consumption)
        
        dropout_rate: default: 0.6, optional
            Dropout rate used in both features and attention coefficients (before masking)
            
        regularizer: default: tf.keras.regularizers.l2(0.0005), optional
            Regularization applied on weight in fully-connected layer and attention coefficients.
            
        activation: tf.keras.activations. default: 'relu'
            Activation function used after aggregating the featrures (eq. 4)
            
    """
    def __init__(
        self, 
        output_dim,
        num_attn_heads=1,
        mask=False,
        dropout_rate=0.6,
        regularizer=tf.keras.regularizers.l2(0.0005),
        activation='relu'
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_attn_heads = num_attn_heads
        
        self.mask = mask
        self.kernels = []
        self.biases = []
        self.attn_kernels_1 = []
        self.attn_kernels_2 = []
        self.regularizer = regularizer
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        """Use the shape of input to initialize the shape of kernels.
        
        Parameters:
            input_shape: List[tf.TensorShape]
                List consists of two TensorShape objects. The first element corresponds to the 
                shape of the feature tensor H, and the second element corresponds to the shape of 
                the adjacency matrix A. For instance, given a graph with N nodes and each node has
                F distinct features, the input shape should be:
                - [TensorShape([None, F]), TensorShape([None, N])]
        """
        input_dim = int(input_shape[0][-1])
        for head in range(self.num_attn_heads):
            kernel = self.add_weight(f"weight_{head}", shape=(input_dim, self.output_dim), regularizer=self.regularizer)
            bias = self.add_weight(f"bias_{head}", shape=(self.output_dim,))
            attn_kernel_1 = self.add_weight(f"attn_weight_1_{head}", shape=(self.output_dim, 1), regularizer=self.regularizer)
            attn_kernel_2 = self.add_weight(f"attn_weight_2_{head}", shape=(self.output_dim, 1), regularizer=self.regularizer)
            
            self.kernels.append(kernel)
            self.biases.append(bias)
            self.attn_kernels_1.append(attn_kernel_1)
            self.attn_kernels_2.append(attn_kernel_2)
        
        return
    
    def call(self, inputs):
        """Forward propagation of GraphAttentionLayer.
        
        Parameters:
            input_shape: List[tf.Tensor]
                List consists of two Tensor objects. The first element corresponds to the feature 
                tensor H, and the second element corresponds to adjacency matrix A.
        """
        def concat_attn(node_A, node_attn, node_HW):
            """Concatenate attention without masking
        
            Parameters:
                node_A: List[tf.Tensor]
                    Adjacency matrix with respect to the target node, shape: (1, batch_nodes).

                node_attn:
                    linear combination of HW with respect to the target node in (fig. 1, left), 
                    shape: (1, 1).
                    
                node_HW:
                    HW_i (eq. 1). Used when there's no neighboring nodes. (this can maybe be removed
                    in the future, if we can make sure all nodes are connected with themselves)
                    
            Return:
                (attn, attn, attn):
                    Due to the limitation for tf.map_fn that the number of the return tensor needs
                    to be the same. The attn Tensor is the h_i vector in eq. 4. 
                    shape: (1, F')
            """
            neighbor = tf.where(node_A != 0)
            if tf.size(neighbor) == 0:
                res = tf.reshape(node_HW, (-1, ))
            else:
                neighbor_idx = tf.reshape(neighbor, (-1, )) # neighbor_idx.shape: (neighbor_nodes, )
                neighbor_attn = tf.gather(attn_2, neighbor_idx, axis=0) # neighbor_attn.shape: (neighbor_nodes, 1)
                neighbor_HW = tf.gather(HW, neighbor_idx, axis=0) # neighbor_HW.shape: (neighbor_nodes, F')
                attn = tf.transpose(node_attn + neighbor_attn) # attn.shape: (1, neighbor_nodes)
                attn = tf.keras.layers.LeakyReLU(alpha=0.2)(attn)
                attn = tf.keras.layers.Softmax()(attn) # dropout is not applied when not using mask
                res = tf.squeeze(tf.matmul(attn, neighbor_HW))  # res.shape: (F', )
            return res, res, res
        
        H = inputs[0] # H.shape: (batch_nodes, F)
        A = inputs[1] # A.shape: (batch_nodes, all_nodes)
        idx = inputs[2] # idx.shape: (batch_nodes, )
        batch_A = tf.gather(A, idx, axis=1)
        results = []
        
        for head in range(self.num_attn_heads):
            kernel = self.kernels[head]
            bias = self.biases[head]
            attn_kernel_1 = self.attn_kernels_1[head]
            attn_kernel_2 = self.attn_kernels_2[head]
            
            HW = tf.matmul(H, kernel) # HW.shape: (batch_nodes, F')
            HW = self.dropout(HW)
            attn_1 = tf.matmul(HW, attn_kernel_1) # attn_1.shape: (batch_nodes, 1)
            attn_2 = tf.matmul(HW, attn_kernel_2) # attn_2.shape: (batch_ndoes, 1)

            if self.mask:
                attn_1, attn_2 = tf.meshgrid(attn_1, attn_2) # attn.shape: (batch_nodes, batch_nodes)
                attn = tf.transpose(attn_1 + attn_2)
                attn += tf.float32.min * (1.0 - batch_A)

                attn = tf.keras.layers.LeakyReLU(alpha=0.2)(attn)
                attn = tf.keras.layers.Softmax()(attn)
                attn = self.dropout(attn)
                res = tf.transpose(tf.matmul(tf.transpose(HW), attn))
            else:
                res, _, _ = tf.map_fn(fn=lambda x: concat_attn(*x), elems=(batch_A, attn_1, HW))
              
            results.append(self.activation(res + bias))
        
        results = tf.concat(results, axis=1)
        
        return results