import tensorflow as tf

@tf.function
def tf_parallel_map(*args, **kwargs):
    return tf.map_fn(*args, **kwargs)

class GraphAttentionLayer(tf.keras.layers.Layer):
    """GraphAttentionLayer
    
    Tensorflow implementation for Graph Attention Network.
    
    Parameters:
        output_dim: int, required.
            Output dimensions of GraphAttentionLayer. (F')
            
        num_attn_heads: int, default: 1, optional
            Number of attention heads (K in eq. 5)
        
        dropout_rate: default: 0.6, optional
            Dropout rate used in both features and attention coefficients
            
        is_last: default: False, optional
            Whether or not this layer is the last layer. If set to True, average value
            would be taken from different attention heads, and the activation would be 
            applied after the average. If set to False, return contatenation of the 
            results from different attention heads.
        
        regularizer: default: tf.keras.regularizers.l2(0.0005), optional
            Regularization applied on weight in fully-connected layer and attention 
            coefficients.
            
        activation: tf.keras.activations. default: 'relu'
            Activation function used after aggregating the featrures (eq. 4)
            
    """
    def __init__(
        self, 
        output_dim,
        num_attn_heads=1,
        dropout_rate=0.6,
        is_last=False,
        initializer=tf.keras.initializers.glorot_normal(),
        regularizer=tf.keras.regularizers.l2(0.0005),
        activation='elu',
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_attn_heads = num_attn_heads
        self.is_last = is_last
        
        self.initializer = initializer
        self.regularizer = regularizer
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
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
        self.attn_kernels = self.add_weight(f"attn_kernel", shape=(self.num_attn_heads, 2, self.output_dim, 1), initializer=self.initializer, regularizer=self.regularizer)
        self.kernels = self.add_weight(f"weight", shape=(self.num_attn_heads, input_dim, self.output_dim), initializer=self.initializer, regularizer=self.regularizer)
        self.biases = self.add_weight(f"bias", shape=(self.num_attn_heads, self.output_dim))
        
        return

    def call(self, inputs):
        """Forward propagation of GraphAttentionLayer.
        
        Parameters:
            input_shape: List[tf.Tensor]
                List consists of two Tensor objects. The first element corresponds to the feature 
                tensor H, and the second element corresponds to adjacency matrix A.
        """
        # H.shape: (batch_nodes, F)
        # A.shape: (batch_nodes, all_nodes)
        # idx.shape: (batch_nodes, )
        H = inputs[0]
        A = inputs[1]
        idx = inputs[2] 
        
        batch_A = tf.gather(A, idx, axis=1)
        if(batch_A.shape[-1] == 1):
            batch_A = tf.squeeze(batch_A, axis=-1)

        # HW.shape: (batch_nodes, num_attn_heads, F')
        HW = tf.tensordot(H, self.kernels, axes=[[-1], [1]]) 
        HW = self.dropout(HW)

        # HW.shape: (num_attn_heads, batch_nodes, F')
        HW = tf.transpose(HW, [1, 0, 2])

        # unnorm_attn.shape: (num_attn_heads, batch_nodes, batch_nodes)
        unnorm_attn, _ = tf_parallel_map(GraphAttentionLayer.compute_unnorm_attn, (HW, self.attn_kernels))

        unnorm_attn = tf.where(tf.equal(batch_A, 0.0), tf.float32.min, unnorm_attn)

        unnorm_attn = tf.keras.layers.LeakyReLU(alpha=0.2)(unnorm_attn)
        norm_attn = tf.keras.layers.Softmax()(unnorm_attn)
        norm_attn = self.dropout(norm_attn)

        # results.shape: (num_attn_heads, batch_nodes, F')
        results, _, _ = tf_parallel_map(GraphAttentionLayer.attention_pooling, (HW, norm_attn, self.biases))
        
        if self.is_last:
            results = self.activation(tf.reduce_mean(results, axis=0))
        else:
            results = self.activation(results)
            results = tf.transpose(results, [1, 0, 2])
            results = tf.reshape(results, (-1, results.shape[1] * results.shape[2]))

        return results

    @staticmethod
    def compute_unnorm_attn(x):
        HW = x[0]
        attn_kernels = x[1]
        attn_kernel_1 = attn_kernels[0]
        attn_kernel_2 = attn_kernels[1]

        # attn_x.shape: (batch_nodes, 1)
        attn_1 = tf.matmul(HW, attn_kernel_1) 
        attn_2 = tf.matmul(HW, attn_kernel_2)

        # attn.shape: (batch_nodes, batch_nodes)
        attn_1, attn_2 = tf.meshgrid(attn_1, attn_2) 
        attn = tf.transpose(attn_1 + attn_2)

        return attn, attn

    @staticmethod
    def attention_pooling(x):
        HW = x[0]
        attn = x[1]
        bias = x[2]

        res = tf.tensordot(attn, HW, axes=[[-1], [0]]) + bias

        return res, res, res