import tensorflow as tf

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
        activation='relu'
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_attn_heads = num_attn_heads
        self.is_last = is_last

        self.kernels = []
        self.biases = []
        self.attn_kernels_1 = []
        self.attn_kernels_2 = []
        self.initializer = initializer
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
            kernel = self.add_weight(f"weight_{head}", shape=(input_dim, self.output_dim), initializer=self.initializer, regularizer=self.regularizer)
            bias = self.add_weight(f"bias_{head}", shape=(self.output_dim,))
            attn_kernel_1 = self.add_weight(f"attn_weight_1_{head}", shape=(self.output_dim, 1), initializer=self.initializer, regularizer=self.regularizer)
            attn_kernel_2 = self.add_weight(f"attn_weight_2_{head}", shape=(self.output_dim, 1), initializer=self.initializer, regularizer=self.regularizer)
            
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
        H = inputs[0] # H.shape: (batch_nodes, F)
        A = inputs[1] # A.shape: (batch_nodes, all_nodes)
        idx = inputs[2] # idx.shape: (batch_nodes, )
        batch_A = tf.gather(A, idx, axis=1)
        if(batch_A.shape[-1] == 1):
            batch_A = tf.squeeze(batch_A, axis=-1)
            
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

            attn_1, attn_2 = tf.meshgrid(attn_1, attn_2) # attn.shape: (batch_nodes, batch_nodes)
            attn = tf.transpose(attn_1 + attn_2)
            attn += tf.float32.min * (1.0 - batch_A)

            attn = tf.keras.layers.LeakyReLU(alpha=0.2)(attn)
            attn = tf.keras.layers.Softmax()(attn)
            attn = self.dropout(attn)
            
            res = tf.matmul(attn, HW)

            if self.is_last:
                results.append(res + bias)
            else:
                results.append(self.activation(res + bias))

        if self.is_last:
            results = self.activation(tf.math.add_n(results) / self.num_attn_heads)
        else:
            results = tf.concat(results, axis=1)
        
        return results