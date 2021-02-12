import tensorflow as tf

class GraphAttentionLayer(tf.keras.layers.Layer):
    """GraphAttentionLayer
    
    Tensorflow implementation for Graph Attention Network.
    
    Parameters:
        output_dim: int, required.
            Output dimensions of GraphAttentionLayer.
            
        num_attn_heads: int, default: 1, optional
            Number of attention heads (K in eq. 5)
            
        activation: tf.keras.activations. default: 'relu'
            Activation function used after aggregating the featrures (eq. 4)
            
    """
    def __init__(self, output_dim, num_attn_heads=1, activation='relu'):
        super().__init__()
        self.output_dim = output_dim
        self.num_attn_heads = num_attn_heads
        
        self.kernels = []
        self.biases = []
        self.attn_kernels_1 = []
        self.attn_kernels_2 = []
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
        input_dim = int(input_shape[0][-1]) # the shape of the first instance in the feature tensor H
        for head in range(self.num_attn_heads):
            kernel = self.add_weight(f"weight_{head}", shape=(input_dim, self.output_dim))
            bias = self.add_weight(f"bias_{head}", shape=(self.output_dim,))
            # let a = [a1, a2]^T
            # (Whi || Whi)a = Whia1 + Whia2
            attn_kernel_1 = self.add_weight(f"attn_weight_1_{head}", shape=(self.output_dim, 1))
            attn_kernel_2 = self.add_weight(f"attn_weight_2_{head}", shape=(self.output_dim, 1))
            
            self.kernels.append(kernel)
            self.biases.append(bias)
            self.attn_kernels_1.append(attn_kernel_1)
            self.attn_kernels_2.append(attn_kernel_2)
        
    def call(self, inputs):
        """Forward propagation of GraphAttentionLayer.
        
        Parameters:
            input_shape: List[tf.Tensor]
                List consists of two Tensor objects. The first element corresponds to the feature 
                tensor H, and the second element corresponds to adjacency matrix A.
        """
        H = inputs[0]
        A = inputs[1]
        
        results = []
        
        for head in range(self.num_attn_heads):
            kernel = self.kernels[head]
            bias = self.biases[head]
            attn_kernel_1 = self.attn_kernels_1[head]
            attn_kernel_2 = self.attn_kernels_2[head]
            
            HW = tf.matmul(H, kernel)
            attn_1 = tf.matmul(HW, attn_kernel_1)
            attn_2 = tf.matmul(HW, attn_kernel_2)
            attn_1, attn_2 = tf.meshgrid(attn_1, attn_2)
            attn = tf.transpose(attn_1 + attn_2)
             
            attn += tf.float32.min * (1.0 - A)
            
            attn = tf.keras.layers.LeakyReLU(alpha=0.2)(attn)
            attn = tf.keras.layers.Softmax()(attn)
            attn = tf.transpose(tf.matmul(tf.transpose(HW), attn))
            
            res = attn + bias
            
            results.append(self.activation(res))
        
        results = tf.concat(results, axis=1)
        
        return results