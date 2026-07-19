import tensorflow as tf
from tensorflow.keras import layers

class ScaleAttention(layers.Layer):
    """
    Custom Keras Layer that applies an attention mechanism across different scales.
    
    Input shape: (batch_size, vector_size, num_scales)
    Output shape:
        If return_attention_weights=False: (batch_size, vector_size) or (batch_size, max_len, vector_size)
        If return_attention_weights=True:  [(..., vector_size), (..., num_scales)]
    """
    def __init__(self, return_attention_weights=False, **kwargs):
        super(ScaleAttention, self).__init__(**kwargs)
        self.return_attention_weights = return_attention_weights

    def build(self, input_shape):
        # input_shape can be (batch, vector_size, num_scales) or (batch, max_len, vector_size, num_scales)
        self.vector_size = input_shape[-2]
        self.num_scales = input_shape[-1]
        
        # Trainable weights for attention scoring
        # We learn a weight matrix W and bias b to project the vector_size down to a score
        self.W = self.add_weight(name='attention_W',
                                 shape=(self.vector_size, 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        
        self.b = self.add_weight(name='attention_b',
                                 shape=(1,),
                                 initializer='zeros',
                                 trainable=True)
                                 
        super(ScaleAttention, self).build(input_shape)

    def call(self, inputs):
        # inputs can be 3D (batch, V, S) or 4D (batch, L, V, S)
        rank = len(inputs.shape)
        
        if rank == 3:
            # (batch, V, S) -> (batch, S, V)
            x = tf.transpose(inputs, perm=[0, 2, 1])
        elif rank == 4:
            # (batch, L, V, S) -> (batch, L, S, V)
            x = tf.transpose(inputs, perm=[0, 1, 3, 2])
        else:
            raise ValueError(f"ScaleAttention requires 3D or 4D input, got {rank}D")
        
        # Calculate attention scores
        # x's last dimension is V. W is (V, 1).
        # tensordot(x, W, axes=[[-1], [0]])
        # For rank 3: (batch, S, V) dot (V, 1) -> (batch, S, 1)
        # For rank 4: (batch, L, S, V) dot (V, 1) -> (batch, L, S, 1)
        scores = tf.tensordot(x, self.W, axes=[[-1], [0]]) + self.b
        scores = tf.nn.tanh(scores)
        
        # Apply softmax over the 'num_scales' axis which is now axis=-2
        # alpha shape: (batch, S, 1) or (batch, L, S, 1)
        alpha = tf.nn.softmax(scores, axis=-2)
        
        # Compute weighted sum
        # x is (..., S, V). alpha is (..., S, 1).
        weighted_x = x * alpha
        
        # Sum over the num_scales dimension (axis=-2)
        # fused_representation shape: (batch, V) or (batch, L, V)
        fused_representation = tf.reduce_sum(weighted_x, axis=-2)
        
        if self.return_attention_weights:
            return fused_representation, tf.squeeze(alpha, axis=-1)
            
        return fused_representation

    def get_config(self):
        config = super(ScaleAttention, self).get_config()
        config.update({
            'return_attention_weights': self.return_attention_weights
        })
        return config
