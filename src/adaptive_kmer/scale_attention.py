import tensorflow as tf
from tensorflow.keras import layers

class ScaleAttention(layers.Layer):
    def __init__(self, return_attention_weights=False, **kwargs):
        super(ScaleAttention, self).__init__(**kwargs)
        self.return_attention_weights = return_attention_weights

    def build(self, input_shape):
        self.vector_size = input_shape[-2]
        self.num_scales = input_shape[-1]
        
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
        rank = len(inputs.shape)
        
        if rank == 3:
            x = tf.transpose(inputs, perm=[0, 2, 1])
        elif rank == 4:
            # (batch, L, V, S) -> (batch, L, S, V)
            x = tf.transpose(inputs, perm=[0, 1, 3, 2])
        else:
            raise ValueError(f"ScaleAttention requires 3D or 4D input, got {rank}D")
        
        scores = tf.tensordot(x, self.W, axes=[[-1], [0]]) + self.b
        scores = tf.nn.tanh(scores)
        
        alpha = tf.nn.softmax(scores, axis=-2)
        
        weighted_x = x * alpha
        
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
