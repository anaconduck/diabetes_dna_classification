import tensorflow as tf
from tensorflow import keras

def build_lstm_model(config):
    vector_size = config['models'].get('computed_input_dim', config['encoding']['word2vec']['vector_size'])
    lstm_units = config['models']['lstm']['lstm_units']
    dense_1 = config['models']['lstm']['dense_units_1']
    dense_2 = config['models']['lstm']['dense_units_2']
    dropout = config['models']['lstm']['dropout_rate']
    output_units = config['models']['lstm']['output_units']
    use_scale_attention = config['models'].get('lstm', {}).get('use_scale_attention', False)
    
    # Check if we are using adaptive multi-scale
    encoding_type = config['encoding'].get('type', 'kmer_word2vec')
    if encoding_type == 'adaptive_word2vec':
        scales = config['encoding'].get('scales', [3, 4, 5])
        num_scales = len(scales)
    else:
        num_scales = 1
        
    if num_scales > 1:
        inputs = keras.layers.Input(shape=(None, vector_size, num_scales))
    else:
        inputs = keras.layers.Input(shape=(None, vector_size))
        
    x = inputs
    
    if num_scales > 1:
        if use_scale_attention:
            from src.adaptive_kmer.scale_attention import ScaleAttention
            x = ScaleAttention()(x)
        else:
            # Simple average over the scales (axis=-1)
            x = tf.reduce_mean(x, axis=-1)
            
    # x is now (batch_size, sequence_length, vector_size)
    # Mask out the [PAD] zero-vectors so LSTM ignores them
    x = keras.layers.Masking(mask_value=0.0)(x)
    
    # Apply L2 regularization to heavily penalize large weights and reduce overfitting on small dataset
    l2_reg = tf.keras.regularizers.l2(1e-3)
    
    x = keras.layers.LSTM(units=lstm_units, activation='tanh', kernel_regularizer=l2_reg)(x) # standard tanh to prevent exploding gradients
    x = keras.layers.Dense(units=dense_1, activation='relu', kernel_regularizer=l2_reg)(x)
    x = keras.layers.Dense(units=dense_2, activation='relu', kernel_regularizer=l2_reg)(x)
    x = keras.layers.Dropout(rate=dropout)(x)
    outputs = keras.layers.Dense(units=output_units, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model
