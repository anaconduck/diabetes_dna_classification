import tensorflow as tf
from tensorflow import keras

def build_attention_lstm_model(config):
    """
    Builds an LSTM model with a simple Attention mechanism architecture according to the configuration.
    """
    vector_size = config['models'].get('computed_input_dim', config['encoding']['word2vec']['vector_size'])
    
    # Retrieve hyperparameters specific to this model, or fallback to default lstm
    model_cfg = config['models'].get('attention_lstm', config['models']['lstm'])
    
    lstm_units = model_cfg['lstm_units']
    dense_1 = model_cfg['dense_units_1']
    dense_2 = model_cfg['dense_units_2']
    dropout = model_cfg['dropout_rate']
    output_units = model_cfg['output_units']
    
    # Using Functional API for Attention mechanism
    inputs = keras.Input(shape=(None, vector_size))
    
    # LSTM layer
    masked_inputs = keras.layers.Masking(mask_value=0.0)(inputs)
    lstm_out = keras.layers.LSTM(units=lstm_units, return_sequences=True, activation='tanh')(masked_inputs)
    
    # Simple self-attention mechanism
    attention_weights = keras.layers.Dense(1, activation='tanh')(lstm_out)
    attention_weights = keras.layers.Softmax(axis=1)(attention_weights)
    context_vector = keras.layers.Multiply()([lstm_out, attention_weights])
    context_vector = keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
    
    # Fully connected layers
    dense_out = keras.layers.Dense(units=dense_1, activation='relu')(context_vector)
    dense_out = keras.layers.Dense(units=dense_2, activation='relu')(dense_out)
    dense_out = keras.layers.Dropout(rate=dropout)(dense_out)
    outputs = keras.layers.Dense(units=output_units, activation='sigmoid')(dense_out)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model
