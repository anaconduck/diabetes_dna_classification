import tensorflow as tf
from tensorflow import keras

def build_dnn_model(config):
    vector_size = config['models'].get('computed_input_dim', config['encoding']['word2vec']['vector_size'])
    
    model_cfg = config['models'].get('dnn', config['models']['lstm'])
    
    dense_1 = model_cfg.get('dense_units_1', 128)
    dense_2 = model_cfg.get('dense_units_2', 64)
    dropout = model_cfg.get('dropout_rate', 0.2)
    output_units = model_cfg.get('output_units', 1)
    
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(vector_size,)),
        keras.layers.Dense(units=dense_1, activation='relu'),
        keras.layers.Dropout(rate=dropout),
        keras.layers.Dense(units=dense_2, activation='relu'),
        keras.layers.Dropout(rate=dropout),
        keras.layers.Dense(units=output_units, activation='sigmoid')
    ])
    
    return model
