import tensorflow as tf
from tensorflow import keras

def build_cnn_model(config):
    vector_size = config['models'].get('computed_input_dim', config['encoding']['word2vec']['vector_size'])
    
    model_cfg = config['models'].get('cnn', config['models']['lstm'])
    
    filters = model_cfg.get('filters', 64)
    kernel_size = model_cfg.get('kernel_size', 3)
    dense_1 = model_cfg.get('dense_units_1', 50)
    dropout = model_cfg.get('dropout_rate', 0.2)
    output_units = model_cfg.get('output_units', 1)
    
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(None, vector_size)),
        keras.layers.Masking(mask_value=0.0),
        keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=dense_1, activation='relu'),
        keras.layers.Dropout(rate=dropout),
        keras.layers.Dense(units=output_units, activation='sigmoid')
    ])
    
    return model
