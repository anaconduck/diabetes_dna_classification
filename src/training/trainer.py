import tensorflow as tf
import os
import numpy as np
import pickle

def train_model(model, X_train, Y_train, X_val, Y_val, config, class_weights=None):
    """
    Compiles and trains the model (if Keras), then saves the weights.
    For Scikit-Learn models, it just fits the model.
    """
    optimizer = config['training'].get('optimizer', 'adam')
    epochs = config['training'].get('epochs', 10)
    batch_size = config['training'].get('batch_size', 32)
    save_dir = config['paths']['weights_dir']
    model_type = config['models'].get('type', 'lstm')
    
    # Check if Keras model
    is_keras = hasattr(model, 'compile')
    
    # Reshaping
    X_train_final = X_train
    X_val_final = X_val
    if is_keras and model_type in ['lstm', 'attention_lstm', 'cnn']:
        if len(X_train.shape) == 2:
            X_train_final = np.expand_dims(X_train, axis=-1)
            X_val_final = np.expand_dims(X_val, axis=-1)
        else:
            X_train_final = X_train
            X_val_final = X_val
        
    Y_train_np = np.array(Y_train)
    Y_val_np = np.array(Y_val)
    
    history = None
    
    if is_keras:
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        
        # Add Early Stopping to prevent overfitting and restore best golden epoch weights
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True,
            verbose=1
        )
        
        history = model.fit(X_train_final, Y_train_np, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            validation_data=(X_val_final, Y_val_np),
                            class_weight=class_weights,
                            callbacks=[early_stopping])
    else:
        print(f"Training Scikit-Learn model ({model_type})...")
        model.fit(X_train_final, Y_train_np)
    
    # Save the model
    os.makedirs(save_dir, exist_ok=True)
    prefix = config.get('run_prefix', '')
    if prefix:
        prefix = f"{prefix}_"
        
    if is_keras:
        model_path = os.path.join(save_dir, f'{prefix}model.h5')
        model.save(model_path)
    else:
        model_path = os.path.join(save_dir, f'{prefix}model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
    return model, history
