from src.models.lstm import build_lstm_model
from src.models.dnn import build_dnn_model
from src.models.cnn import build_cnn_model
from src.models.traditional_ml import build_rf_model, build_svm_model

def get_model(config):
    """
    Factory function to build the model based on the configuration.
    """
    model_type = config.get('models', {}).get('type', 'lstm')
    
    if model_type == 'lstm':
        return build_lstm_model(config)
    elif model_type == 'dnn':
        return build_dnn_model(config)
    elif model_type == 'cnn':
        return build_cnn_model(config)
    elif model_type == 'rf':
        return build_rf_model(config)
    elif model_type == 'svm':
        return build_svm_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
