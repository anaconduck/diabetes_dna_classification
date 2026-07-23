from src.utils.config_parser import load_config
from src.datasets.data_loader import load_and_preprocess_data
from src.encoding.factory import EncoderFactory
from src.models.factory import get_model
from src.training.trainer import train_model
from src.evaluation.evaluator import evaluate_model, plot_history
import os

def main(config_dict=None):
    print("Loading configuration...")
    config = config_dict if config_dict else load_config("configs/config.yaml")
    
    training_set, test_set, class_weights = load_and_preprocess_data(config)
    print(f"Training set shape: {training_set.shape}")
    print(f"Testing set shape: {test_set.shape}")
    
    encoder_factory = EncoderFactory(config)
    
    print("Extracting sequence features...")
    X_train_features, Y_train = encoder_factory.extract_features(training_set)
    X_test_features, Y_test = encoder_factory.extract_features(test_set)
    
    print("Training sequence embedder model...")
    embedder_model = encoder_factory.train_embedder(X_train_features, save_dir=config['paths']['weights_dir'])
    
    print("Creating sequence embeddings...")
    X_train_embedding = encoder_factory.create_embeddings(embedder_model, X_train_features)
    X_test_embedding = encoder_factory.create_embeddings(embedder_model, X_test_features)
    
    print("Building model...")
    model_type = config.get('models', {}).get('type', 'lstm')
    is_sequential = model_type in ['lstm', 'attention_lstm', 'cnn']
    if is_sequential:
        config['models']['computed_input_dim'] = X_train_embedding.shape[2]
    else:
        config['models']['computed_input_dim'] = X_train_embedding.shape[1]
    model = get_model(config)
    model.summary()
    
    print("Training the model...")
    model, history = train_model(model, X_train_embedding, Y_train, X_test_embedding, Y_test, config, class_weights)
    
    plot_history(history, config)
    
    print("Evaluating the model...")
    evaluate_model(model, X_test_embedding, Y_test, config)
    print("Results saved in outputs/ directory.")

if __name__ == "__main__":
    main()
