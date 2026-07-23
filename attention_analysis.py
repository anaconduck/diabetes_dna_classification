import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from src.utils.config_parser import load_config
from src.datasets.data_loader import load_and_preprocess_data
from src.encoding.factory import EncoderFactory
from src.adaptive_kmer.scale_attention import ScaleAttention

def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def run_attention_analysis():
    print("="*60)
    print("STAGE 5: SCALE ATTENTION ANALYSIS")
    print("="*60)
    
    config = load_config("configs/config.yaml")
    
    print("Loading data and generating embeddings...")
    config['encoding']['type'] = 'adaptive_word2vec'
    config['encoding']['scales'] = [3, 4, 5]
    
    dataset = load_and_preprocess_data(config)
    training_set, test_set, _ = dataset
    
    encoder_factory = EncoderFactory(config)
    features_train, Y_train = encoder_factory.extract_features(training_set)
    features_test, Y_test = encoder_factory.extract_features(test_set)
    
    w2v_model = encoder_factory.train_embedder(features_train, config['paths']['weights_dir'])
    X_test_embedding = encoder_factory.create_embeddings(w2v_model, features_test)
    
    model_path = os.path.join(config['paths']['weights_dir'], "stage4_baseline_adaptive_multi-scale_scale_attention_model.h5")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run Stage 4 (Ablation Study) first.")
        return
        
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path, custom_objects={'ScaleAttention': ScaleAttention})
    
    attention_layer = None
    for layer in model.layers:
        if isinstance(layer, ScaleAttention):
            attention_layer = layer
            break
            
    if attention_layer is None:
        print("ScaleAttention layer not found in the model!")
        return
        
    W = attention_layer.W.numpy()
    b = attention_layer.b.numpy()
    
    x_transposed = np.transpose(X_test_embedding, axes=[0, 2, 1])
    
    scores = np.tensordot(x_transposed, W, axes=[[2], [0]]) + b
    scores = np.tanh(scores)
    
    alpha = softmax(scores, axis=1)
    alpha_squeezed = np.squeeze(alpha, axis=-1)
    
    scales = config['encoding']['scales']
    
    global_mean_alpha = np.mean(alpha_squeezed, axis=0)
    
    Y_test_np = np.array(Y_test)
    class_0_alpha = np.mean(alpha_squeezed[Y_test_np == 0], axis=0)
    class_1_alpha = np.mean(alpha_squeezed[Y_test_np == 1], axis=0)
    
    print("\n--- Attention Weights Report ---")
    for i, scale in enumerate(scales):
        print(f"Scale K={scale} | Global: {global_mean_alpha[i]:.4f} | Normal (0): {class_0_alpha[i]:.4f} | Diabetes (1): {class_1_alpha[i]:.4f}")
        
    outputs_dir = config['paths']['outputs_dir']
    os.makedirs(outputs_dir, exist_ok=True)
    
    x = np.arange(len(scales))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, global_mean_alpha, width, label='Global Average', color='skyblue')
    rects2 = ax.bar(x, class_0_alpha, width, label='Normal (Class 0)', color='lightgreen')
    rects3 = ax.bar(x + width, class_1_alpha, width, label='Diabetes (Class 1)', color='salmon')
    
    ax.set_ylabel('Attention Weight (\u03B1)')
    ax.set_title('Scale Attention Distribution across k-mers')
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={s}" for s in scales])
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plot_path = os.path.join(outputs_dir, "scale_attention_analysis.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nAttention visualization saved to: {plot_path}")

if __name__ == "__main__":
    run_attention_analysis()
