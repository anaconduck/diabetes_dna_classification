import os
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

from src.utils.config_parser import load_config
from src.encoding.factory import EncoderFactory
from src.models.factory import get_model
from src.training.trainer import train_model
from src.evaluation.evaluator import evaluate_model

def apply_imbalance_strategy(df, config):
    imbalance_strategy = config['datasets'].get('imbalance_strategy', 'none')
    minority_class = config['datasets']['minority_class']
    majority_class = config['datasets']['majority_class']
    n_samples = config['datasets'].get('undersample_n_samples', 100)
    random_state = config['datasets']['random_state']
    
    if imbalance_strategy == 'undersample':
        df_major = df[df['class'] == majority_class]
        df_minor = df[df['class'] == minority_class]
        
        df_downsampled = resample(df_major, 
                                  replace=False, 
                                  n_samples=n_samples, 
                                  random_state=random_state)
        df_final = pd.concat([df_downsampled, df_minor])
    else:
        df_final = df
        
    class_weights = None
    if imbalance_strategy == 'class_weight':
        classes = np.unique(df_final['class'])
        weights = compute_class_weight('balanced', classes=classes, y=df_final['class'])
        class_weights = dict(zip(classes, weights))
        
    return df_final, class_weights

def run_robustness_cv():
    print("="*60)
    print("STAGE 6: ROBUSTNESS EVALUATION (5-FOLD CROSS VALIDATION)")
    print("="*60)
    
    config = load_config("configs/config.yaml")
    
    config['models']['type'] = 'lstm'
    if 'lstm' not in config['models']:
        config['models']['lstm'] = {}
    config['models']['lstm']['use_scale_attention'] = True
    config['encoding']['type'] = 'adaptive_word2vec'
    config['encoding']['scales'] = [3, 4, 5]
    
    data_path = config['datasets']['data_path']
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)
        
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['datasets']['random_state'])
    
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(df, df['class'])):
        print(f"\n--- Running Fold {fold+1}/5 ---")
        
        # Split Data
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        # Apply Imbalance Strategy ONLY on Training Set
        train_df, class_weights = apply_imbalance_strategy(train_df, config)
        
        config['run_prefix'] = f"stage6_fold{fold+1}"
        
        # Feature Extraction
        print("Extracting multi-scale features...")
        encoder_factory = EncoderFactory(config)
        features_train, Y_train = encoder_factory.extract_features(train_df)
        features_test, Y_test = encoder_factory.extract_features(test_df)
        
        # Train Word2Vec 
        print("Training Adaptive Word2Vec...")
        w2v_model = encoder_factory.train_embedder(features_train, config['paths']['weights_dir'])
        
        print("Creating embeddings...")
        X_train_embedding = encoder_factory.create_embeddings(w2v_model, features_train)
        X_test_embedding = encoder_factory.create_embeddings(w2v_model, features_test)
        
        print("Building Model...")
        model = get_model(config)
        
        model, _ = train_model(model, X_train_embedding, Y_train, X_test_embedding, Y_test, config, class_weights)
        
        accuracy, precision, recall, f1, roc_auc = evaluate_model(model, X_test_embedding, Y_test, config)
        
        fold_metrics.append({
            'Fold': fold + 1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })
        
    df_results = pd.DataFrame(fold_metrics)
    
    mean_row = {
        'Fold': 'Mean',
        'Accuracy': df_results['Accuracy'].mean(),
        'Precision': df_results['Precision'].mean(),
        'Recall': df_results['Recall'].mean(),
        'F1-Score': df_results['F1-Score'].mean(),
        'ROC-AUC': df_results['ROC-AUC'].mean()
    }
    
    std_row = {
        'Fold': 'Std Dev',
        'Accuracy': df_results['Accuracy'].std(),
        'Precision': df_results['Precision'].std(),
        'Recall': df_results['Recall'].std(),
        'F1-Score': df_results['F1-Score'].std(),
        'ROC-AUC': df_results['ROC-AUC'].std()
    }
    
    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    outputs_dir = config['paths']['outputs_dir']
    os.makedirs(outputs_dir, exist_ok=True)
    out_path = os.path.join(outputs_dir, "stage6_robustness_summary.xlsx")
    df_results.to_excel(out_path, index=False)
    
    print(f"\nRobustness Evaluation completed! See {out_path} for detailed 5-Fold metrics.")
    print("Mean Performance:")
    print(f"Accuracy: {mean_row['Accuracy']:.4f} ± {std_row['Accuracy']:.4f}")
    print(f"F1-Score: {mean_row['F1-Score']:.4f} ± {std_row['F1-Score']:.4f}")
    print(f"ROC-AUC: {mean_row['ROC-AUC']:.4f} ± {std_row['ROC-AUC']:.4f}")

if __name__ == "__main__":
    run_robustness_cv()
