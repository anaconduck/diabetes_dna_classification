import copy
import pandas as pd
from src.utils.config_parser import load_config
from train import main

def run_imbalance_study():
    """
    Runs a benchmark across different imbalanced data handling strategies.
    This helps in selecting the best strategy before doing model selection or feature ablation.
    """
    base_config = load_config("configs/config.yaml")
    
    # Strategies to compare
    strategies = ['none', 'undersampling', 'class_weight']
    
    # Fixed Settings for fair comparison
    fixed_model = 'lstm'
    fixed_encoding = 'kmer_word2vec'
    fixed_ksize = 3
    
    results = []
    
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"TESTING IMBALANCE STRATEGY: {strategy.upper()}")
        print(f"{'='*50}\n")
        
        # Deep copy to avoid cross-contamination
        run_config = copy.deepcopy(base_config)
        
        # Enforce fixed baseline
        run_config['models']['type'] = fixed_model
        run_config['encoding']['type'] = fixed_encoding
        run_config['encoding']['kmer_size'] = fixed_ksize
        
        # Set Strategy
        run_config['datasets']['imbalance_strategy'] = strategy
        
        # Unique prefix for outputs
        run_config['run_prefix'] = f"imbalance_{strategy}"
        
        # Run pipeline
        try:
            main(config_dict=run_config)
            
            # Retrieve metrics
            report_path = f"outputs/imbalance_{strategy}_classification_report.txt"
            accuracy = None
            precision = None
            recall = None
            f1_score = None
            roc_auc = None
            
            with open(report_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("Accuracy:"):
                        accuracy = float(line.split(":")[1].strip())
                    elif line.startswith("Precision:"):
                        precision = float(line.split(":")[1].strip())
                    elif line.startswith("Recall:"):
                        recall = float(line.split(":")[1].strip())
                    elif line.startswith("F1-score:"):
                        f1_score = float(line.split(":")[1].strip())
                    elif line.startswith("ROC-AUC:"):
                        roc_auc = float(line.split(":")[1].strip())
                        
            results.append({
                'Strategy': strategy.upper(),
                'Model': fixed_model.upper(),
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1_score,
                'ROC-AUC': roc_auc,
                'Status': 'Success'
            })
            
        except Exception as e:
            print(f"ERROR during imbalance strategy {strategy}: {e}")
            results.append({
                'Strategy': strategy.upper(),
                'Model': fixed_model.upper(),
                'Accuracy': None,
                'Precision': None,
                'Recall': None,
                'F1-Score': None,
                'ROC-AUC': None,
                'Status': f'Failed: {e}'
            })
            
    # Save Benchmark Summary
    df_results = pd.DataFrame(results)
    df_results.to_excel("outputs/imbalance_summary.xlsx", index=False)
    print("\nAll imbalance testing completed! See outputs/imbalance_summary.xlsx for details.")

if __name__ == "__main__":
    run_imbalance_study()
