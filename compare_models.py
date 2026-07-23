import copy
import pandas as pd
from src.utils.config_parser import load_config
from train import main

def run_model_comparison():
    base_config = load_config("configs/config.yaml")
    
    models = ['lstm', 'cnn', 'dnn', 'rf', 'svm']
    
    fixed_encoding = 'kmer_word2vec'
    fixed_ksize = 3
    
    results = []
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"BENCHMARKING MODEL: {model_type.upper()}")
        print(f"{'='*50}\n")
        
        # Deep copy to avoid cross-contamination
        run_config = copy.deepcopy(base_config)
        
        # Enforce fixed encoding for fair comparison
        run_config['encoding']['type'] = fixed_encoding
        run_config['encoding']['kmer_size'] = fixed_ksize
        
        # Set Model Type
        run_config['models']['type'] = model_type
        
        run_config['run_prefix'] = f"benchmark_{model_type}"
        
        try:
            main(config_dict=run_config)
            
            report_path = f"outputs/benchmark_{model_type}_classification_report.txt"
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
                'Model': model_type.upper(),
                'Encoding': fixed_encoding,
                'K-mer Size': fixed_ksize,
                'Status': 'Success',
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1_score,
                'ROC-AUC': roc_auc
            })
            
        except Exception as e:
            print(f"ERROR during benchmarking {model_type}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'Model': model_type.upper(),
                'Encoding': fixed_encoding,
                'K-mer Size': fixed_ksize,
                'Status': f'Failed: {e}',
                'Accuracy': None,
                'Precision': None,
                'Recall': None,
                'F1-Score': None,
                'ROC-AUC': None
            })
            
    df_results = pd.DataFrame(results)
    df_results.to_excel("outputs/model_comparison_summary.xlsx", index=False)
    print("\nAll benchmarking completed! See outputs/model_comparison_summary.xlsx for details.")

if __name__ == "__main__":
    run_model_comparison()
