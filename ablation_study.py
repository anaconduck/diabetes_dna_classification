import copy
import pandas as pd
from src.utils.config_parser import load_config
from train import main

def run_ablation_study():
    base_config = load_config("configs/config.yaml")
    
    encodings = ['kmer_word2vec', 'one_hot']
    kmer_sizes = [3, 4, 5]
    
    results = []
    
    for encoding in encodings:
        for ksize in kmer_sizes:
            print(f"\n{'='*50}")
            print(f"RUNNING ABLATION: Encoding={encoding}, K-mer={ksize}")
            print(f"{'='*50}\n")
            
            # Deep copy to avoid modifying the base across iterations
            run_config = copy.deepcopy(base_config)
            
            # Override settings
            run_config['encoding']['type'] = encoding
            run_config['encoding']['kmer_size'] = ksize
            run_config['run_prefix'] = f"{encoding}_k{ksize}"
            
            # Run pipeline
            try:
                main(config_dict=run_config)
                
                # Retrieve metrics from classification report to store in excel
                report_path = f"outputs/{encoding}_k{ksize}_classification_report.txt"
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
                    'Encoding': encoding,
                    'K-mer Size': ksize,
                    'Status': 'Success',
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1_score,
                    'ROC-AUC': roc_auc
                })
                
            except Exception as e:
                print(f"ERROR during {encoding}_k{ksize}: {e}")
                results.append({
                    'Encoding': encoding,
                    'K-mer Size': ksize,
                    'Status': f'Failed: {e}',
                    'Accuracy': None,
                    'Precision': None,
                    'Recall': None,
                    'F1-Score': None,
                    'ROC-AUC': None
                })
                
    # Save Summary
    df_results = pd.DataFrame(results)
    df_results.to_excel("outputs/ablation_summary.xlsx", index=False)
    print("\nAll runs completed! See outputs/ablation_summary.xlsx for details.")

if __name__ == "__main__":
    run_ablation_study()
