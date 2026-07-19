import copy
import pandas as pd
from src.utils.config_parser import load_config
from train import main

def run_stage4_ablation():
    """
    Stage 4: Ablation Study
    Evaluates the 2x2 factorial combinations of Adaptive Multi-scale and Scale Attention.
    """
    base_config = load_config("configs/config.yaml")
    
    # 4 Configurations for the Ablation Study
    experiments = [
        {
            'name': 'Baseline',
            'encoding_type': 'kmer_word2vec',
            'use_attention': False
        },
        {
            'name': 'Baseline + Adaptive Multi-scale',
            'encoding_type': 'adaptive_word2vec',
            'use_attention': False
        },
        {
            'name': 'Baseline + Scale Attention',
            'encoding_type': 'kmer_word2vec',
            'use_attention': True
        },
        {
            'name': 'Baseline + Adaptive Multi-scale + Scale Attention',
            'encoding_type': 'adaptive_word2vec',
            'use_attention': True
        }
    ]
    
    # Ensure baseline is using LSTM and the winning imbalance strategy
    fixed_model = 'lstm'
    fixed_ksize = 3
    fixed_scales = [3, 4, 5]
    
    results = []
    
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"RUNNING ABLATION: {exp['name']}")
        print(f"{'='*60}\n")
        
        # Deep copy to avoid cross-contamination
        run_config = copy.deepcopy(base_config)
        
        # Set base model
        run_config['models']['type'] = fixed_model
        
        # Set Encoding
        run_config['encoding']['type'] = exp['encoding_type']
        run_config['encoding']['kmer_size'] = fixed_ksize
        run_config['encoding']['scales'] = fixed_scales
        
        # Set Attention Flag
        if 'lstm' not in run_config['models']:
            run_config['models']['lstm'] = {}
        run_config['models']['lstm']['use_scale_attention'] = exp['use_attention']
        
        # Unique prefix for outputs
        prefix_name = exp['name'].replace(' + ', '_').replace(' ', '_').lower()
        run_config['run_prefix'] = f"stage4_{prefix_name}"
        
        # Run pipeline
        try:
            main(config_dict=run_config)
            
            # Retrieve metrics
            report_path = f"outputs/stage4_{prefix_name}_classification_report.txt"
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
                'Experiment': exp['name'],
                'Encoding': exp['encoding_type'],
                'Attention': exp['use_attention'],
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1_score,
                'ROC-AUC': roc_auc,
                'Status': 'Success'
            })
            
        except Exception as e:
            print(f"ERROR during {exp['name']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'Experiment': exp['name'],
                'Encoding': exp['encoding_type'],
                'Attention': exp['use_attention'],
                'Accuracy': None,
                'Precision': None,
                'Recall': None,
                'F1-Score': None,
                'ROC-AUC': None,
                'Status': f'Failed: {e}'
            })
            
    # Save Ablation Summary
    df_results = pd.DataFrame(results)
    df_results.to_excel("outputs/stage4_ablation_summary.xlsx", index=False)
    print("\nStage 4 Ablation Study completed! See outputs/stage4_ablation_summary.xlsx for details.")

if __name__ == "__main__":
    run_stage4_ablation()
