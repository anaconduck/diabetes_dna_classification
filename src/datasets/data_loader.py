import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def load_and_preprocess_data(config):
    """
    Loads dataset, applies undersampling to balance classes, 
    and splits into training and testing sets.
    """
    data_path = config['datasets']['data_path']
    test_size = config['datasets']['test_size']
    random_state = config['datasets']['random_state']
    minority_class = config['datasets']['minority_class']
    majority_class = config['datasets']['majority_class']
    n_samples = config['datasets']['undersample_n_samples']
    imbalance_strategy = config['datasets'].get('imbalance_strategy', 'undersample')

    # Load Data
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
        df = pd.read_excel(data_path)
    else:
        raise ValueError("Dataset format not supported. Please use .csv or .xlsx")

    # Imbalance Handling
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

    # Split Dataset
    training_set, test_set = train_test_split(df_final, 
                                              test_size=test_size, 
                                              random_state=random_state)
                                              
    # Compute Class Weights if strategy requires it
    class_weights = None
    if imbalance_strategy == 'class_weight':
        classes = np.unique(training_set['class'])
        weights = compute_class_weight('balanced', classes=classes, y=training_set['class'])
        class_weights = dict(zip(classes, weights))

    return training_set, test_set, class_weights
