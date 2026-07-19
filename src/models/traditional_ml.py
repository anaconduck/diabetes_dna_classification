from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def build_rf_model(config):
    """
    Builds a Random Forest Classifier.
    """
    model_cfg = config['models'].get('rf', {})
    n_estimators = model_cfg.get('n_estimators', 100)
    random_state = config['datasets'].get('random_state', 42)
    
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

def build_svm_model(config):
    """
    Builds a Support Vector Machine Classifier.
    """
    model_cfg = config['models'].get('svm', {})
    C = model_cfg.get('C', 1.0)
    kernel = model_cfg.get('kernel', 'rbf')
    random_state = config['datasets'].get('random_state', 42)
    
    # probability=True is required to output probabilities for ROC AUC
    return SVC(C=C, kernel=kernel, probability=True, random_state=random_state)
