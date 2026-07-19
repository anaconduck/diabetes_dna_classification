import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, Y_test, config):
    """
    Evaluates the model and saves the classification report.
    """
    outputs_dir = config['paths']['outputs_dir']
    os.makedirs(outputs_dir, exist_ok=True)
    prefix = config.get('run_prefix', '')
    if prefix:
        prefix = f"{prefix}_"
        
    model_type = config['models'].get('type', 'lstm')
    is_keras = hasattr(model, 'compile')
    
    # Reshape if necessary
    X_test_final = X_test
    if is_keras and model_type in ['lstm', 'attention_lstm', 'cnn']:
        if len(X_test.shape) == 2:
            X_test_final = np.expand_dims(X_test, axis=-1)
        else:
            X_test_final = X_test
        
    if is_keras:
        y_pred_prob = model.predict(X_test_final).flatten()
        y_pred = np.where(y_pred_prob > 0.5, 1, 0)
    else:
        # Scikit-learn
        y_pred = model.predict(X_test_final)
        if hasattr(model, "predict_proba"):
            y_pred_prob = model.predict_proba(X_test_final)[:, 1]
        else:
            # Fallback if probability is not supported
            y_pred_prob = y_pred

    Y_test_np = np.array(Y_test)
    
    accuracy = accuracy_score(Y_test_np, y_pred)
    precision = precision_score(Y_test_np, y_pred, zero_division=0)
    recall = recall_score(Y_test_np, y_pred, zero_division=0)
    f1 = f1_score(Y_test_np, y_pred, zero_division=0)
    
    roc_auc = 0.0
    try:
        roc_auc = roc_auc_score(Y_test_np, y_pred_prob)
    except ValueError:
        pass # In case there's only one class in y_true
        
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(Y_test_np, y_pred)
    print(conf_matrix)
    
    print("Classification Report:")
    class_report = classification_report(Y_test_np, y_pred, zero_division=0)
    print(class_report)
    
    # Save the report to a file
    report_path = os.path.join(outputs_dir, f"{prefix}classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1-score: {f1}\n")
        f.write(f"ROC-AUC: {roc_auc}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix) + "\n\n")
        f.write("Classification Report:\n")
        f.write(class_report + "\n")
        
    # Plot ROC Curve
    try:
        fpr, tpr, thresholds = roc_curve(Y_test_np, y_pred_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic {prefix.strip("_")}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(outputs_dir, f'{prefix}roc_curve.png'))
        plt.close()
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")
        
    return accuracy, precision, recall, f1, roc_auc

def plot_history(history, config):
    """
    Plots the training and validation accuracy and loss over epochs.
    For Scikit-Learn models, history will be None, so we skip plotting.
    """
    if history is None:
        print("Skipping history plot (not a Keras model).")
        return
        
    outputs_dir = config['paths']['outputs_dir']
    os.makedirs(outputs_dir, exist_ok=True)
    prefix = config.get('run_prefix', '')
    if prefix:
        prefix = f"{prefix}_"
    
    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history.get('accuracy', []), label='Train Accuracy')
    plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
    plt.title(f'Model Accuracy {prefix.strip("_")}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(outputs_dir, f'{prefix}accuracy_curve.png'))
    plt.close()
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history.get('loss', []), label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Validation Loss')
    plt.title(f'Model Loss {prefix.strip("_")}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(outputs_dir, f'{prefix}loss_curve.png'))
    plt.close()
