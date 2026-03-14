import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from data_preprocessing import load_data, split_data

# Get the project root directory
BASE_DIR = Path(__file__).parent.parent

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        metrics['y_pred_proba'] = y_pred_proba
    
    metrics['y_pred'] = y_pred
    return metrics

def plot_confusion_matrix(cm, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")

def plot_roc_curve(y_test, y_pred_proba, output_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"ROC curve saved to: {output_path}")

def plot_feature_importance(model, feature_names, output_path):
    """Plot and save feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title('Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Feature importance saved to: {output_path}")
    else:
        print("Model does not have feature_importances_ attribute")

def main():
    # Create results directory
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Load model
    model_path = BASE_DIR / "models" / "model.pkl"
    if not model_path.exists():
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    
    # Load and split data
    data_path = BASE_DIR / "data" / "healthcare_readmission_dataset.csv"
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)
    
    print(f"Test set size: {len(X_test)} samples")
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Print metrics
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'], 
        results_dir / "confusion_matrix.png"
    )
    
    # ROC curve
    if 'y_pred_proba' in metrics:
        plot_roc_curve(
            y_test, 
            metrics['y_pred_proba'], 
            results_dir / "roc_curve.png"
        )
    
    # Feature importance
    plot_feature_importance(
        model, 
        X_test.columns.tolist(), 
        results_dir / "feature_importance.png"
    )
    
    print(f"\nAll evaluation results saved to: {results_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
