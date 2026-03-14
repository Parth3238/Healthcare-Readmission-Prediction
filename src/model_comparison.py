"""
Model Comparison Script
Compares multiple ML algorithms for healthcare readmission prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Get project root
BASE_DIR = Path(__file__).parent.parent


def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    data_path = BASE_DIR / "data" / "healthcare_readmission_dataset.csv"
    df = pd.read_csv(data_path)
    
    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate multiple models."""
    
    # Scale data for algorithms that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'scaled': True
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'scaled': False
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'scaled': False
        },
        'SVM (RBF)': {
            'model': SVC(kernel='rbf', probability=True, random_state=42),
            'scaled': True
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'scaled': False
        },
        'AdaBoost': {
            'model': AdaBoostClassifier(random_state=42),
            'scaled': False
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'scaled': True
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(n_neighbors=5),
            'scaled': True
        }
    }
    
    results = {}
    trained_models = {}
    
    print("=" * 70)
    print("TRAINING AND EVALUATING MODELS")
    print("=" * 70)
    
    for name, config in models.items():
        print(f"\nTraining {name}...")
        
        model = config['model']
        
        # Use scaled or unscaled data
        if config['scaled']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        }
        
        trained_models[name] = model
        
        print(f"  Accuracy:  {results[name]['Accuracy']:.4f}")
        print(f"  Precision: {results[name]['Precision']:.4f}")
        print(f"  Recall:    {results[name]['Recall']:.4f}")
        print(f"  F1-Score:  {results[name]['F1-Score']:.4f}")
        print(f"  ROC-AUC:   {results[name]['ROC-AUC']:.4f}")
    
    return results, trained_models


def cross_validate_models(X, y, trained_models):
    """Perform cross-validation on all models."""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS (5-Fold)")
    print("=" * 70)
    
    cv_results = {}
    
    for name, model in trained_models.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_results[name] = cv_scores
        
        print(f"\n{name}:")
        print(f"  CV Scores: {cv_scores}")
        print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_results


def visualize_results(results, cv_results, output_dir):
    """Create visualizations for model comparison."""
    results_df = pd.DataFrame(results).T
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        bars = ax.bar(range(len(results_df)), results_df[metric], 
                       color=colors, edgecolor='black', linewidth=1.5)
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(results_df)))
        ax.set_xticklabels(results_df.index, rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    # Cross-validation plot
    ax = axes[5]
    cv_means = [cv_results[model].mean() for model in results_df.index]
    cv_stds = [cv_results[model].std() for model in results_df.index]
    
    bars = ax.bar(range(len(results_df)), cv_means, yerr=cv_stds,
                   color=colors, edgecolor='black', linewidth=1.5, capsize=5)
    ax.set_title('Cross-Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df.index, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('Machine Learning Model Comparison - Healthcare Readmission Prediction',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_model_comparison.png',
                dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'comprehensive_model_comparison.png'}")
    plt.show()
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(results_df, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'Score'})
    plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance_heatmap.png',
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_performance_heatmap.png'}")
    plt.show()


def save_best_model(results, trained_models, feature_names):
    """Save the best performing model."""
    # Find best model based on F1-Score
    best_model_name = max(results, key=lambda x: results[x]['F1-Score'])
    best_model = trained_models[best_model_name]
    
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_model_name}")
    print("=" * 70)
    print(f"Accuracy:  {results[best_model_name]['Accuracy']:.4f}")
    print(f"Precision: {results[best_model_name]['Precision']:.4f}")
    print(f"Recall:    {results[best_model_name]['Recall']:.4f}")
    print(f"F1-Score:  {results[best_model_name]['F1-Score']:.4f}")
    print(f"ROC-AUC:   {results[best_model_name]['ROC-AUC']:.4f}")
    
    # Save model
    model_path = BASE_DIR / "models" / "best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\nBest model saved to: {model_path}")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_path = BASE_DIR / "results" / "model_comparison_results.csv"
    results_df.to_csv(results_path)
    print(f"Results saved to: {results_path}")
    
    return best_model_name, best_model


def main():
    """Main function to run model comparison."""
    print("=" * 70)
    print("HEALTHCARE READMISSION PREDICTION - MODEL COMPARISON")
    print("=" * 70)
    
    # Load data
    print("\nLoading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(feature_names)}")
    
    # Create results directory
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Train and evaluate models
    results, trained_models = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    # Cross-validation
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    cv_results = cross_validate_models(X_full, y_full, trained_models)
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(results, cv_results, results_dir)
    
    # Save best model
    best_model_name, best_model = save_best_model(results, trained_models, feature_names)
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\nBest performing model: {best_model_name}")
    print("Check the 'results' folder for detailed visualizations.")


if __name__ == "__main__":
    main()
