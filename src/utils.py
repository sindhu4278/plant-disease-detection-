import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def plot_history(history, model_name, save_dir):
    """Plots training and validation accuracy and loss, saving the plot."""
    os.makedirs(save_dir, exist_ok=True)
    
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    
    epochs_range = range(1, len(acc) + 1)
    
    plt.figure(figsize=(14, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='#ff7f0e', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy Curves')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='#2ca02c', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='#d62728', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss Curves')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_history.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved training curves to: {plot_path}")
    return plot_path

def plot_confusion_matrix(y_true, y_pred, classes, model_name, save_dir):
    """Generates and saves a confusion matrix heatmap."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=classes, 
        yticklabels=classes,
        cbar=True,
        annot_kws={"size": 12}
    )
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")
    return cm_path

def evaluate_predictions(y_true, y_pred, classes, model_name):
    """Computes and prints model classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    print("\n" + "="*50)
    print(f" EVALUATION REPORT FOR {model_name.upper()} ")
    print("="*50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("-"*50)
    print("Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))
    print("="*50 + "\n")
    
    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    }

def save_model_comparison(metrics_list, save_dir):
    """Saves comparison of multiple models to a CSV file."""
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(metrics_list)
    csv_path = os.path.join(save_dir, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved model comparison table to: {csv_path}")
    return csv_path
