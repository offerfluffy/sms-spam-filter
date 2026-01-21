import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import pandas as pd
import os


class ModelEvaluator:
    """Class for visualizing model performance."""

    def __init__(self, output_dir='results/'):
        """
        Initialize evaluator and create results directory.
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def plot_confusion_matrix(self, model, X_test, y_test, model_name):
        """
        Plot and save Confusion Matrix.
        """
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Save plot
        plt.savefig(f'{self.output_dir}/{model_name}_confusion_matrix.png')
        plt.close()
        print(f"Saved confusion matrix for {model_name}")

    def plot_roc_curve(self, model, X_test, y_test, model_name):
        """
        Plot and save ROC Curve (requires predict_proba).
        """
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")

            # Save plot
            plt.savefig(f'{self.output_dir}/{model_name}_roc_curve.png')
            plt.close()
            print(f"Saved ROC curve for {model_name}")

    def plot_model_comparison(self, results: dict):
        """
        Plot comparison of model accuracies.
        """
        models = list(results.keys())
        accuracies = list(results.values())

        plt.figure(figsize=(10, 6))
        sns.barplot(x=models, y=accuracies, palette='viridis')
        plt.ylim(0.8, 1.0)  # Zoom in on the