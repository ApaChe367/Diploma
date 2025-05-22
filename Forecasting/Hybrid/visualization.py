"""Visualization functions for solar production forecasting."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

def plot_training_history(history, output_dir):
    """
    Plot training history.
    
    Args:
        history: Training history
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('Training and Validation MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()


def plot_predictions(y_true, y_pred, output_dir):
    """
    Plot actual vs predicted values.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Scatter plot
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('Actual Energy (Wh)')
    plt.ylabel('Predicted Energy (Wh)')
    plt.title('Actual vs. Predicted')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Histogram of errors
    plt.subplot(2, 2, 2)
    errors = y_pred - y_true
    plt.hist(errors, bins=50, alpha=0.75, color='skyblue')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error (Wh)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Time series (if timestamps are available)
    plt.subplot(2, 2, 3)
    plt.plot(range(len(y_true)), y_true, label='Actual')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Energy (Wh)')
    plt.title('Time Series Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Residuals
    plt.subplot(2, 2, 4)
    plt.scatter(y_pred, errors, alpha=0.3, s=10)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Energy (Wh)')
    plt.ylabel('Residuals (Wh)')
    plt.title('Residuals vs. Predicted')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.png'), dpi=300)
    plt.close()


def plot_feature_importance(model, feature_names, output_dir):
    """
    Plot feature importance based on model weights.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        output_dir: Directory to save plots
    """
    # This is a placeholder for more sophisticated feature importance analysis
    # For hybrid models, this would require custom implementations
    # For now, we just create a placeholder visualization
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance Analysis')
    plt.text(0.5, 0.5, 'Feature importance for hybrid models\nrequires custom implementation',
             ha='center', va='center', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()


def plot_future_predictions(future_df, output_dir):
    """
    Plot future predictions.
    
    Args:
        future_df: DataFrame with future predictions
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(future_df['timestamp'], future_df['predicted_energy'], 'o-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Predicted Energy (Wh)')
    plt.title('Future Energy Production Forecast')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'future_predictions.png'), dpi=300)
    plt.close()


def create_dashboard(training_history, evaluation_results, future_predictions, output_dir):
    """
    Create a comprehensive dashboard with multiple visualizations.
    
    Args:
        training_history: Training history
        evaluation_results: Dictionary with evaluation metrics
        future_predictions: DataFrame with future predictions
        output_dir: Directory to save the dashboard
    """
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training history
    plot_training_history(training_history, output_dir)
    
    # Plot actual vs. predicted
    plot_predictions(
        evaluation_results['y_test'],
        evaluation_results['y_pred'],
        output_dir
    )
    
    # Plot future predictions
    plot_future_predictions(future_predictions, output_dir)
    
    # Create summary report
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write("Solar Production Forecasting Model - Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Evaluation Metrics:\n")
        f.write(f"  MAE: {evaluation_results['mae']:.2f} Wh\n")
        f.write(f"  RMSE: {evaluation_results['rmse']:.2f} Wh\n")
        f.write(f"  R²: {evaluation_results['r2']:.4f}\n")
        if not np.isnan(evaluation_results.get('mape', np.nan)):
            f.write(f"  MAPE: {evaluation_results['mape']:.2f}%\n")
        
        f.write("\nTraining Summary:\n")
        f.write(f"  Final training loss: {training_history.history['loss'][-1]:.6f}\n")
        f.write(f"  Final validation loss: {training_history.history['val_loss'][-1]:.6f}\n")
        f.write(f"  Final training MAE: {training_history.history['mean_absolute_error'][-1]:.2f} Wh\n")
        f.write(f"  Final validation MAE: {training_history.history['val_mean_absolute_error'][-1]:.2f} Wh\n")
        
        f.write("\nFuture Predictions:\n")
        f.write(f"  Average predicted energy: {future_predictions['predicted_energy'].mean():.2f} Wh\n")
        f.write(f"  Maximum predicted energy: {future_predictions['predicted_energy'].max():.2f} Wh\n")
        f.write(f"  Total predicted energy: {future_predictions['predicted_energy'].sum():.2f} Wh\n")
    
    print(f"Dashboard created in {output_dir}")