import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import differential_evolution
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Import your existing classes from the main script
import sys
sys.path.append(r"C:\Users\Lospsy\Desktop\Thesis\Forecast")
from Production_Forecast_V7_1_Ensemble import (
    SolarProductionPINN, 
    SolarProductionDataset,
    train_pinn,
    evaluate_model_improved
)

class BayesianHyperparameterOptimizer:
    """
    Bayesian optimization specifically tailored for your solar forecasting project.
    Uses your existing 179 grid search results as a warm start.
    """
    
    def __init__(self, existing_results_path, output_dir):
        self.existing_results_path = Path(existing_results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load existing results
        self._load_existing_results()
        
        # Initialize Gaussian Process
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=25
        )
        
        # Fit GP with existing data
        print(f"Fitting Gaussian Process with {len(self.X_observed)} existing configurations...")
        self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
        
        print(f"Bayesian Optimizer initialized")
        print(f"Current best validation loss: {-self.best_score:.6f}")
        print(f"Current best R²: {self.best_r2:.4f}")
        
    def _load_existing_results(self):
        """Load your 179 grid search results"""
        with open(self.existing_results_path, 'r') as f:
            data = json.load(f)
        
        self.grid_data = data
        self.X_observed = []
        self.y_observed = []
        self.all_configs = []
        self.all_r2_scores = []
        
        # Convert existing results
        for result in data['results']:
            config = result['config']
            metrics = result['metrics']
            
            # Convert to array format for GP
            x = np.array([
                config['hidden_size'],
                config['num_layers'],
                config['batch_size'],
                np.log10(config['learning_rate']),
                config['physics_weight']
            ])
            
            # Use negative val_loss for maximization
            y = -metrics['val_loss']
            
            self.X_observed.append(x)
            self.y_observed.append(y)
            self.all_configs.append(config)
            self.all_r2_scores.append(metrics['r2'])
        
        # Find current best
        self.best_idx = np.argmax(self.y_observed)
        self.best_score = self.y_observed[self.best_idx]
        self.best_config = self.all_configs[self.best_idx]
        self.best_r2 = self.all_r2_scores[self.best_idx]
        
    def acquisition_ucb(self, X, kappa=2.0):
        """Upper Confidence Bound acquisition function"""
        X = np.atleast_2d(X)
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu + kappa * sigma
    
    def acquisition_ei(self, X, xi=0.01):
        """Expected Improvement acquisition function"""
        X = np.atleast_2d(X)
        mu, sigma = self.gp.predict(X, return_std=True)
        
        with np.errstate(divide='warn'):
            Z = (mu - self.best_score - xi) / sigma
            ei = (mu - self.best_score - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
            
        return ei
    
    def suggest_next_params(self, n_candidates=10000, method='ei'):
        """
        Suggest next parameters using acquisition function optimization
        """
        # Define bounds
        bounds = [
            (16, 256),      # hidden_size
            (2, 6),         # num_layers
            (8, 128),       # batch_size
            (-4, -2),       # log10(learning_rate)
            (0.01, 0.5)     # physics_weight
        ]
        
        # Use appropriate acquisition function
        if method == 'ei':
            acq_func = lambda x: -self.acquisition_ei(x.reshape(1, -1))
        else:  # ucb
            acq_func = lambda x: -self.acquisition_ucb(x.reshape(1, -1))
        
        # Global optimization with differential evolution
        result = differential_evolution(
            acq_func,
            bounds,
            seed=42,
            maxiter=300,
            popsize=20,
            workers=1
        )
        
        # Check if we've tested something very similar
        x_next = result.x
        min_distance = min([np.linalg.norm(x_next - x_obs) for x_obs in self.X_observed])
        
        if min_distance < 0.5:  # Too similar to existing
            print("Suggested configuration too similar to existing. Exploring...")
            # Add random perturbation
            x_next += np.random.normal(0, 0.1, size=x_next.shape)
            # Clip to bounds
            for i, (low, high) in enumerate(bounds):
                x_next[i] = np.clip(x_next[i], low, high)
        
        # Convert to parameter dictionary
        params = {
            'hidden_size': int(round(x_next[0] / 8) * 8),  # Round to nearest 8
            'num_layers': int(round(x_next[1])),
            'batch_size': int(round(x_next[2] / 8) * 8),   # Round to nearest 8
            'learning_rate': round(10**x_next[3], 5),
            'physics_weight': round(x_next[4], 3)
        }
        
        return params, x_next
    
    def update_with_result(self, params, val_loss, r2, save=True):
        """Update GP with new result"""
        # Convert params to array
        x = np.array([
            params['hidden_size'],
            params['num_layers'],
            params['batch_size'],
            np.log10(params['learning_rate']),
            params['physics_weight']
        ])
        
        y = -val_loss  # Negative for maximization
        
        # Add to observations
        self.X_observed.append(x)
        self.y_observed.append(y)
        self.all_configs.append(params)
        self.all_r2_scores.append(r2)
        
        # Update best if necessary
        if y > self.best_score:
            self.best_score = y
            self.best_config = params
            self.best_r2 = r2
            print(f"🎉 New best configuration found!")
        
        # Refit GP
        print("Refitting Gaussian Process...")
        self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
        
        # Save progress
        if save:
            self._save_progress()
    
    def _save_progress(self):
        """Save current optimization state"""
        results = {
            'total_configs_tested': len(self.X_observed),
            'grid_search_configs': len(self.grid_data['results']),
            'bayesian_configs': len(self.X_observed) - len(self.grid_data['results']),
            'best_config': self.best_config,
            'best_val_loss': -self.best_score,
            'best_r2': self.best_r2,
            'all_bayesian_configs': self.all_configs[len(self.grid_data['results']):],
            'all_bayesian_val_losses': [-y for y in self.y_observed[len(self.grid_data['results']):]],
            'all_bayesian_r2_scores': self.all_r2_scores[len(self.grid_data['results']):]
        }
        
        with open(self.output_dir / "bayesian_progress.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    def plot_optimization_progress(self):
        """Create visualization of optimization progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Validation loss over iterations
        ax = axes[0, 0]
        grid_losses = [-y for y in self.y_observed[:len(self.grid_data['results'])]]
        bayesian_losses = [-y for y in self.y_observed[len(self.grid_data['results']):]]
        
        ax.scatter(range(len(grid_losses)), grid_losses, alpha=0.5, color='orange', label='Grid Search')
        if bayesian_losses:
            ax.scatter(range(len(grid_losses), len(self.y_observed)), bayesian_losses, 
                      alpha=0.7, color='green', s=100, label='Bayesian')
        
        # Plot best so far
        best_so_far = []
        current_best = float('inf')
        for loss in [-y for y in self.y_observed]:
            current_best = min(current_best, loss)
            best_so_far.append(current_best)
        ax.plot(range(len(best_so_far)), best_so_far, 'r-', linewidth=2, label='Best So Far')
        
        ax.axvline(x=len(grid_losses), color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Configuration Number')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Optimization Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. R² scores
        ax = axes[0, 1]
        grid_r2 = self.all_r2_scores[:len(self.grid_data['results'])]
        bayesian_r2 = self.all_r2_scores[len(self.grid_data['results']):]
        
        ax.scatter(range(len(grid_r2)), grid_r2, alpha=0.5, color='orange')
        if bayesian_r2:
            ax.scatter(range(len(grid_r2), len(self.all_r2_scores)), bayesian_r2, 
                      alpha=0.7, color='green', s=100)
        
        ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='R²=0.9')
        ax.axvline(x=len(grid_r2), color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Configuration Number')
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Parameter exploration (hidden_size vs num_layers)
        ax = axes[1, 0]
        X_array = np.array(self.X_observed)
        scatter = ax.scatter(X_array[:, 0], X_array[:, 1], 
                            c=[-y for y in self.y_observed], 
                            cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Val Loss')
        ax.set_xlabel('Hidden Size')
        ax.set_ylabel('Number of Layers')
        ax.set_title('Parameter Space Exploration')
        ax.grid(True, alpha=0.3)
        
        # 4. Learning curve - Bayesian vs Random
        ax = axes[1, 1]
        if len(self.y_observed) > len(self.grid_data['results']):
            n_bayesian = len(self.y_observed) - len(self.grid_data['results'])
            
            # Actual Bayesian progress
            bayesian_best = []
            current_best = min(grid_losses)
            for loss in bayesian_losses:
                current_best = min(current_best, loss)
                bayesian_best.append(current_best)
            
            # Simulated random search from same starting point
            random_losses = np.random.choice(grid_losses, size=n_bayesian)
            random_best = []
            current_best = min(grid_losses)
            for loss in random_losses:
                current_best = min(current_best, loss)
                random_best.append(current_best)
            
            iterations = range(1, n_bayesian + 1)
            ax.plot(iterations, bayesian_best, 'g-', linewidth=2, label='Bayesian')
            ax.plot(iterations, random_best, 'r--', linewidth=2, label='Random (simulated)')
            ax.set_xlabel('Bayesian Iterations')
            ax.set_ylabel('Best Validation Loss')
            ax.set_title('Bayesian vs Random Search Efficiency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "bayesian_progress.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional plot: Parameter importance
        self._plot_parameter_importance()
    
    def _plot_parameter_importance(self):
        """Analyze which parameters have the most impact"""
        if len(self.y_observed) < 20:
            return
        
        from sklearn.ensemble import RandomForestRegressor
        
        # Prepare data
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        # Fit random forest for feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        param_names = ['Hidden Size', 'Num Layers', 'Batch Size', 'Learning Rate (log)', 'Physics Weight']
        importances = rf.feature_importances_
        
        # Plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(param_names, importances, color='skyblue')
        plt.ylabel('Relative Importance')
        plt.title('Parameter Importance Analysis')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, imp in zip(bars, importances):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f'{imp:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "parameter_importance_bayesian.png", dpi=300, bbox_inches='tight')
        plt.close()


def train_and_evaluate_config(config, dataset, device):
    """
    Train and evaluate a single configuration.
    This replaces the placeholder evaluation function.
    """
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            num_workers=0, pin_memory=True)
    
    # Create model
    model = SolarProductionPINN(
        input_size=len(dataset.feature_names),
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        panel_efficiency=0.146,  # Your values
        panel_area=1.642,
        temp_coeff=-0.0044
    ).to(device)
    
    # Train model
    model, history = train_pinn(
        model, train_loader, val_loader,
        epochs=150,  # Reduced for faster iteration
        lr=config['learning_rate'],
        physics_weight=config['physics_weight'],
        device=device,
        early_stopping=True,
        patience=20,
        verbose=True
    )
    
    # Evaluate
    eval_results = evaluate_model_improved(model, test_loader, dataset, device=device)
    
    # Return key metrics
    return {
        'val_loss': history['val_loss'][-1],
        'train_loss': history['train_loss'][-1],
        'test_loss': eval_results['test_loss'],
        'r2': eval_results['r2'],
        'mae': eval_results['mae'],
        'rmse': eval_results['rmse'],
        'epochs_trained': len(history['train_loss'])
    }


def main():
    """
    Main function to run Bayesian optimization continuing from grid search
    """
    print("="*80)
    print("BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print("Continuing from Grid Search Results")
    print("="*80)
    
    # Configuration
    grid_results_path = r"C:\Users\Lospsy\Desktop\Thesis\Results\PINN_hyperparameter_tuning\hyperparameter_tuning\tuning_results.json"
    output_dir = r"C:\Users\Lospsy\Desktop\Thesis\Results\PINN_hyperparameter_tuning\bayesian_optimization"
    data_path = r"C:\Users\Lospsy\Desktop\Thesis\Results\cleaned_forecast_data.csv"
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Add time features if missing
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['dayofweek'] = df.index.dayofweek
    
    # Create dataset
    dataset = SolarProductionDataset(df, seq_length=24, forecast_horizon=1, normalize=True)
    print(f"Dataset created: {len(dataset)} samples")
    
    # Initialize optimizer
    optimizer = BayesianHyperparameterOptimizer(grid_results_path, output_dir)
    
    # Number of Bayesian iterations
    n_iterations = 30  # Adjust as needed
    
    print(f"\nStarting Bayesian optimization for {n_iterations} iterations...")
    print("This should find better configurations much faster than grid search!")
    
    for i in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"BAYESIAN ITERATION {i+1}/{n_iterations}")
        print(f"{'='*60}")
        
        # Get suggestion
        if i < 5:
            # Use EI for initial exploration
            params, x_array = optimizer.suggest_next_params(method='ei')
        else:
            # Switch to UCB for more exploitation
            params, x_array = optimizer.suggest_next_params(method='ucb')
        
        print(f"\nTesting configuration:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        try:
            # Train and evaluate
            results = train_and_evaluate_config(params, dataset, optimizer.device)
            
            print(f"\nResults:")
            print(f"  Validation Loss: {results['val_loss']:.6f}")
            print(f"  R² Score: {results['r2']:.4f}")
            print(f"  MAE: {results['mae']:.0f} Wh")
            print(f"  Epochs trained: {results['epochs_trained']}")
            
            # Update optimizer
            optimizer.update_with_result(params, results['val_loss'], results['r2'])
            
        except Exception as e:
            print(f"Error evaluating configuration: {e}")
            # Add a penalty for failed configurations
            optimizer.update_with_result(params, 1.0, 0.0)
        
        # Plot progress every 5 iterations
        if (i + 1) % 5 == 0 or i == n_iterations - 1:
            optimizer.plot_optimization_progress()
            print(f"\nProgress plots saved to: {output_dir}")
    
    # Final summary
    print("\n" + "="*80)
    print("BAYESIAN OPTIMIZATION COMPLETE!")
    print("="*80)
    print(f"Total configurations tested: {len(optimizer.X_observed)}")
    print(f"  - From grid search: {len(optimizer.grid_data['results'])}")
    print(f"  - From Bayesian: {n_iterations}")
    print(f"\nBest configuration found:")
    for k, v in optimizer.best_config.items():
        print(f"  {k}: {v}")
    print(f"\nBest validation loss: {-optimizer.best_score:.6f}")
    print(f"Best R² score: {optimizer.best_r2:.4f}")
    
    # Compare with grid search
    grid_best_loss = optimizer.grid_data['best_val_loss']
    improvement = (grid_best_loss - (-optimizer.best_score)) / grid_best_loss * 100
    
    if improvement > 0:
        print(f"\n🎉 Improvement over grid search: {improvement:.1f}%")
    else:
        print(f"\n Grid search best is still optimal")
    
    print("="*80)


if __name__ == "__main__":
    main()