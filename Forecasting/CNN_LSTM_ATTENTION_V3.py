import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.optim.lr_scheduler import _LRScheduler
import math
from torch.cuda.amp import autocast, GradScaler
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import sys
import warnings
import requests
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATASET CLASS (ESSENTIAL)
# ============================================================================

class SolarProductionDataset(Dataset):
    def __init__(self, dataframe, seq_length=24, forecast_horizon=1, normalize=True):
        self.df = dataframe.copy()
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        
        # Automatically detect feature columns (exclude target)
        exclude_cols = ['E_ac']
        feature_columns = [col for col in self.df.columns if col not in exclude_cols]
        
        print(f"📊 Using {len(feature_columns)} features for training")
        
        # ============= FIX: ENSURE ALL DATA IS NUMERIC =============
        print("🔧 Converting all features to numeric...")
        
        # Convert features to numeric, coercing errors to NaN
        for col in feature_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Convert target to numeric
        self.df['E_ac'] = pd.to_numeric(self.df['E_ac'], errors='coerce')
        
        # Fill any remaining NaN values
        self.df = self.df.fillna(0)
        
        # Check data types
        non_numeric_cols = []
        for col in feature_columns:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            print(f"⚠️ Non-numeric columns found: {non_numeric_cols}")
            # Force convert to float
            for col in non_numeric_cols:
                self.df[col] = self.df[col].astype(float, errors='ignore')
        
        # Ensure we have numeric arrays
        self.features = self.df[feature_columns].values.astype(np.float32)
        self.targets = self.df['E_ac'].values.astype(np.float32)
        self.feature_names = feature_columns
        
        print(f"✅ Features shape: {self.features.shape}, dtype: {self.features.dtype}")
        print(f"✅ Targets shape: {self.targets.shape}, dtype: {self.targets.dtype}")
        
        if normalize:
            self._normalize_features()
        
        self.valid_indices = len(self.df) - seq_length - forecast_horizon + 1
        
        if self.valid_indices <= 0:
            raise ValueError(f"Not enough data points. Need at least {seq_length + forecast_horizon} points, got {len(self.df)}")
        
    def _normalize_features(self):
        # Ensure float32 for all operations
        self.features = self.features.astype(np.float32)
        self.targets = self.targets.astype(np.float32)
        
        self.feature_mins = np.min(self.features, axis=0).astype(np.float32)
        self.feature_maxs = np.max(self.features, axis=0).astype(np.float32)
        
        range_vals = self.feature_maxs - self.feature_mins
        range_vals[range_vals == 0] = 1
        
        self.features = (self.features - self.feature_mins) / range_vals
        
        self.target_max = np.max(self.targets).astype(np.float32)
        self.target_min = np.min(self.targets).astype(np.float32)
        if self.target_max > self.target_min:
            self.targets_normalized = ((self.targets - self.target_min) / (self.target_max - self.target_min)).astype(np.float32)
        else:
            self.targets_normalized = self.targets.astype(np.float32)
    
    def __len__(self):
        return self.valid_indices
    
    def __getitem__(self, idx):
        features = self.features[idx:idx + self.seq_length].astype(np.float32)
        targets = self.targets_normalized[idx + self.seq_length:idx + self.seq_length + self.forecast_horizon].astype(np.float32)
        
        # Double-check data types before tensor conversion
        if features.dtype != np.float32:
            features = features.astype(np.float32)
        if targets.dtype != np.float32:
            targets = targets.astype(np.float32)
            
        return torch.FloatTensor(features), torch.FloatTensor(targets)
    
    def denormalize_targets(self, normalized_values):
        if hasattr(self, 'target_max') and hasattr(self, 'target_min'):
            return normalized_values * (self.target_max - self.target_min) + self.target_min
        return normalized_values

# ============================================================================
# 2. ATTENTION MECHANISMS (ESSENTIAL FOR SOTA)
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism for time series"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, hidden_size = x.shape
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.fc_out(context)
        
        return output, attn_weights

class MultiScaleAttentionV2(nn.Module):
    """Multi-scale attention for better temporal modeling"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.local_attention = nn.MultiheadAttention(hidden_size, num_heads//2, dropout=dropout, batch_first=True)
        self.global_attention = nn.MultiheadAttention(hidden_size, num_heads//2, dropout=dropout, batch_first=True)
        
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        local_out, local_weights = self.local_attention(x, x, x)
        
        if x.shape[1] > 12:
            x_down = F.avg_pool1d(x.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
            global_out, _ = self.global_attention(x_down, x_down, x_down)
            global_out = F.interpolate(global_out.transpose(1, 2), size=x.shape[1], mode='linear', align_corners=False).transpose(1, 2)
        else:
            global_out, _ = self.global_attention(x, x, x)
        
        combined = torch.cat([local_out, global_out], dim=-1)
        output = self.fusion(combined)
        output = self.layer_norm(output + x)
        output = self.dropout(output)
        
        return output, local_weights

class SpatialTemporalAttention(nn.Module):
    """Attention that considers both spatial (feature) and temporal dimensions"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.temporal_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        # FIX: Create separate feature attention with correct embedding dimension
        # When we transpose, seq_len becomes the embedding dimension, so we need to handle this
        # Option 1: Use a linear layer to project to correct dimension
        self.feature_projection = nn.Linear(hidden_size, hidden_size)
        
        # Option 2: Use a simpler approach - just use temporal attention
        # and combine with feature-wise operations
        self.feature_norm = nn.LayerNorm(hidden_size)
        self.feature_linear = nn.Linear(hidden_size, hidden_size)
        
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Temporal attention (works fine)
        temporal_out, _ = self.temporal_attention(x, x, x)
        
        # Feature attention - FIXED approach
        # Instead of using MultiheadAttention on transposed features,
        # use feature-wise transformations
        
        # Method 1: Feature-wise linear transformation with attention-like mechanism
        batch_size, seq_len, hidden_size = x.shape
        
        # Apply feature-wise attention manually
        # Compute attention weights across features for each time step
        feature_weights = torch.softmax(
            torch.sum(x * x, dim=1, keepdim=True) / (hidden_size ** 0.5), 
            dim=2
        )  # [batch, 1, hidden_size]
        
        # Apply feature attention
        feature_out = x * feature_weights  # Broadcast across sequence length
        feature_out = self.feature_linear(feature_out)
        feature_out = self.feature_norm(feature_out)
        
        # Combine temporal and feature outputs
        combined = torch.cat([temporal_out, feature_out], dim=-1)
        output = self.fusion(combined)
        output = self.layer_norm(output + x)  # Residual connection
        output = self.dropout(output)
        
        return output

# ============================================================================
# 3. ADVANCED CNN BLOCKS (ESSENTIAL FOR SOTA)
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualCNNBlock(nn.Module):
    """Enhanced CNN block with residual connections and SE attention"""
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.2):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.se = SEBlock(out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class TemporalBlock(nn.Module):
    """Temporal Convolutional Block with residual connection - FIXED VERSION"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        
        # Use standard Conv1d with proper padding to maintain sequence length
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Batch normalization for better training stability
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection - ensure sizes match
        res = x if self.downsample is None else self.downsample(x)
        
        # Make sure sizes match before adding
        if out.size() != res.size():
            # If sizes don't match, adjust to minimum size
            min_size = min(out.size(2), res.size(2))
            out = out[:, :, :min_size]
            res = res[:, :, :min_size]
        
        return self.relu(out + res)

# ============================================================================
# 4. ADVANCED LOSS FUNCTIONS (ESSENTIAL FOR SOTA)
# ============================================================================

class SolarForecastingLoss(nn.Module):
    """Comprehensive loss function for solar energy forecasting"""
    def __init__(self, weights=None, peak_hours=(10, 16)):
        super().__init__()
        self.weights = weights or {
            'mse': 1.0,
            'mae': 0.5,
            'peak_mse': 2.0,
            'smooth': 0.1,
            'directional': 0.3
        }
        self.peak_hours = peak_hours
    
    def forward(self, predictions, targets, timestamps=None):
        total_loss = 0
        loss_components = {}
        
        mse_loss = F.mse_loss(predictions, targets)
        total_loss += self.weights['mse'] * mse_loss
        loss_components['mse'] = mse_loss.item()
        
        mae_loss = F.l1_loss(predictions, targets)
        total_loss += self.weights['mae'] * mae_loss
        loss_components['mae'] = mae_loss.item()
        
        if timestamps is not None:
            peak_mask = ((timestamps >= self.peak_hours[0]) & 
                        (timestamps <= self.peak_hours[1])).float()
            if peak_mask.sum() > 0:
                peak_mse = F.mse_loss(
                    predictions * peak_mask.unsqueeze(1),
                    targets * peak_mask.unsqueeze(1)
                ) / (peak_mask.mean() + 1e-8)
                total_loss += self.weights['peak_mse'] * peak_mse
                loss_components['peak_mse'] = peak_mse.item()
        
        if predictions.shape[1] > 1:
            pred_diff = predictions[:, 1:] - predictions[:, :-1]
            target_diff = targets[:, 1:] - targets[:, :-1]
            smooth_loss = F.mse_loss(pred_diff, target_diff)
            total_loss += self.weights['smooth'] * smooth_loss
            loss_components['smooth'] = smooth_loss.item()
        
        if predictions.shape[1] > 1:
            pred_direction = torch.sign(predictions[:, 1:] - predictions[:, :-1])
            target_direction = torch.sign(targets[:, 1:] - targets[:, :-1])
            directional_loss = 1 - torch.mean((pred_direction == target_direction).float())
            total_loss += self.weights['directional'] * directional_loss
            loss_components['directional'] = directional_loss.item()
        
        return total_loss, loss_components

class FocalMSELoss(nn.Module):
    """Focal MSE Loss - focuses on harder examples"""
    def __init__(self, alpha=2.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        mse = F.mse_loss(predictions, targets, reduction='none')
        relative_error = torch.abs((predictions - targets) / (targets + 1e-8))
        focal_weight = torch.pow(relative_error + 1e-8, self.gamma)
        focal_loss = self.alpha * focal_weight * mse
        return focal_loss.mean()

class QuantileLoss(nn.Module):
    """Quantile Loss for uncertainty estimation"""
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, predictions, targets):
        losses = []
        for i, q in enumerate(self.quantiles):
            error = targets.squeeze(-1) - predictions[:, i]
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss)
        return torch.mean(torch.stack(losses))

# ============================================================================
# 5. ADVANCED SCHEDULERS (ESSENTIAL FOR SOTA)
# ============================================================================

class WarmupCosineScheduler(_LRScheduler):
    """Learning rate scheduler with warmup and cosine annealing"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-7, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * self.last_epoch / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]

class AdamWWithScheduling:
    """Enhanced AdamW with automatic scheduling and monitoring"""
    def __init__(self, model, lr=1e-3, weight_decay=1e-4, 
                 scheduler_type='warmup_cosine', total_epochs=200):
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                              betas=(0.9, 0.999), eps=1e-8)
        
        if scheduler_type == 'warmup_cosine':
            self.scheduler = WarmupCosineScheduler(
                self.optimizer, warmup_epochs=10, total_epochs=total_epochs
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=15, verbose=True
            )
        
        self.scheduler_type = scheduler_type
    
    def step(self, loss=None):
        if self.scheduler_type == 'warmup_cosine':
            self.scheduler.step()
        else:
            self.scheduler.step(loss)
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

# ============================================================================
# 6. ADVANCED TRAINING STRATEGIES (VALUABLE FOR SOTA)
# ============================================================================

class CurriculumLearning:
    """Curriculum learning for time series forecasting"""
    def __init__(self, easy_horizon=1, hard_horizon=24, transition_epochs=50):
        self.easy_horizon = easy_horizon
        self.hard_horizon = hard_horizon
        self.transition_epochs = transition_epochs
        self.current_epoch = 0
    
    def get_current_horizon(self):
        if self.current_epoch < self.transition_epochs:
            progress = self.current_epoch / self.transition_epochs
            current_horizon = int(self.easy_horizon + 
                                (self.hard_horizon - self.easy_horizon) * progress)
            return max(self.easy_horizon, current_horizon)
        return self.hard_horizon
    
    def step_epoch(self):
        self.current_epoch += 1

class KnowledgeDistillation:
    """Knowledge distillation from ensemble to single model"""
    def __init__(self, teacher_models, temperature=3.0, alpha=0.7):
        self.teacher_models = teacher_models
        self.temperature = temperature
        self.alpha = alpha
        
        for teacher in self.teacher_models:
            teacher.eval()
    
    def distillation_loss(self, student_logits, teacher_logits, targets):
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
        
        standard_loss = F.mse_loss(student_logits, targets)
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * standard_loss
        return total_loss, distill_loss, standard_loss
    
    def get_teacher_predictions(self, inputs):
        with torch.no_grad():
            teacher_preds = []
            for teacher in self.teacher_models:
                pred = teacher(inputs)
                teacher_preds.append(pred)
            ensemble_pred = torch.mean(torch.stack(teacher_preds), dim=0)
            return ensemble_pred

class GradientClipping:
    """Enhanced gradient clipping with monitoring"""
    def __init__(self, max_norm=1.0, norm_type=2.0, monitor=True):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.monitor = monitor
        self.grad_norms = []
    
    def clip_gradients(self, model):
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_norm, self.norm_type
        )
        
        if self.monitor:
            self.grad_norms.append(total_norm.item())
        
        return total_norm
    
    def get_average_grad_norm(self, last_n=100):
        if len(self.grad_norms) == 0:
            return 0.0
        return np.mean(self.grad_norms[-last_n:])

# ============================================================================
# 7. ENSEMBLE METHODS (VALUABLE FOR SOTA)
# ============================================================================

class BayesianEnsemble:
    """Bayesian ensemble for uncertainty quantification"""
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else [1/len(models)] * len(models)
    
    def predict(self, x, return_uncertainty=True):
        predictions = []
        uncertainties = []
        
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'use_uncertainty') and model.use_uncertainty:
                    pred, unc = model(x)
                else:
                    pred = model(x)
                    unc = torch.zeros_like(pred)
                
                predictions.append(pred * weight)
                uncertainties.append(unc * weight)
        
        ensemble_pred = torch.sum(torch.stack(predictions), dim=0)
        
        if return_uncertainty:
            aleatoric = torch.sum(torch.stack(uncertainties), dim=0)
            epistemic = torch.var(torch.stack([p/w for p, w in zip(predictions, self.weights)]), dim=0)
            total_uncertainty = aleatoric + epistemic
            return ensemble_pred, total_uncertainty
        
        return ensemble_pred

# ============================================================================
# 8. STATE-OF-THE-ART MODEL ARCHITECTURES
# ============================================================================

class CNNLSTMAttention(nn.Module):
    """Advanced CNN-LSTM-Attention model for solar energy forecasting"""
    def __init__(self, input_size, seq_length=24, forecast_horizon=1,
                 cnn_filters=[64, 128, 256], kernel_sizes=[3, 5, 7],
                 lstm_hidden=256, lstm_layers=3, num_heads=8,
                 dropout=0.2, use_batch_norm=True):
        super().__init__()
        
        self.input_size = input_size
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.use_batch_norm = use_batch_norm
        
        # CNN Feature Extraction Layers
        self.cnn_layers = nn.ModuleList()
        in_channels = input_size
        
        for i, (filters, kernel_size) in enumerate(zip(cnn_filters, kernel_sizes)):
            padding = kernel_size // 2
            conv = nn.Conv1d(in_channels, filters, kernel_size, padding=padding)
            self.cnn_layers.append(conv)
            
            if use_batch_norm:
                self.cnn_layers.append(nn.BatchNorm1d(filters))
            
            self.cnn_layers.append(nn.ReLU())
            self.cnn_layers.append(nn.Dropout(dropout))
            
            if i < len(cnn_filters) - 1:
                dilation = 2 ** i
                temporal_block = TemporalBlock(filters, filters, kernel_size, 
                                             stride=1, dilation=dilation, 
                                             padding=(kernel_size-1)*dilation//2, 
                                             dropout=dropout)
                self.cnn_layers.append(temporal_block)
            
            in_channels = filters
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1] + cnn_filters[-1] * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Multi-Head Attention
        self.attention = MultiHeadAttention(
            hidden_size=lstm_hidden * 2,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feature fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU()
        )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_hidden // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, forecast_horizon)
        )
        
        self.skip_connection = nn.Linear(input_size * seq_length, forecast_horizon)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(lstm_hidden * 2)
        self.layer_norm2 = nn.LayerNorm(lstm_hidden // 2)
        
    def forward(self, x, return_attention=False):
        batch_size, seq_length, input_size = x.shape
        
        skip_input = x.reshape(batch_size, -1)
        skip_output = self.skip_connection(skip_input)
        
        # CNN Feature Extraction
        x_cnn = x.transpose(1, 2)
        
        for layer in self.cnn_layers:
            x_cnn = layer(x_cnn)
        
        avg_pool = self.global_avg_pool(x_cnn).squeeze(-1)
        max_pool = self.global_max_pool(x_cnn).squeeze(-1)
        global_features = torch.cat([avg_pool, max_pool], dim=1)
        
        x_cnn = x_cnn.transpose(1, 2)
        global_features_expanded = global_features.unsqueeze(1).expand(-1, seq_length, -1)
        x_combined = torch.cat([x_cnn, global_features_expanded], dim=2)
        
        # Bidirectional LSTM
        lstm_out, (hidden, cell) = self.lstm(x_combined)
        lstm_out = self.layer_norm1(lstm_out)
        
        # Multi-Head Attention
        attn_out, attn_weights = self.attention(lstm_out)
        attn_out = attn_out + lstm_out
        
        # Temporal aggregation
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        last_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        
        attn_avg = torch.mean(attn_out, dim=1)
        combined = (last_hidden + attn_avg) / 2
        
        # Feature fusion
        fused = self.fusion_layer(combined)
        fused = self.layer_norm2(fused)
        
        # Final prediction with skip connection
        output = self.output_layer(fused)
        output = output + 0.1 * skip_output
        output = F.relu(output)
        
        if return_attention:
            return output, attn_weights
        return output

class CNNLSTMAttentionWithUncertainty(CNNLSTMAttention):
    """Extended version with uncertainty estimation"""
    def __init__(self, *args, **kwargs):
        # Extract uncertainty-specific parameter before passing to parent
        use_uncertainty = kwargs.pop('use_uncertainty', True)  # Remove from kwargs
        
        # Now call parent with remaining kwargs
        super().__init__(*args, **kwargs)
        
        lstm_hidden = kwargs.get('lstm_hidden', 256)
        forecast_horizon = kwargs.get('forecast_horizon', 1)
        dropout = kwargs.get('dropout', 0.2)
        
        # Add uncertainty layer
        self.uncertainty_layer = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, forecast_horizon),
            nn.Softplus()
        )
        
        # Store the flag
        self.use_uncertainty = use_uncertainty
    
    def forward(self, x, return_attention=False):
        batch_size, seq_length, input_size = x.shape
        
        skip_input = x.reshape(batch_size, -1)
        skip_output = self.skip_connection(skip_input)
        
        x_cnn = x.transpose(1, 2)
        for layer in self.cnn_layers:
            x_cnn = layer(x_cnn)
        
        avg_pool = self.global_avg_pool(x_cnn).squeeze(-1)
        max_pool = self.global_max_pool(x_cnn).squeeze(-1)
        global_features = torch.cat([avg_pool, max_pool], dim=1)
        
        x_cnn = x_cnn.transpose(1, 2)
        global_features_expanded = global_features.unsqueeze(1).expand(-1, seq_length, -1)
        x_combined = torch.cat([x_cnn, global_features_expanded], dim=2)
        
        lstm_out, (hidden, cell) = self.lstm(x_combined)
        lstm_out = self.layer_norm1(lstm_out)
        
        attn_out, attn_weights = self.attention(lstm_out)
        attn_out = attn_out + lstm_out
        
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        last_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        
        attn_avg = torch.mean(attn_out, dim=1)
        combined = (last_hidden + attn_avg) / 2
        
        fused = self.fusion_layer(combined)
        fused = self.layer_norm2(fused)
        
        predictions = self.output_layer(fused)
        predictions = predictions + 0.1 * skip_output
        predictions = F.relu(predictions)
        
        uncertainty = self.uncertainty_layer(combined)
        
        if return_attention:
            return predictions, uncertainty, attn_weights
        return predictions, uncertainty

class StateOfTheArtSolarModel(nn.Module):
    """Ultimate state-of-the-art model with all advanced features"""
    def __init__(self, input_size, seq_length=24, forecast_horizon=1,
             cnn_filters=[64, 128, 256, 512], kernel_sizes=[3, 5, 7, 9],
             lstm_hidden=512, lstm_layers=3, num_heads=16,
             dropout=0.2, use_uncertainty=True):
        super().__init__()
        
        self.input_size = input_size
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.use_uncertainty = use_uncertainty
        
        # SIMPLIFIED CNN with ResidualBlocks - might need this fix too
        self.cnn_blocks = nn.ModuleList()
        in_channels = input_size
    
        for i, (filters, kernel_size) in enumerate(zip(cnn_filters, kernel_sizes)):
            # Use a simpler CNN block instead of ResidualCNNBlock if there are issues
            simple_block = nn.Sequential(
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(filters, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters),
                nn.ReLU()
            )
            self.cnn_blocks.append(simple_block)
            in_channels = filters
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Enhanced LSTM with Residual Connections
        self.lstm_layers = nn.ModuleList()
        lstm_input_size = cnn_filters[-1] + cnn_filters[-1] * 2
        
        for i in range(lstm_layers):
            self.lstm_layers.append(nn.LSTM(
                input_size=lstm_input_size if i == 0 else lstm_hidden * 2,
                hidden_size=lstm_hidden,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if i < lstm_layers - 1 else 0
            ))
        
        self.lstm_layer_norms = nn.ModuleList([
            nn.LayerNorm(lstm_hidden * 2) for _ in range(lstm_layers)
        ])
        
        # Advanced Attention Mechanisms
        self.multi_scale_attention = MultiScaleAttentionV2(
            lstm_hidden * 2, num_heads=num_heads, dropout=dropout
        )
        
        self.spatial_temporal_attention = SpatialTemporalAttention(
            lstm_hidden * 2, num_heads=num_heads//2, dropout=dropout
        )
        
        # Feature Fusion
        fusion_input_size = lstm_hidden * 2
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_size, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.LayerNorm(lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(lstm_hidden // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, forecast_horizon)
        )
        
        # Uncertainty head
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(lstm_hidden // 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, forecast_horizon),
                nn.Softplus()
            )
        
        # Multi-scale skip connections
        self.skip_connections = nn.ModuleList([
            nn.Linear(input_size * seq_length, forecast_horizon),
            nn.Linear(cnn_filters[-1], forecast_horizon),
            nn.Linear(lstm_hidden * 2, forecast_horizon)
        ])
        
    def forward(self, x, return_attention=False):
        batch_size, seq_length, input_size = x.shape
        
        skip_direct = x.reshape(batch_size, -1)
        
        # Enhanced CNN
        x_cnn = x.transpose(1, 2)
        
        for cnn_block in self.cnn_blocks:
            x_cnn = cnn_block(x_cnn)
        
        avg_pool = self.global_avg_pool(x_cnn).squeeze(-1)
        max_pool = self.global_max_pool(x_cnn).squeeze(-1)
        global_features = torch.cat([avg_pool, max_pool], dim=1)
        skip_cnn = avg_pool
        
        x_cnn = x_cnn.transpose(1, 2)
        global_features_expanded = global_features.unsqueeze(1).expand(-1, seq_length, -1)
        lstm_input = torch.cat([x_cnn, global_features_expanded], dim=2)
        
        # Enhanced LSTM
        hidden_state = lstm_input
        
        for i, (lstm_layer, layer_norm) in enumerate(zip(self.lstm_layers, self.lstm_layer_norms)):
            lstm_out, _ = lstm_layer(hidden_state)
            lstm_out = layer_norm(lstm_out)
            
            if i > 0 and lstm_out.shape == hidden_state.shape:
                lstm_out = lstm_out + hidden_state
            
            hidden_state = lstm_out
        
        lstm_final = hidden_state
        skip_lstm = lstm_final.mean(dim=1)
        
        # Advanced Attention
        attended_multi = self.multi_scale_attention(lstm_final)[0]
        attended_spatial = self.spatial_temporal_attention(attended_multi)
        
        attention_output = attended_spatial + lstm_final
        
        attention_weights = torch.softmax(torch.sum(attention_output, dim=2), dim=1)
        aggregated = torch.sum(attention_output * attention_weights.unsqueeze(2), dim=1)
        
        # Feature Fusion
        fused_features = self.feature_fusion(aggregated)
        
        # Predictions with skip connections
        main_prediction = self.prediction_head(fused_features)
        
        skip_1 = self.skip_connections[0](skip_direct)
        skip_2 = self.skip_connections[1](skip_cnn)
        skip_3 = self.skip_connections[2](skip_lstm)
        
        final_prediction = main_prediction + 0.1*skip_1 + 0.05*skip_2 + 0.05*skip_3
        final_prediction = F.relu(final_prediction)
        
        if self.use_uncertainty:
            uncertainty = self.uncertainty_head(fused_features)
            if return_attention:
                return final_prediction, uncertainty, attention_weights
            return final_prediction, uncertainty
        else:
            if return_attention:
                return final_prediction, attention_weights
            return final_prediction

# ============================================================================
# 9. WEATHER INTEGRATION (VALUABLE - YOU WERE RIGHT!)
# ============================================================================

def fetch_weather_forecast(lat, lon, api_key):
    """Fetch weather forecast data from OpenWeatherMap API"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Weather API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Weather API request failed: {e}")
        return None

def process_forecast_data(forecast_json):
    """Process raw forecast JSON into a pandas DataFrame"""
    forecast_data = []
    
    for item in forecast_json['list']:
        timestamp = datetime.fromtimestamp(item['dt'])
        
        forecast_data.append({
            'datetime': timestamp,
            'forecast_temp': item['main']['temp'],
            'forecast_humidity': item['main']['humidity'],
            'forecast_pressure': item['main']['pressure'],
            'forecast_clouds': item['clouds']['all'],
            'forecast_wind_speed': item['wind']['speed'],
            'forecast_wind_direction': item['wind'].get('deg', 0),
            'forecast_precipitation': item.get('rain', {}).get('3h', 0),
            'forecast_weather_code': item['weather'][0]['id'],
            'forecast_weather_main': item['weather'][0]['main'],
        })
    
    df_forecast = pd.DataFrame(forecast_data)
    df_forecast.set_index('datetime', inplace=True)
    
    return df_forecast

def engineer_weather_features(df, df_forecast):
    """Merge historical data with forecast and engineer additional features"""
    df_merged = df.copy()
    
    # Add forecast features
    for col in df_forecast.columns:
        df_merged[col] = np.nan
    
    common_dates = df_merged.index.intersection(df_forecast.index)
    for col in df_forecast.columns:
        df_merged.loc[common_dates, col] = df_forecast.loc[common_dates, col]
    
    # Engineer weather interaction features
    if 'forecast_clouds' in df_merged.columns:
        df_merged['clear_sky_ratio'] = 1 - (df_merged['forecast_clouds'] / 100)
    
    if 'forecast_weather_code' in df_merged.columns:
        weather_categories = {
            'clear': (df_merged['forecast_weather_code'] == 800),
            'partly_cloudy': ((df_merged['forecast_weather_code'] >= 801) & (df_merged['forecast_weather_code'] <= 803)),
            'cloudy': (df_merged['forecast_weather_code'] == 804),
            'precipitation': ((df_merged['forecast_weather_code'] >= 300) & (df_merged['forecast_weather_code'] <= 599)),
            'snow': ((df_merged['forecast_weather_code'] >= 600) & (df_merged['forecast_weather_code'] <= 699)),
            'thunderstorm': ((df_merged['forecast_weather_code'] >= 200) & (df_merged['forecast_weather_code'] <= 299))
        }
        
        for category, condition in weather_categories.items():
            df_merged[f'weather_{category}'] = condition.astype(int)
    
    if 'forecast_clouds' in df_merged.columns and 'forecast_precipitation' in df_merged.columns:
        cloud_factor = 1 - (df_merged['forecast_clouds'] / 100) * 0.8
        precip_factor = 1 - np.minimum(df_merged['forecast_precipitation'] * 0.5, 0.9)
        df_merged['irradiance_factor'] = cloud_factor * precip_factor
    
    return df_merged

# ============================================================================
# 10. ENHANCED FEATURE ENGINEERING (ESSENTIAL)
# ============================================================================

def create_enhanced_features_v2(df):
    """Enhanced feature engineering - 40+ features for SOTA performance"""
    df_enhanced = df.copy()
    
    print("🔧 Creating enhanced features...")
    print(f"Index type: {type(df_enhanced.index)}")
    print(f"Is DatetimeIndex: {isinstance(df_enhanced.index, pd.DatetimeIndex)}")
    
    # Check if we have datetime index
    has_datetime_index = isinstance(df_enhanced.index, pd.DatetimeIndex)
    
    # If index contains datetime strings but isn't DatetimeIndex, try to convert
    if not has_datetime_index and len(df_enhanced.index) > 0:
        try:
            # Check if index contains datetime-like strings
            sample_val = str(df_enhanced.index[0])
            if any(char in sample_val for char in ['-', ':', 'T', '+']):
                print("Attempting to convert index to DatetimeIndex...")
                dt_index = pd.to_datetime(df_enhanced.index, errors='coerce')
                if not dt_index.isna().all():
                    if dt_index.dt.tz is not None:
                        dt_index = dt_index.dt.tz_localize(None)
                    df_enhanced.index = pd.DatetimeIndex(dt_index)
                    has_datetime_index = True
                    print("✅ Successfully converted to DatetimeIndex")
        except Exception as e:
            print(f"⚠️ Could not convert index to DatetimeIndex: {e}")
    
    if has_datetime_index:
        print("Creating features from DatetimeIndex...")
        # Cyclical temporal features from datetime index
        df_enhanced['hour_sin'] = np.sin(2 * np.pi * df_enhanced.index.hour / 24).astype(np.float32)
        df_enhanced['hour_cos'] = np.cos(2 * np.pi * df_enhanced.index.hour / 24).astype(np.float32)
        df_enhanced['day_sin'] = np.sin(2 * np.pi * df_enhanced.index.dayofyear / 365.25).astype(np.float32)
        df_enhanced['day_cos'] = np.cos(2 * np.pi * df_enhanced.index.dayofyear / 365.25).astype(np.float32)
        df_enhanced['month_sin'] = np.sin(2 * np.pi * df_enhanced.index.month / 12).astype(np.float32)
        df_enhanced['month_cos'] = np.cos(2 * np.pi * df_enhanced.index.month / 12).astype(np.float32)
        
        try:
            df_enhanced['week_sin'] = np.sin(2 * np.pi * df_enhanced.index.isocalendar().week / 52).astype(np.float32)
            df_enhanced['week_cos'] = np.cos(2 * np.pi * df_enhanced.index.isocalendar().week / 52).astype(np.float32)
        except:
            # Fallback if isocalendar fails
            df_enhanced['week_sin'] = np.sin(2 * np.pi * (df_enhanced.index.dayofyear // 7) / 52).astype(np.float32)
            df_enhanced['week_cos'] = np.cos(2 * np.pi * (df_enhanced.index.dayofyear // 7) / 52).astype(np.float32)
        
        print("✅ DatetimeIndex features created")
    else:
        print("Creating features from existing columns or approximations...")
        # Create cyclical features from existing hour/month/dayofweek columns
        if 'hour' in df_enhanced.columns:
            df_enhanced['hour_sin'] = np.sin(2 * np.pi * pd.to_numeric(df_enhanced['hour'], errors='coerce') / 24).astype(np.float32)
            df_enhanced['hour_cos'] = np.cos(2 * np.pi * pd.to_numeric(df_enhanced['hour'], errors='coerce') / 24).astype(np.float32)
        
        if 'month' in df_enhanced.columns:
            df_enhanced['month_sin'] = np.sin(2 * np.pi * pd.to_numeric(df_enhanced['month'], errors='coerce') / 12).astype(np.float32)
            df_enhanced['month_cos'] = np.cos(2 * np.pi * pd.to_numeric(df_enhanced['month'], errors='coerce') / 12).astype(np.float32)
        
        if 'dayofweek' in df_enhanced.columns:
            df_enhanced['dayofweek_sin'] = np.sin(2 * np.pi * pd.to_numeric(df_enhanced['dayofweek'], errors='coerce') / 7).astype(np.float32)
            df_enhanced['dayofweek_cos'] = np.cos(2 * np.pi * pd.to_numeric(df_enhanced['dayofweek'], errors='coerce') / 7).astype(np.float32)
        
        # Create approximate day of year and week features using position
        n_samples = len(df_enhanced)
        hours_elapsed = np.arange(n_samples).astype(np.float32)
        day_of_year_approx = (hours_elapsed // 24) % 365
        df_enhanced['day_sin'] = np.sin(2 * np.pi * day_of_year_approx / 365.25).astype(np.float32)
        df_enhanced['day_cos'] = np.cos(2 * np.pi * day_of_year_approx / 365.25).astype(np.float32)
        
        # Week approximation
        week_approx = (day_of_year_approx // 7) % 52
        df_enhanced['week_sin'] = np.sin(2 * np.pi * week_approx / 52).astype(np.float32)
        df_enhanced['week_cos'] = np.cos(2 * np.pi * week_approx / 52).astype(np.float32)
        
        print("✅ Approximated temporal features created")
    
    # Solar physics features
    if 'zenith' in df_enhanced.columns:
        zenith_numeric = pd.to_numeric(df_enhanced['zenith'], errors='coerce').fillna(90)
        df_enhanced['solar_elevation'] = (90 - zenith_numeric).astype(np.float32)
        df_enhanced['sun_up'] = (df_enhanced['solar_elevation'] > 0).astype(np.float32)
        df_enhanced['zenith_sin'] = np.sin(np.radians(zenith_numeric)).astype(np.float32)
        df_enhanced['zenith_cos'] = np.cos(np.radians(zenith_numeric)).astype(np.float32)
        print("✅ Solar physics features created")
    
    # Weather interaction features
    if all(col in df_enhanced.columns for col in ['SolRad_Hor', 'Air Temp']):
        solrad = pd.to_numeric(df_enhanced['SolRad_Hor'], errors='coerce').fillna(0)
        airtemp = pd.to_numeric(df_enhanced['Air Temp'], errors='coerce').fillna(25)
        
        df_enhanced['temp_efficiency'] = (1 - 0.004 * np.maximum(0, airtemp - 25)).astype(np.float32)
        df_enhanced['irradiance_temp_adjusted'] = (solrad * df_enhanced['temp_efficiency']).astype(np.float32)
        
        if 'SolRad_Dif' in df_enhanced.columns:
            solrad_dif = pd.to_numeric(df_enhanced['SolRad_Dif'], errors='coerce').fillna(0)
            df_enhanced['direct_normal'] = (solrad - solrad_dif).astype(np.float32)
            df_enhanced['clearness_index'] = (solrad / (solrad + solrad_dif + 1e-6)).astype(np.float32)
        
        print("✅ Weather interaction features created")
    
    # Lag features
    important_cols = ['E_ac', 'SolRad_Hor', 'Air Temp']
    for col in important_cols:
        if col in df_enhanced.columns:
            col_numeric = pd.to_numeric(df_enhanced[col], errors='coerce')
            for lag in [1, 2, 3, 6, 12, 24]:
                df_enhanced[f'{col}_lag_{lag}h'] = col_numeric.shift(lag).astype(np.float32)
    print("✅ Lag features created")
    
    # Rolling statistics
    for col in important_cols:
        if col in df_enhanced.columns:
            col_numeric = pd.to_numeric(df_enhanced[col], errors='coerce')
            for window in [3, 6, 12, 24]:
                df_enhanced[f'{col}_roll_mean_{window}h'] = col_numeric.rolling(window, min_periods=1).mean().astype(np.float32)
                df_enhanced[f'{col}_roll_std_{window}h'] = col_numeric.rolling(window, min_periods=1).std().fillna(0).astype(np.float32)
    print("✅ Rolling statistics features created")
    
    # Binary indicators
    if has_datetime_index:
        df_enhanced['is_weekend'] = (df_enhanced.index.dayofweek >= 5).astype(np.float32)
        df_enhanced['is_peak_hour'] = ((df_enhanced.index.hour >= 10) & (df_enhanced.index.hour <= 16)).astype(np.float32)
    else:
        if 'dayofweek' in df_enhanced.columns:
            df_enhanced['is_weekend'] = (pd.to_numeric(df_enhanced['dayofweek'], errors='coerce') >= 5).astype(np.float32)
        if 'hour' in df_enhanced.columns:
            hour_numeric = pd.to_numeric(df_enhanced['hour'], errors='coerce')
            df_enhanced['is_peak_hour'] = ((hour_numeric >= 10) & (hour_numeric <= 16)).astype(np.float32)
    print("✅ Binary indicator features created")
    
    # ============= FINAL DATA TYPE CLEANUP =============
    print("🔧 Final data type cleanup...")
    
    # Convert ALL columns to numeric
    for col in df_enhanced.columns:
        if col != df_enhanced.index.name:  # Skip index column if it has a name
            df_enhanced[col] = pd.to_numeric(df_enhanced[col], errors='coerce')
    
    # Fill NaN values
    initial_na = df_enhanced.isna().sum().sum()
    df_enhanced = df_enhanced.fillna(method='ffill').fillna(method='bfill').fillna(0)
    final_na = df_enhanced.isna().sum().sum()
    
    # Final dtype check
    object_cols = df_enhanced.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"⚠️ Converting remaining object columns to float: {list(object_cols)}")
        for col in object_cols:
            df_enhanced[col] = pd.to_numeric(df_enhanced[col], errors='coerce').fillna(0).astype(np.float32)
    
    print(f"✅ Feature engineering complete!")
    print(f"   Final shape: {df_enhanced.shape}")
    print(f"   Features created: {df_enhanced.shape[1]} total")
    print(f"   NaN values filled: {initial_na} → {final_na}")
    print(f"   Data types: {df_enhanced.dtypes.value_counts().to_dict()}")
    
    return df_enhanced

# ============================================================================
# 11. ADVANCED TRAINING FUNCTION
# ============================================================================

def train_state_of_art_model_advanced(model, train_loader, val_loader, epochs=200, lr=0.001, 
                                     device='cuda', save_path=None, patience=30,
                                     use_curriculum=False, use_distillation=False,
                                     teacher_models=None):
    """Ultimate training function with all advanced techniques"""
    model = model.to(device)
    
    # Enhanced optimizer
    optimizer_wrapper = AdamWWithScheduling(
        model, lr=lr, weight_decay=1e-4,
        scheduler_type='warmup_cosine', total_epochs=epochs
    )
    
    # Advanced loss function
    criterion = SolarForecastingLoss(weights={
        'mse': 1.0, 'mae': 0.3, 'peak_mse': 2.0,
        'smooth': 0.1, 'directional': 0.2
    })
    
    # Advanced components
    grad_clipper = GradientClipping(max_norm=1.0, monitor=True)
    curriculum = CurriculumLearning(easy_horizon=1, hard_horizon=24, transition_epochs=50) if use_curriculum else None
    distillation = KnowledgeDistillation(teacher_models) if use_distillation and teacher_models else None
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    history = {
        'train_loss': [], 'val_loss': [], 'learning_rate': [],
        'train_mae': [], 'val_mae': [], 'grad_norms': []
    }
    
    best_val_loss = float('inf')
    no_improve = 0
    
    print(f"🚀 Advanced training starting on {device}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🔧 Curriculum learning: {'Enabled' if use_curriculum else 'Disabled'}")
    print(f"👨‍🏫 Knowledge distillation: {'Enabled' if use_distillation else 'Disabled'}")
    
    for epoch in range(epochs):
        if curriculum:
            curriculum.step_epoch()
            current_horizon = curriculum.get_current_horizon()
        else:
            current_horizon = None
        
        # Training phase
        model.train()
        train_loss = 0
        train_mae = 0
        train_components = {}
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)
            
            # Curriculum learning
            if current_horizon and targets.shape[1] > current_horizon:
                targets = targets[:, :current_horizon]
            
            # Data augmentation
            if torch.rand(1) < 0.3:
                noise_std = 0.01 * torch.std(features, dim=(1, 2), keepdim=True)
                noise = torch.randn_like(features) * noise_std
                features = features + noise
                
                scale = 1 + 0.05 * (torch.rand(1) - 0.5)
                features = features * scale.to(device)
            
            optimizer_wrapper.zero_grad()
            
            if scaler:
                with autocast():
                    if hasattr(model, 'use_uncertainty') and model.use_uncertainty:
                        predictions, uncertainty = model(features)
                    else:
                        predictions = model(features)
                    
                    if targets.dim() == 1:
                        targets = targets.unsqueeze(1)
                    if predictions.shape[1] > targets.shape[1]:
                        predictions = predictions[:, :targets.shape[1]]
                    
                    # Knowledge distillation
                    if distillation:
                        teacher_preds = distillation.get_teacher_predictions(features)
                        loss, distill_loss, standard_loss = distillation.distillation_loss(
                            predictions, teacher_preds, targets
                        )
                    else:
                        timestamps = features[:, -1, -1] if features.shape[2] > 10 else None
                        loss, loss_components = criterion(predictions, targets, timestamps)
                        for k, v in loss_components.items():
                            train_components[k] = train_components.get(k, 0) + v
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer_wrapper.optimizer)
                grad_norm = grad_clipper.clip_gradients(model)
                scaler.step(optimizer_wrapper.optimizer)
                scaler.update()
            else:
                if hasattr(model, 'use_uncertainty') and model.use_uncertainty:
                    predictions, uncertainty = model(features)
                else:
                    predictions = model(features)
                
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                if predictions.shape[1] > targets.shape[1]:
                    predictions = predictions[:, :targets.shape[1]]
                
                if distillation:
                    teacher_preds = distillation.get_teacher_predictions(features)
                    loss, distill_loss, standard_loss = distillation.distillation_loss(
                        predictions, teacher_preds, targets
                    )
                else:
                    timestamps = features[:, -1, -1] if features.shape[2] > 10 else None
                    loss, loss_components = criterion(predictions, targets, timestamps)
                    for k, v in loss_components.items():
                        train_components[k] = train_components.get(k, 0) + v
                
                loss.backward()
                grad_norm = grad_clipper.clip_gradients(model)
                optimizer_wrapper.optimizer.step()
            
            train_loss += loss.item()
            with torch.no_grad():
                mae = torch.mean(torch.abs(predictions - targets))
                train_mae += mae.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                
                if hasattr(model, 'use_uncertainty') and model.use_uncertainty:
                    predictions, uncertainty = model(features)
                else:
                    predictions = model(features)
                
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                if predictions.shape[1] > targets.shape[1]:
                    predictions = predictions[:, :targets.shape[1]]
                
                timestamps = features[:, -1, -1] if features.shape[2] > 10 else None
                loss, _ = criterion(predictions, targets, timestamps)
                mae = torch.mean(torch.abs(predictions - targets))
                
                val_loss += loss.item()
                val_mae += mae.item()
        
        # Update learning rate
        optimizer_wrapper.step()
        current_lr = optimizer_wrapper.get_lr()
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_mae = train_mae / len(train_loader)
        avg_val_mae = val_mae / len(val_loader)
        avg_grad_norm = grad_clipper.get_average_grad_norm(last_n=50)
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(current_lr)
        history['train_mae'].append(avg_train_mae)
        history['val_mae'].append(avg_val_mae)
        history['grad_norms'].append(avg_grad_norm)
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_wrapper.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'history': history,
                    'model_config': {
                        'input_size': model.input_size,
                        'seq_length': model.seq_length,
                        'forecast_horizon': model.forecast_horizon
                    }
                }, save_path)
                print(f'  ✅ Model saved (val_loss: {best_val_loss:.6f})')
        else:
            no_improve += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1:3d}/{epochs} - '
                  f'Train: {avg_train_loss:.6f} (MAE: {avg_train_mae:.4f}), '
                  f'Val: {avg_val_loss:.6f} (MAE: {avg_val_mae:.4f}), '
                  f'LR: {current_lr:.2e}, GradNorm: {avg_grad_norm:.3f}')
            
            if curriculum:
                print(f'  Current horizon: {current_horizon}')
        
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return model, history

# ============================================================================
# 12. EVALUATION AND VISUALIZATION FUNCTIONS
# ============================================================================

def evaluate_model_comprehensive(model, test_loader, dataset, device='cuda', return_predictions=True):
    """Comprehensive model evaluation with all metrics"""
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            
            if hasattr(model, 'use_uncertainty') and model.use_uncertainty:
                predictions, uncertainty = model(features)
                all_uncertainties.append(uncertainty.cpu().numpy())
            else:
                predictions = model(features)
            
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            if predictions.shape[1] > targets.shape[1]:
                predictions = predictions[:, :targets.shape[1]]
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0).squeeze()
    all_targets = np.concatenate(all_targets, axis=0).squeeze()
    
    # FIX: Check length instead of truthiness for numpy arrays
    if len(all_uncertainties) > 0:  # Changed from: if all_uncertainties:
        all_uncertainties = np.concatenate(all_uncertainties, axis=0).squeeze()
    else:
        all_uncertainties = None
    
    # Denormalize
    if hasattr(dataset, 'denormalize_targets'):
        all_predictions = dataset.denormalize_targets(all_predictions)
        all_targets = dataset.denormalize_targets(all_targets)
        if all_uncertainties is not None:  # Changed from: if all_uncertainties:
            scale = dataset.target_max - dataset.target_min
            all_uncertainties = all_uncertainties * scale
    
    # Calculate comprehensive metrics
    mse = np.mean((all_targets - all_predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_targets - all_predictions))
    
    mask = all_targets > 100
    mape = np.mean(np.abs((all_targets[mask] - all_predictions[mask]) / all_targets[mask])) * 100 if np.any(mask) else float('inf')
    
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Peak hours performance
    peak_mae = peak_rmse = peak_r2 = None
    if len(all_targets) >= 24:
        n_days = len(all_targets) // 24
        if n_days > 0:
            daily_targets = all_targets[:n_days*24].reshape(n_days, 24)
            daily_preds = all_predictions[:n_days*24].reshape(n_days, 24)
            
            peak_targets = daily_targets[:, 10:16].flatten()
            peak_preds = daily_preds[:, 10:16].flatten()
            
            peak_mae = np.mean(np.abs(peak_targets - peak_preds))
            peak_rmse = np.sqrt(np.mean((peak_targets - peak_preds) ** 2))
            
            peak_ss_res = np.sum((peak_targets - peak_preds) ** 2)
            peak_ss_tot = np.sum((peak_targets - np.mean(peak_targets)) ** 2)
            peak_r2 = 1 - (peak_ss_res / peak_ss_tot) if peak_ss_tot > 0 else 0
    
    results = {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2,
        'peak_mae': peak_mae, 'peak_rmse': peak_rmse, 'peak_r2': peak_r2
    }
    
    if return_predictions:
        results['predictions'] = all_predictions
        results['targets'] = all_targets
    
    if all_uncertainties is not None:  # Changed from: if all_uncertainties:
        results['uncertainties'] = all_uncertainties
    
    return results

def create_ensemble_models(base_model_config, dataset, device='cuda', n_models=3):
    """Create ensemble of models with different architectures and seeds"""
    models = []
    
    for i, seed in enumerate([42, 123, 456, 789, 999][:n_models]):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create different model variants with appropriate configs
        if i == 0:
            # StateOfTheArtSolarModel - uses all config parameters
            model = StateOfTheArtSolarModel(**base_model_config)
        elif i == 1:
            # CNNLSTMAttentionWithUncertainty - uses all config parameters
            model = CNNLSTMAttentionWithUncertainty(**base_model_config)
        else:
            # CNNLSTMAttention - needs config without 'use_uncertainty'
            config_for_basic_model = {k: v for k, v in base_model_config.items() 
                                    if k != 'use_uncertainty'}  # Remove problematic parameter
            model = CNNLSTMAttention(**config_for_basic_model)
        
        model.to(device)
        models.append(model)
    
    print(f"✅ Created ensemble of {n_models} models")
    return models

def check_gpu_status():
    """Check GPU status and configuration"""
    print("\n" + "="*60)
    print("GPU/CUDA STATUS CHECK")
    print("="*60)
    
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"CUDA devices: {device_count}")
        
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        
        try:
            x = torch.tensor([1.0], device='cuda')
            print("✅ CUDA test successful")
        except Exception as e:
            print(f"❌ CUDA test failed: {e}")
    
    print("="*60)

# ============================================================================
# 13. UPDATED MAIN FUNCTION WITH ALL COMPONENTS
# ============================================================================
def diagnose_data_types(df):
    """Diagnose data type issues in the dataframe"""
    print("🔍 DATA TYPE DIAGNOSTIC REPORT")
    print("="*50)
    
    print(f"📊 DataFrame shape: {df.shape}")
    print(f"📋 Total columns: {len(df.columns)}")
    
    # Check data types
    dtype_counts = df.dtypes.value_counts()
    print(f"\n📈 Data type distribution:")
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} columns")
    
    # Find object columns
    object_cols = df.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"\n⚠️ Object columns found: {len(object_cols)}")
        for col in object_cols[:10]:  # Show first 10
            unique_vals = df[col].unique()[:5]  # Show first 5 unique values
            print(f"   {col}: {unique_vals}")
        if len(object_cols) > 10:
            print(f"   ... and {len(object_cols) - 10} more")
    
    # Check for mixed types
    print(f"\n🔍 Checking for mixed data types...")
    mixed_type_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                non_numeric_count = numeric_series.isna().sum() - df[col].isna().sum()
                if non_numeric_count > 0:
                    mixed_type_cols.append((col, non_numeric_count))
            except:
                mixed_type_cols.append((col, len(df)))
    
    if mixed_type_cols:
        print(f"   Found {len(mixed_type_cols)} columns with non-numeric values:")
        for col, count in mixed_type_cols[:10]:
            print(f"   {col}: {count} non-numeric values")
    else:
        print("   ✅ No mixed type issues found")
    
    # Check for NaN values
    nan_cols = df.isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    if len(nan_cols) > 0:
        print(f"\n⚠️ Columns with NaN values:")
        for col, count in nan_cols.head(10).items():
            print(f"   {col}: {count} NaN values")
    else:
        print(f"\n✅ No NaN values found")
    
    return object_cols, mixed_type_cols

# Quick fix function
def quick_fix_datatypes(df):
    """Quick fix for data type issues"""
    df_fixed = df.copy()
    
    print("🔧 Applying quick fixes...")
    
    # Convert all columns to numeric where possible
    fixed_cols = []
    for col in df_fixed.columns:
        if df_fixed[col].dtype == 'object':
            try:
                df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
                fixed_cols.append(col)
            except:
                pass
    
    if fixed_cols:
        print(f"✅ Converted {len(fixed_cols)} columns to numeric")
    
    # Fill NaN values
    nan_count = df_fixed.isna().sum().sum()
    if nan_count > 0:
        df_fixed = df_fixed.fillna(method='ffill').fillna(method='bfill').fillna(0)
        print(f"✅ Filled {nan_count} NaN values")
    
    # Final check
    remaining_objects = df_fixed.select_dtypes(include=['object']).columns
    if len(remaining_objects) > 0:
        print(f"⚠️ Still have {len(remaining_objects)} object columns:")
        print(f"   {list(remaining_objects)}")
    else:
        print("✅ All columns are now numeric!")
    
    return df_fixed


def main():
    """Complete state-of-the-art pipeline with all advanced components"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("🌞 COMPLETE STATE-OF-THE-ART SOLAR FORECASTING PIPELINE")
    print("="*80)
    
    # Check GPU
    check_gpu_status()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # =============================================================================
    # STEP 1: DATA LOADING WITH WEATHER INTEGRATION
    # =============================================================================
    print("\n📊 STEP 1: DATA LOADING WITH WEATHER INTEGRATION")
    print("-" * 60)
    
    # Load and process data with robust datetime handling
    data_path = "C:/Users/Lospsy/Desktop/Thesis/Results/forecast_data.csv"
    df = pd.read_csv(data_path)
    
    # Debug datetime column
    datetime_col_name = df.columns[0]
    print(f"Datetime column: {datetime_col_name}")
    print(f"First few values: {df[datetime_col_name].head()}")
    print(f"Data type: {df[datetime_col_name].dtype}")
    
    # Robust datetime processing
    try:
        print("Attempting datetime conversion...")
        datetime_series = pd.to_datetime(df[datetime_col_name], errors='coerce')
        
        # Check if conversion was successful
        if datetime_series.isna().all():
            raise ValueError("All datetime values failed to convert")
        
        print(f"Datetime conversion successful. Sample: {datetime_series.head(2)}")
        
        # Check and handle timezone info
        if hasattr(datetime_series, 'dt') and datetime_series.dt.tz is not None:
            print(f"Timezone detected: {datetime_series.dt.tz}")
            print("Removing timezone info...")
            datetime_series = datetime_series.dt.tz_localize(None)
            print(f"After tz removal: {datetime_series.head(2)}")
        
        # Set as index and ensure it's a DatetimeIndex
        df.index = pd.DatetimeIndex(datetime_series)
        df = df.drop(columns=[datetime_col_name])
        print("✅ Automatic datetime conversion successful!")
        print(f"Final index type: {type(df.index)}")
        print(f"Is DatetimeIndex: {isinstance(df.index, pd.DatetimeIndex)}")
        print(f"Index range: {df.index.min()} to {df.index.max()}")
        
    except Exception as e:
        print(f"⚠️ Automatic conversion failed: {e}")
        print("Using manual approach...")
        
        # Manual conversion - create datetime index from scratch
        start_time = str(df.iloc[0, 0])
        print(f"Manual conversion from: {start_time}")
        
        # Remove timezone info from string if present
        if '+' in start_time:
            start_time = start_time.split('+')[0]
        elif 'T' in start_time and len(start_time.split('T')[1]) > 8:
            start_time = start_time[:19]  # Keep only YYYY-MM-DD HH:MM:SS
        
        # Create datetime range
        try:
            start_dt = pd.to_datetime(start_time)
            n_hours = len(df)
            new_index = pd.date_range(start=start_dt, periods=n_hours, freq='h')
            
            # Set new index and remove datetime column
            df.index = new_index
            df = df.drop(columns=[datetime_col_name])
            print("✅ Manual datetime conversion successful!")
        except Exception as e2:
            print(f"❌ Manual conversion also failed: {e2}")
            # Fallback: create simple integer index with time features
            df = df.drop(columns=[datetime_col_name])
            df.index = range(len(df))
            print("Using fallback integer index...")
    
    # Add basic time features (handle both datetime and integer indices)
    print(f"Index type after conversion: {type(df.index)}")
    
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['dayofweek'] = df.index.dayofweek
        print("✅ Time features created from DatetimeIndex")
    else:
        # If we don't have datetime index, create basic time features
        # Assuming hourly data, create cyclical features
        print("Creating time features from non-datetime index...")
        n_hours = len(df)
        
        # Convert index to numeric if it's not already
        if hasattr(df.index, 'astype'):
            try:
                numeric_index = pd.to_numeric(df.index, errors='coerce')
                if not numeric_index.isna().all():
                    df['hour'] = (numeric_index % 24).astype(int)
                    df['month'] = ((numeric_index // (24 * 30)) % 12 + 1).astype(int)
                    df['dayofweek'] = ((numeric_index // 24) % 7).astype(int)
                else:
                    # Fallback: use position-based features
                    df['hour'] = (np.arange(len(df)) % 24)
                    df['month'] = ((np.arange(len(df)) // (24 * 30)) % 12) + 1
                    df['dayofweek'] = ((np.arange(len(df)) // 24) % 7)
            except:
                # Final fallback: use position-based features
                df['hour'] = (np.arange(len(df)) % 24)
                df['month'] = ((np.arange(len(df)) // (24 * 30)) % 12) + 1
                df['dayofweek'] = ((np.arange(len(df)) // 24) % 7)
        else:
            # Use position-based features
            df['hour'] = (np.arange(len(df)) % 24)
            df['month'] = ((np.arange(len(df)) // (24 * 30)) % 12) + 1
            df['dayofweek'] = ((np.arange(len(df)) // 24) % 7)
        
        print("⚠️ Time features created from index approximation")
    
    # Handle missing values
    print(f"\n🔧 Handling missing values...")
    print(f"Data shape before missing value handling: {df.shape}")
    
    # Wind speed default
    if 'WS_10m' in df.columns:
        missing_wind = df['WS_10m'].isna().sum()
        if missing_wind > 0:
            print(f"Setting default wind speed for {missing_wind} missing values")
            df['WS_10m'] = df['WS_10m'].fillna(3.0)
    
    # Mismatch calculation
    if 'mismatch' in df.columns:
        missing_mismatch = df['mismatch'].isna().sum()
        if missing_mismatch > 0 and 'ac_power_output' in df.columns and 'Load (kW)' in df.columns:
            print(f"Calculating mismatch values for {missing_mismatch} missing values")
            df['mismatch'] = df['mismatch'].fillna(df['ac_power_output'] / 1000 - df['Load (kW)'])
    
    # Fill remaining missing values
    initial_na_count = df.isna().sum().sum()
    if initial_na_count > 0:
        print(f"Filling {initial_na_count} remaining missing values...")
        df = df.ffill().bfill()
        final_na_count = df.isna().sum().sum()
        print(f"Missing values after filling: {final_na_count}")
    
    print(f"✅ Data shape after processing: {df.shape}")
    
    print(f"✅ Basic data loaded: {df.shape}")
    
    # Weather forecast integration
    print("\n🌤️ Weather forecast integration...")
    try:
        lat, lon = 37.98983, 23.74328
        api_key = "7273588818d8b2bb8597ee797baf4935"
        
        forecast_json = fetch_weather_forecast(lat, lon, api_key)
        if forecast_json and isinstance(df.index, pd.DatetimeIndex):
            df_forecast = process_forecast_data(forecast_json)
            df = engineer_weather_features(df, df_forecast)
            print(f"✅ Enhanced with weather data: {df.shape}")
        elif forecast_json:
            print("⚠️ Weather data available but datetime index required for integration")
            print("Proceeding without weather integration...")
        else:
            print("⚠️ Weather forecast not available, using original data")
    except Exception as e:
        print(f"⚠️ Weather integration failed: {e}")
        print("Proceeding with original data...")
    
    # Enhanced feature engineering
    df_enhanced = create_enhanced_features_v2(df)
    print("🔍 Final data type check...")
    object_cols = df_enhanced.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"⚠️ Object columns remaining: {list(object_cols)}")
        for col in object_cols:
            print(f"   {col}: {df_enhanced[col].dtype} - Sample: {df_enhanced[col].head(2).values}")
            df_enhanced[col] = pd.to_numeric(df_enhanced[col], errors='coerce').fillna(0)
    else:
        print("✅ All columns are numeric!")
    # =============================================================================
    # STEP 2: DATASET AND MODEL CREATION
    # =============================================================================
    print("\n📈 STEP 2: DATASET AND MODEL CREATION")
    print("-" * 60)
    
    dataset = SolarProductionDataset(df_enhanced, seq_length=24, forecast_horizon=1, normalize=True)
    
    # Data splitting
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    
    print(f"📊 Data split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create state-of-the-art model
    model_config = {
        'input_size': len(dataset.feature_names),
        'seq_length': 24,
        'forecast_horizon': 1,
        'cnn_filters': [64, 128, 256, 512],
        'kernel_sizes': [3, 5, 7, 9],
        'lstm_hidden': 512,
        'lstm_layers': 3,
        'num_heads': 16,
        'dropout': 0.3,
        'use_uncertainty': True
    }
    
    model = StateOfTheArtSolarModel(**model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🎯 State-of-the-art model created with {total_params:,} parameters")
    
    # =============================================================================
    # STEP 3: ADVANCED TRAINING
    # =============================================================================
    print("\n🚀 STEP 3: ADVANCED TRAINING WITH ALL TECHNIQUES")
    print("-" * 60)
    
    output_dir = Path("C:/Users/Lospsy/Desktop/Thesis/Results/CompleteStateOfTheArt")
    output_dir.mkdir(exist_ok=True, parents=True)
    model_save_path = output_dir / "ultimate_solar_model.pt"
    
    # Train with all advanced techniques
    model, history = train_state_of_art_model_advanced(
        model, train_loader, val_loader,
        epochs=200, lr=0.001, device=device,
        save_path=model_save_path, patience=30,
        use_curriculum=True, use_distillation=False
    )
    
    # =============================================================================
    # STEP 4: ENSEMBLE CREATION AND TRAINING (OPTIONAL)
    # =============================================================================
    print("\n👥 STEP 4: ENSEMBLE MODEL CREATION")
    print("-" * 60)
    
    # Create ensemble (optional - comment out if training single model)
    ensemble_models = create_ensemble_models(model_config, dataset, device=device, n_models=3)
    
    # Train ensemble models (simplified training for ensemble)
    print("Training ensemble models...")
    trained_ensemble = []
    for i, ensemble_model in enumerate(ensemble_models):
        print(f"Training ensemble model {i+1}/3...")
        ensemble_save_path = output_dir / f"ensemble_model_{i+1}.pt"
        
        # Simplified training for ensemble
        trained_model, _ = train_state_of_art_model_advanced(
            ensemble_model, train_loader, val_loader,
            epochs=100, lr=0.001, device=device,
            save_path=ensemble_save_path, patience=20
        )
        trained_ensemble.append(trained_model)
    
    # Create Bayesian ensemble
    bayesian_ensemble = BayesianEnsemble(trained_ensemble)
    
    # =============================================================================
    # STEP 5: COMPREHENSIVE EVALUATION
    # =============================================================================
    print("\n📊 STEP 5: COMPREHENSIVE EVALUATION")
    print("-" * 60)
    
    # Evaluate main model
    eval_results = evaluate_model_comprehensive(
        model, test_loader, dataset, device=device, return_predictions=True
    )
    
    # Evaluate ensemble
    ensemble_predictions = []
    ensemble_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            ensemble_pred, ensemble_unc = bayesian_ensemble.predict(features)
            ensemble_predictions.append(ensemble_pred.cpu().numpy())
            ensemble_targets.append(targets.numpy())
    
    ensemble_predictions = np.concatenate(ensemble_predictions, axis=0).squeeze()
    ensemble_targets = np.concatenate(ensemble_targets, axis=0).squeeze()
    
    if hasattr(dataset, 'denormalize_targets'):
        ensemble_predictions = dataset.denormalize_targets(ensemble_predictions)
        ensemble_targets = dataset.denormalize_targets(ensemble_targets)
    
    # Ensemble metrics
    ensemble_mse = np.mean((ensemble_targets - ensemble_predictions) ** 2)
    ensemble_rmse = np.sqrt(ensemble_mse)
    ensemble_mae = np.mean(np.abs(ensemble_targets - ensemble_predictions))
    ensemble_r2 = 1 - (np.sum((ensemble_targets - ensemble_predictions) ** 2) / 
                      np.sum((ensemble_targets - np.mean(ensemble_targets)) ** 2))
    
    # =============================================================================
    # STEP 6: RESULTS AND COMPARISON
    # =============================================================================
    print("\n🏆 STEP 6: FINAL RESULTS AND COMPARISON")
    print("-" * 60)
    
    baseline_results = {'r2': 0.9506, 'mae': 4432, 'rmse': 12327, 'mape': 27.36}
    
    print("\n" + "="*80)
    print("🎊 COMPLETE STATE-OF-THE-ART RESULTS")
    print("="*80)
    
    print(f"{'Model':<25} {'R²':<10} {'MAE (Wh)':<12} {'RMSE (Wh)':<12} {'MAPE (%)':<10}")
    print("-" * 75)
    print(f"{'Baseline LSTM':<25} {baseline_results['r2']:<10.4f} {baseline_results['mae']:<12.0f} {baseline_results['rmse']:<12.0f} {baseline_results['mape']:<10.2f}")
    print(f"{'SOTA Single Model':<25} {eval_results['r2']:<10.4f} {eval_results['mae']:<12.0f} {eval_results['rmse']:<12.0f} {eval_results['mape']:<10.2f}")
    print(f"{'SOTA Ensemble':<25} {ensemble_r2:<10.4f} {ensemble_mae:<12.0f} {ensemble_rmse:<12.0f} {'N/A':<10}")
    
    # Calculate improvements
    single_r2_improvement = (eval_results['r2'] - baseline_results['r2']) / baseline_results['r2'] * 100
    single_mae_improvement = (baseline_results['mae'] - eval_results['mae']) / baseline_results['mae'] * 100
    
    ensemble_r2_improvement = (ensemble_r2 - baseline_results['r2']) / baseline_results['r2'] * 100
    ensemble_mae_improvement = (baseline_results['mae'] - ensemble_mae) / baseline_results['mae'] * 100
    
    print(f"\n📈 PERFORMANCE IMPROVEMENTS:")
    print(f"Single Model - R²: {single_r2_improvement:+.2f}%, MAE: {single_mae_improvement:+.2f}%")
    print(f"Ensemble Model - R²: {ensemble_r2_improvement:+.2f}%, MAE: {ensemble_mae_improvement:+.2f}%")
    
    if eval_results['peak_mae'] is not None:
        print(f"\n🌟 Peak Hours Performance:")
        print(f"  Peak MAE:  {eval_results['peak_mae']:>10.2f} Wh")
        print(f"  Peak RMSE: {eval_results['peak_rmse']:>10.2f} Wh")
        print(f"  Peak R²:   {eval_results['peak_r2']:>10.4f}")
    
    # Save comprehensive results
    final_results = {
        'single_model': {k: float(v) if isinstance(v, (np.number, np.ndarray)) and np.isscalar(v) else v 
                        for k, v in eval_results.items() if k not in ['predictions', 'targets']},
        'ensemble_model': {
            'r2': float(ensemble_r2), 'mae': float(ensemble_mae), 
            'rmse': float(ensemble_rmse), 'mse': float(ensemble_mse)
        },
        'improvements': {
            'single_r2': float(single_r2_improvement),
            'single_mae': float(single_mae_improvement),
            'ensemble_r2': float(ensemble_r2_improvement),
            'ensemble_mae': float(ensemble_mae_improvement)
        },
        'model_config': model_config,
        'training_summary': {
            'total_epochs': len(history['train_loss']) if history else 0,
            'best_val_loss': min(history['val_loss']) if history else None,
            'features_used': len(dataset.feature_names),
            'weather_integration': 'forecast_temp' in df_enhanced.columns
        }
    }
    
    with open(output_dir / "complete_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Complete results saved to: {output_dir}")
    print("\n🎉 STATE-OF-THE-ART PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return model, eval_results, final_results

if __name__ == "__main__":
    main()