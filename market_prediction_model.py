#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn.functional as F
from scipy.stats import pearsonr
from scipy.stats.mstats import winsorize
from custom_loss_functions import (FocalRegressionLoss, SoftClassificationLoss, 
                                  CrossEntropyRegressionLoss, HybridLoss,
                                  GaussianNLLLoss, ClassificationRegressionHybridLoss)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class MarketDataset(Dataset):
    def __init__(self, features, targets, sequence_length=30):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        feature_seq = self.features[idx:idx + self.sequence_length]
        target = self.targets[idx + self.sequence_length - 1]
        
        # 对于分类任务，目标值应该是标量（dim=1）
        return torch.FloatTensor(feature_seq), torch.tensor(target, dtype=torch.long)

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes=20, d_model=128, nhead=8, num_layers=4, dropout=0.2):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))  
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
    def forward(self, x):
        # x_shape: [batch_size, seq_len, input_size]
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)                   # [batch, seq_len, d_model]
        # 位置编码
        pos_encoding = self.positional_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_encoding
        # Transformer编码
        transformer_out = self.transformer_encoder(x)  # [batch, seq_len, d_model]
        # 最后一个时间步的输出进行分类
        last_hidden = transformer_out[:, -1, :]        # [batch, d_model]
        logits = self.classifier(last_hidden)
        
        return logits

class AutoEncoder(nn.Module):
    def __init__(self, input_size, encoding_size, dropout=0.2):
        super(AutoEncoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.BatchNorm1d(input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 2, input_size // 4),
            nn.BatchNorm1d(input_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 4, encoding_size),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, input_size // 4),
            nn.BatchNorm1d(input_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 4, input_size // 2),
            nn.BatchNorm1d(input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 2, input_size)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def encode(self, x):
        return self.encoder(x)

class MarketPredictor:
    def __init__(self, data_path, sequence_length=30, test_days=10, encoding_size=10, loss_type='cross_entropy', l1_lambda=0.001):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.test_days = test_days
        self.encoding_size = encoding_size  # 自编码器编码维度
        self.loss_type = loss_type
        self.l1_lambda = l1_lambda          # L1
        self.scaler_features = RobustScaler()
        self.scaler_targets  = StandardScaler()
        self.autoencoder = None   # 自编码器
        self.model  = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.allowed_values = np.array([-6.0, -5.0, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 
                                       0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
        self.num_classes = len(self.allowed_values)  # 分类数量
        print(f"Using device: {self.device}")
        print(f"AutoEncoder encoding size: {encoding_size}")
        print(f"Sequence length: {sequence_length}")
        print(f"Loss function type: {loss_type}")
        print(f"L1 regularization lambda: {l1_lambda}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class values: {self.allowed_values}")
    

    """------------- part0: utils -------------"""
    def get_loss_function(self):
        """根据loss_type返回相应的损失函数"""
        if self.loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.loss_type == 'mse':
            return nn.MSELoss()
        elif self.loss_type == 'smoothl1':
            return nn.SmoothL1Loss()
        elif self.loss_type == 'focal':
            return FocalRegressionLoss(alpha=1.0, gamma=2.0)
        elif self.loss_type == 'soft_classification':
            return SoftClassificationLoss(self.allowed_values, temperature=1.0)
        elif self.loss_type == 'cross_entropy_regression':
            return CrossEntropyRegressionLoss(self.allowed_values, sigma=0.5)
        elif self.loss_type == 'hybrid':
            return HybridLoss(self.allowed_values, mse_weight=0.3, ce_weight=0.7)
        elif self.loss_type == 'gaussian_nll':
            return GaussianNLLLoss(eps=1e-6)
        elif self.loss_type == 'classification_regression_hybrid':
            return ClassificationRegressionHybridLoss(num_classes=20, value_range=(-0.1, 0.1), alpha=0.5)
        else:
            print(f"Unknown loss type: {self.loss_type}, using CrossEntropyLoss")
            return nn.CrossEntropyLoss()

    def calculate_l1_regularization(self):
        """计算L1正则项"""
        l1_reg = 0
        for param in self.model.parameters():
            l1_reg += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_reg

    def continuous_to_class_labels(self, continuous_values):
        """将连续值转换为分类标签（类别索引）"""
        class_labels = np.zeros(len(continuous_values), dtype=np.int64)
        for i, value in enumerate(continuous_values):
            # 找到最接近的允许值的索引
            distances = np.abs(self.allowed_values - value)
            closest_idx = np.argmin(distances)
            class_labels[i] = closest_idx
        return class_labels
    
    def class_labels_to_continuous(self, class_labels):
        """将分类标签转换回连续值"""
        return self.allowed_values[class_labels]

    def constrain_predictions(self, predictions):
        """
        将连续预测值约束到指定的离散值
        """
        constrained = np.zeros_like(predictions)
        for i, pred in enumerate(predictions):
            # 找到最接近的允许值
            distances = np.abs(self.allowed_values - pred)
            closest_idx = np.argmin(distances)
            constrained[i] = self.allowed_values[closest_idx]
        return constrained
        

    """------------- part1: Load data & Train model -------------"""
    def load_data(self):
        print("Loading data")
        date_folders = sorted([d for d in os.listdir(self.data_path) 
                              if os.path.isdir(os.path.join(self.data_path, d))])
        
        all_features = []
        all_targets  = []
        all_times    = []
        all_lastprices = []
        
        for date_folder in date_folders:
            folder_path = os.path.join(self.data_path, date_folder)
            
            factor_file = os.path.join(folder_path, 'factor_values.csv')
            target_file = os.path.join(folder_path, 'y_values.csv')
            time_file = os.path.join(folder_path,   'times.csv')
            
            if os.path.exists(factor_file) and os.path.exists(target_file) and os.path.exists(time_file):
                features = pd.read_csv(factor_file, header=None).values
                targets = pd.read_csv(target_file, header=None).values.flatten()
                times = pd.read_csv(time_file, header=None).values.flatten()
                
                min_length = min(len(features), len(targets), len(times))
                features = features[:min_length]
                targets = targets[:min_length]
                times = times[:min_length]
                
                lastprices = features[:, -1] if features.shape[1] > 0 else targets
                
                all_features.append(features)
                all_targets.append(targets)
                all_lastprices.append(lastprices)
                all_times.extend([f"{date_folder}_{i}" for i in range(len(times))])
                
                print(f"Loading {date_folder}: {len(features)} records")
        
        self.features = np.vstack(all_features)
        self.targets = np.concatenate(all_targets)
        self.lastprices = np.concatenate(all_lastprices)
        self.times = all_times
        
        print(f"Total loaded: {len(self.features)} records, {self.features.shape[1]} features")
        
        self._clean_data()
        
        return self.features, self.targets
    
    def _clean_data(self):
        print("Cleaning data...")
        
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        self.targets = np.nan_to_num(self.targets, nan=0.0, posinf=0.0, neginf=0.0)
        self.lastprices = np.nan_to_num(self.lastprices, nan=0.0, posinf=0.0, neginf=0.0)
        
        for i in range(self.features.shape[1]):
            self.features[:, i] = winsorize(self.features[:, i], limits=[0.05, 0.05])
        
        self.targets = winsorize(self.targets, limits=[0.05, 0.05])
        print("Data cleaning completed")
    
    def train_autoencoder(self, train_features, epochs=50, batch_size=256, learning_rate=0.001):
        """训练自编码器进行特征降维"""
        print(f"Training AutoEncoder for dimensionality reduction: {train_features.shape[1]} -> {self.encoding_size}")
        # 创建自编码器
        input_size = train_features.shape[1]
        self.autoencoder = AutoEncoder(input_size, self.encoding_size).to(self.device)
        optimizer = optim.AdamW(self.autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        # Dataloader
        train_tensor = torch.FloatTensor(train_features).to(self.device)
        dataset      = torch.utils.data.TensorDataset(train_tensor)
        dataloader   = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train
        self.autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_data, in dataloader:
                optimizer.zero_grad()
                encoded, decoded = self.autoencoder(batch_data)
                loss = criterion(decoded, batch_data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"AutoEncoder Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        print("AutoEncoder training completed!")
    
    def apply_autoencoder(self, train_features, test_features):
        """应用自编码器进行特征降维"""
        # 训练自编码器
        self.train_autoencoder(train_features)

        self.autoencoder.eval()
        with torch.no_grad():
            train_tensor = torch.FloatTensor(train_features).to(self.device)
            test_tensor = torch.FloatTensor(test_features).to(self.device)
            
            train_encoded = self.autoencoder.encode(train_tensor).cpu().numpy()
            test_encoded = self.autoencoder.encode(test_tensor).cpu().numpy()
        
        print(f"AutoEncoder encoding completed: {train_features.shape[1]} -> {self.encoding_size}")
        print(f"Train features shape: {train_encoded.shape}")
        print(f"Test features shape: {test_encoded.shape}")
        
        return train_encoded, test_encoded
    
    def prepare_data(self):
        print("Preparing training and test data...")
        
        date_folders = sorted([d for d in os.listdir(self.data_path) 
                              if os.path.isdir(os.path.join(self.data_path, d))])
        
        test_dates = date_folders[-self.test_days:]
        train_dates = date_folders[:-self.test_days]
        
        print(f"Training dates: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
        print(f"Test dates: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
        
        train_features, train_targets, train_lastprices = self._load_data_by_dates(train_dates)
        test_features, test_targets, test_lastprices = self._load_data_by_dates(test_dates)
        
        train_features_encoded, test_features_encoded = self.apply_autoencoder(train_features, test_features)
        
        train_features_scaled = self.scaler_features.fit_transform(train_features_encoded)
        test_features_scaled = self.scaler_features.transform(test_features_encoded)
        
        # 对于分类模型，将连续目标值转换为分类标签
        train_targets_class = self.continuous_to_class_labels(train_targets)
        test_targets_class = self.continuous_to_class_labels(test_targets)
        
        print(f"Converted targets to {self.num_classes} classes")
        print(f"Train class distribution: {np.bincount(train_targets_class, minlength=self.num_classes)}")
        print(f"Test class distribution: {np.bincount(test_targets_class, minlength=self.num_classes)}")
        
        self.train_dataset = MarketDataset(train_features_scaled, train_targets_class, self.sequence_length)
        self.test_dataset = MarketDataset(test_features_scaled, test_targets_class, self.sequence_length)
        
        self.test_features_original = test_features
        self.test_targets_original = test_targets
        self.test_lastprices_original = test_lastprices
        
        print(f"Training set size: {len(self.train_dataset)}")
        print(f"Test set size: {len(self.test_dataset)}")
        
        return self.train_dataset, self.test_dataset
    
    def _load_data_by_dates(self, date_list):
        features_list = []
        targets_list = []
        lastprices_list = []
        
        for date_folder in date_list:
            folder_path = os.path.join(self.data_path, date_folder)
            
            factor_file = os.path.join(folder_path, 'factor_values.csv')
            target_file = os.path.join(folder_path, 'y_values.csv')
            
            if os.path.exists(factor_file) and os.path.exists(target_file):
                features = pd.read_csv(factor_file, header=None).values
                targets = pd.read_csv(target_file, header=None).values.flatten()
                
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
                
                lastprices = features[:, -1] if features.shape[1] > 0 else targets
                
                min_length = min(len(features), len(targets))
                features_list.append(features[:min_length])
                targets_list.append(targets[:min_length])
                lastprices_list.append(lastprices[:min_length])
        
        return np.vstack(features_list), np.concatenate(targets_list), np.concatenate(lastprices_list)
    
    def create_model(self):
        input_size = self.encoding_size
        self.model = TransformerModel(
            input_size=input_size,
            num_classes=self.num_classes,
            d_model=128,
            nhead=8,
            num_layers=4,
            dropout=0.2
        ).to(self.device)
        
        print(f"Transformer model created successfully, parameter count: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Model output classes: {self.num_classes}")
        print(f"Model architecture: Transformer with {4} layers, {8} attention heads, d_model={128}")
        return self.model
    
    def train_model(self, epochs=20, batch_size=64, learning_rate=0.001):
        print("=" * 80)
        print("Starting model training...")
        print("=" * 80)
        
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        
        criterion = self.get_loss_function()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate*0.01)
        
        train_losses = []
        ic_values    = []  
        best_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        total_batches = len(train_loader)
        start_time = time.time()
        
        print(f"Training configuration:")
        print(f" - Total epochs: {epochs}")
        print(f" - Batch size: {batch_size}")
        print(f" - Learning rate: {learning_rate}")
        print(f" - Batches per epoch: {total_batches}")
        print(f" - Early stopping patience: {patience}")
        print(f" - Sequence length: {self.sequence_length}")
        print(f" - AutoEncoder features: {self.encoding_size}")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            for batch_idx, (batch_features, batch_targets) in enumerate(train_loader):
                batch_start_time = time.time()
                
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)  
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                
                # L1
                l1_reg = self.calculate_l1_regularization()
                total_loss = loss + l1_reg
                
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += total_loss.item()
                
                batch_time = time.time() - batch_start_time
                
                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    progress = (batch_idx + 1) / total_batches * 100
                    avg_batch_loss = epoch_loss / (batch_idx + 1)
                    it_per_sec = 1.0 / batch_time if batch_time > 0 else 0
                    
                    print(f"  Batch [{batch_idx+1:4d}/{total_batches}] "
                           f"({progress:5.1f}%) | "
                           f"Base Loss: {loss.item():.6f} | "
                           f"L1 Reg: {l1_reg.item():.6f} | "
                           f"Total Loss: {total_loss.item():.6f} | "
                           f"Avg Loss: {avg_batch_loss:.6f} | "
                           f"Speed: {it_per_sec:.2f} it/s")
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            scheduler.step()
            
            # 计算当前轮次的IC值
            current_ic = self.calculate_ic_during_training()
            ic_values.append(current_ic)
            
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                improvement_status = "新最佳"
            else:
                patience_counter += 1
                improvement_status = f"无改进 ({patience_counter}/{patience})"
            
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch+1} completed:")
            print(f" - Average loss: {avg_loss:.6f}")
            print(f" - Best loss: {best_loss:.6f}")
            print(f" - IC value: {current_ic:.6f}")
            print(f" - Learning rate: {current_lr:.8f}")
            print(f" - Epoch time: {epoch_time:.1f}s")
            print(f" - Total time: {total_time/60:.1f}min")
            print(f" - Status: {improvement_status}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered! Training stopped at epoch {epoch+1}")
                print(f"Best loss: {best_loss:.6f}")
                break
        
        self.model.load_state_dict(torch.load('best_model.pth'))
        total_training_time = time.time() - start_time
        
        print(f"\nTraining completed!")
        print(f"Total training time: {total_training_time/60:.1f} minutes")
        print(f"Final best loss: {best_loss:.6f}")
        print("=" * 80)
        
        return train_losses, ic_values
    

    """------------- part2: maodel evaluation -------------"""
    def calculate_ic(self, predictions, test_targets_original):
        if len(predictions) != len(test_targets_original):
            min_len = min(len(predictions), len(test_targets_original))
            predictions = predictions[:min_len]
            test_targets_original = test_targets_original[:min_len]
        
        if len(test_targets_original) < 2:
            return 0.0
        
        try:
            future_returns = np.diff(test_targets_original) / test_targets_original[:-1]
            preds_aligned = predictions[:-1]
            
            if len(preds_aligned) != len(future_returns):
                min_len = min(len(preds_aligned), len(future_returns))
                preds_aligned = preds_aligned[:min_len]
                future_returns = future_returns[:min_len]
            
            if len(preds_aligned) < 2:
                return 0.0
                
            ic, p_value = pearsonr(preds_aligned, future_returns)
            return ic if not np.isnan(ic) else 0.0
        except:
            return 0.0
    
    def calculate_ic_during_training(self):
        """在训练过程中计算IC值"""
        self.model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                
                if self.loss_type == 'cross_entropy':
                    # 对于分类模型，获取预测的类别
                    predicted_classes = torch.argmax(outputs, dim=1)
                    predictions.extend(predicted_classes.cpu().numpy())
                else:
                    # 对于回归模型
                    predictions.extend(outputs.cpu().numpy().flatten())
                
                actuals.extend(batch_targets.numpy().flatten())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        if self.loss_type == 'cross_entropy':
            # 将分类标签转换回连续值
            predictions_original = self.class_labels_to_continuous(predictions)
            actuals_original = self.class_labels_to_continuous(actuals)
        else:
            # 对于回归模型，使用scaler反向转换
            predictions = predictions.reshape(-1, 1)
            actuals = actuals.reshape(-1, 1)
            predictions_original = self.scaler_targets.inverse_transform(predictions).flatten()
            actuals_original = self.scaler_targets.inverse_transform(actuals).flatten()
        
        ic = self.calculate_ic(predictions_original, actuals_original)
        return ic
    
    def calculate_ic_shape(self, predictions, actuals):
        """
        计算IC shape指标 - 衡量预测值与实际值的形状相似性
        """
        try:
            # 计算预测值和实际值的排序相关性
            from scipy.stats import spearmanr
            ic_shape, p_value = spearmanr(predictions, actuals)
            return ic_shape if not np.isnan(ic_shape) else 0.0
        except:
            return 0.0
    
    def calculate_direction_metrics(self, predictions, actuals):
        """
        计算详细的方向正确率指标
        """
        # 基本方向正确率（预测为正且实际为正，或预测为负且实际为负）
        pred_positive = predictions > 0
        actual_positive = actuals > 0
        pred_negative = predictions <= 0
        actual_negative = actuals <= 0
        
        # 预测为正值时实际也为正值的准确率
        positive_correct = np.sum(pred_positive & actual_positive)
        total_positive_pred = np.sum(pred_positive)
        positive_accuracy = positive_correct / total_positive_pred if total_positive_pred > 0 else 0.0
        
        # 预测为负值时实际也为负值的准确率
        negative_correct = np.sum(pred_negative & actual_negative)
        total_negative_pred = np.sum(pred_negative)
        negative_accuracy = negative_correct / total_negative_pred if total_negative_pred > 0 else 0.0
        
        # 总体方向正确率
        total_correct = positive_correct + negative_correct
        total_predictions = len(predictions)
        overall_direction_accuracy = total_correct / total_predictions
        
        return {
            'positive_accuracy': positive_accuracy,
            'negative_accuracy': negative_accuracy,
            'overall_direction_accuracy': overall_direction_accuracy,
            'positive_predictions': total_positive_pred,
            'negative_predictions': total_negative_pred,
            'positive_correct': positive_correct,
            'negative_correct': negative_correct
        }

    def calculate_class_accuracy(self, predictions, actuals, num_classes=10, value_range=None):
        """
        计算每个类别的识别准确率
        将连续的预测值和真实值转换为类别，然后计算每个类别的准确率
        
        Args:
            predictions: 预测值数组
            actuals: 真实值数组
            num_classes: 类别数量
            value_range: 值的范围，如果为None则自动计算
        
        Returns:
            dict: 包含每个类别准确率的字典
        """
        # 处理空数据情况
        if len(predictions) == 0 or len(actuals) == 0:
            return {
                'class_accuracies': {},
                'class_counts': {},
                'class_correct': {},
                'overall_class_accuracy': 0.0,
                'class_boundaries': np.array([]),
                'num_classes': 0
            }
        
        if value_range is None:
            # 自动计算值的范围
            all_values = np.concatenate([predictions, actuals])
            value_min, value_max = np.percentile(all_values, [5, 95])  # 使用5%和95%分位数避免极值影响
        else:
            value_min, value_max = value_range
        
        # 创建类别边界
        class_boundaries = np.linspace(value_min, value_max, num_classes + 1)
        
        def values_to_classes(values):
            """将连续值转换为类别标签"""
            classes = np.zeros_like(values, dtype=int)
            
            for i in range(num_classes):
                if i == num_classes - 1:
                    # 最后一个类别包含右边界
                    mask = (values >= class_boundaries[i]) & (values <= class_boundaries[i + 1])
                else:
                    mask = (values >= class_boundaries[i]) & (values < class_boundaries[i + 1])
                classes[mask] = i
            
            # 处理超出范围的值
            classes[values < class_boundaries[0]] = 0
            classes[values > class_boundaries[-1]] = num_classes - 1
            
            return classes
        
        # 转换为类别
        pred_classes = values_to_classes(predictions)
        actual_classes = values_to_classes(actuals)
        
        # 计算每个类别的准确率
        class_accuracies = {}
        class_counts = {}
        class_correct = {}
        
        for class_idx in range(num_classes):
            # 找到真实值属于当前类别的样本
            actual_class_mask = (actual_classes == class_idx)
            actual_class_count = np.sum(actual_class_mask)
            
            if actual_class_count > 0:
                # 在这些样本中，预测也正确的数量
                correct_predictions = np.sum(pred_classes[actual_class_mask] == class_idx)
                accuracy = correct_predictions / actual_class_count
            else:
                accuracy = 0.0
                correct_predictions = 0
            
            # 计算类别范围标签
            if class_idx == 0:
                range_label = f"≤{class_boundaries[1]:.2f}"
            elif class_idx == num_classes - 1:
                range_label = f">{class_boundaries[class_idx]:.2f}"
            else:
                range_label = f"{class_boundaries[class_idx]:.2f}~{class_boundaries[class_idx+1]:.2f}"
            
            class_accuracies[f"Class_{class_idx}_{range_label}"] = accuracy
            class_counts[f"Class_{class_idx}"] = actual_class_count
            class_correct[f"Class_{class_idx}"] = correct_predictions
        
        # 计算总体分类准确率
        total_correct = np.sum(pred_classes == actual_classes)
        total_samples = len(predictions)
        overall_class_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            'class_accuracies': class_accuracies,
            'class_counts': class_counts,
            'class_correct': class_correct,
            'overall_class_accuracy': overall_class_accuracy,
            'class_boundaries': class_boundaries,
            'num_classes': num_classes
        }

    def evaluate_model(self):
        print("Evaluating classification model...")
        
        self.model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        
        predicted_classes = []
        actual_classes = []
        predicted_probs = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device).long()
                
                # get 模型输出（logits）
                outputs = self.model(batch_features)
                # get softmax概率
                probs = F.softmax(outputs, dim=1)
                # get 预测的类别
                _, predicted = torch.max(outputs, 1)
                
                predicted_classes.extend(predicted.cpu().numpy())
                actual_classes.extend(batch_targets.cpu().numpy())
                predicted_probs.extend(probs.cpu().numpy())
        
        predicted_classes = np.array(predicted_classes)
        actual_classes = np.array(actual_classes)
        predicted_probs = np.array(predicted_probs)
        
        # 将类别标签转换回连续值用于指标计算
        predicted_continuous = self.class_labels_to_continuous(predicted_classes)
        actual_continuous = self.class_labels_to_continuous(actual_classes)
        
        # 分类准确率
        classification_accuracy = np.mean(predicted_classes == actual_classes)
        
        # 每个类别的准确率
        class_accuracies = {}
        class_counts = {}
        class_correct = {}
        
        for class_idx in range(self.num_classes):
            mask = actual_classes == class_idx
            if np.sum(mask) > 0:
                class_accuracy = np.mean(predicted_classes[mask] == actual_classes[mask])
                class_accuracies[f'Class_{class_idx}'] = class_accuracy
                class_counts[f'Class_{class_idx}'] = np.sum(mask)
                class_correct[f'Class_{class_idx}'] = np.sum(predicted_classes[mask] == actual_classes[mask])
            else:
                class_accuracies[f'Class_{class_idx}'] = 0.0
                class_counts[f'Class_{class_idx}'] = 0
                class_correct[f'Class_{class_idx}'] = 0
        
        # Top-3准确率
        top3_accuracy = 0.0
        if predicted_probs.shape[1] >= 3:
            top3_predictions = np.argsort(predicted_probs, axis=1)[:, -3:]
            top3_accuracy = np.mean([actual_classes[i] in top3_predictions[i] for i in range(len(actual_classes))])
        
        # 方向指标（基于连续值）
        direction_metrics = self.calculate_direction_metrics(predicted_continuous, actual_continuous)
        
        # 计算IC指标（基于连续值）
        ic = self.calculate_ic(predicted_continuous, actual_continuous)
        ic_shape = self.calculate_ic_shape(predicted_continuous, actual_continuous)
        
        # 计算MSE和MAE（基于连续值）
        mse = mean_squared_error(actual_continuous, predicted_continuous)
        mae = mean_absolute_error(actual_continuous, predicted_continuous)
        rmse = np.sqrt(mse)
        
        print(f"=== 分类模型评估结果 ===")
        print(f"分类准确率: {classification_accuracy:.4f}")
        print(f"Top-3准确率: {top3_accuracy:.4f}")
        print(f"MSE (连续值): {mse:.6f}")
        print(f"MAE (连续值): {mae:.6f}")
        print(f"RMSE (连续值): {rmse:.6f}")
        print(f"方向准确率: {direction_metrics['overall_direction_accuracy']:.4f}")
        print(f"IC (信息系数): {ic:.6f}")
        print(f"IC Shape (Spearman相关性): {ic_shape:.6f}")
        
        # 输出每个类别的准确率
        print(f"\n=== 每个类别的准确率 ===")
        for class_idx in range(self.num_classes):
            class_name = f'Class_{class_idx}'
            accuracy = class_accuracies[class_name]
            count = class_counts[class_name]
            correct = class_correct[class_name]
            value = self.allowed_values[class_idx]
            print(f"{class_name} (值={value:.2f}): {accuracy:.4f} ({correct}/{count})")
        
        # 输出类别分布
        print(f"\n=== 预测类别分布 ===")
        pred_dist   = np.bincount(predicted_classes, minlength=self.num_classes)
        actual_dist = np.bincount(actual_classes, minlength=self.num_classes)
        for class_idx in range(self.num_classes):
            print(f"Class_{class_idx}: 预测={pred_dist[class_idx]}, 实际={actual_dist[class_idx]}")
        
        return {
            'predicted_classes': predicted_classes,
            'actual_classes': actual_classes,
            'predicted_continuous': predicted_continuous,
            'actual_continuous': actual_continuous,
            'predicted_probs': predicted_probs,
            'classification_accuracy': classification_accuracy,
            'top3_accuracy': top3_accuracy,
            'class_accuracies': class_accuracies,
            'class_counts': class_counts,
            'class_correct': class_correct,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'direction_accuracy': direction_metrics['overall_direction_accuracy'],
            'ic': ic,
            'ic_shape': ic_shape,
            'direction_metrics': direction_metrics
        }
    

    """------------- part3: 可视化 -------------"""
    def plot_results(self, results):
        # 根据模型类型选择合适的预测值和实际值
        if self.loss_type == 'cross_entropy':
            # 分类模型使用连续值进行绘图
            predictions = results['predicted_continuous']
            predictions_original = results['predicted_continuous']  # 分类模型没有原始预测
            actuals = results['actual_continuous']
        else:
            # 回归模型使用约束预测值
            predictions = results['predictions_constrained']        # Use constrained predictions
            predictions_original = results['predictions_original']  # Original predictions
            actuals = results['actuals']
        
        plt.figure(figsize=(18, 20))  
        
        plt.subplot(4, 3, 1)
        plt.scatter(actuals, predictions, alpha=0.5)
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual Values')
        
        plt.subplot(4, 3, 2)
        plt.plot(actuals[:500], label='Actual', alpha=0.7)
        plt.plot(predictions[:500], label='Predicted', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title('Time Series Comparison (First 500 Points)')
        plt.legend()
        
        plt.subplot(4, 3, 3)
        residuals = predictions - actuals
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        
        plt.subplot(4, 3, 4)
        cumulative_error = np.cumsum(np.abs(residuals))
        plt.plot(cumulative_error)
        plt.xlabel('Time')
        plt.ylabel('Cumulative Absolute Error')
        plt.title('Cumulative Absolute Error')
        
        plt.subplot(4, 3, 5)
        actual_direction = np.sign(np.diff(actuals))
        pred_direction = np.sign(np.diff(predictions))
        direction_match = (actual_direction == pred_direction).astype(int)
        rolling_accuracy = np.convolve(direction_match, np.ones(50)/50, mode='valid')
        plt.plot(rolling_accuracy)
        plt.xlabel('Time')
        plt.ylabel('Rolling Direction Accuracy')
        plt.title('50-Period Rolling Direction Accuracy')
        
        plt.subplot(4, 3, 6)
        if self.loss_type == 'cross_entropy':
            # 分类模型指标
            metrics_names = ['MSE', 'MAE', 'RMSE', 'Class Acc', 'Direction Acc', 'IC']
            metrics_values = [results['mse'], results['mae'], results['rmse'], 
                             results['classification_accuracy'], results['direction_accuracy'], 
                             results['ic']]
        else:
            # 回归模型指标
            metrics_names = ['MSE', 'MAE', 'RMSE', 'R²', 'Direction Acc', 'Sharpe', 'IC']
            metrics_values = [results['mse'], results['mae'], results['rmse'], 
                             results['r2'], results['direction_accuracy'], 
                             results['sharpe_ratio'], results['ic']]
        
        colors = ['red' if v < 0 else 'green' for v in metrics_values]
        plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Model Evaluation Metrics')
        plt.xticks(rotation=45)
        
        # 添加 IC 图
        if 'ic_values' in results and results['ic_values']:
            plt.subplot(4, 3, 7)
            epochs = range(1, len(results['ic_values']) + 1)
            plt.plot(epochs, results['ic_values'], 'b-', linewidth=2, marker='o', markersize=4)
            plt.xlabel('Training Epochs')
            plt.ylabel('IC Value')
            plt.title('IC Changes During Training')
            plt.grid(True, alpha=0.3)
            
            # Add best IC value marker
            best_ic_idx = np.argmax(results['ic_values'])
            best_ic_value = results['ic_values'][best_ic_idx]
            plt.axhline(y=best_ic_value, color='r', linestyle='--', alpha=0.7, 
                       label=f'Best IC: {best_ic_value:.4f} (Epoch {best_ic_idx+1})')
            plt.legend()
        
        # 添加类别准确率图表
        if 'class_metrics' in results:
            class_metrics = results['class_metrics']
            plt.subplot(4, 3, 8)
            class_names = []
            class_accuracies = []
            class_counts = []
            
            for class_name, accuracy in class_metrics['class_accuracies'].items():
                class_idx = class_name.split('_')[1]
                range_label = '_'.join(class_name.split('_')[2:])  # 获取范围标签
                class_names.append(f"C{class_idx}\n{range_label}")
                class_accuracies.append(accuracy)
                class_counts.append(class_metrics['class_counts'][f'Class_{class_idx}'])
            
            # 根据样本数量设置颜色深度
            max_count = max(class_counts) if class_counts else 1
            colors = plt.cm.Blues([count/max_count for count in class_counts])
            
            bars = plt.bar(range(len(class_names)), class_accuracies, color=colors, alpha=0.8)
            plt.xlabel('Class Range')
            plt.ylabel('Accuracy')
            plt.title('Class-wise Accuracy')
            plt.xticks(range(len(class_names)), class_names, rotation=45, fontsize=8)
            
            # 在柱状图上添加数值标签
            for i, (bar, acc, count) in enumerate(zip(bars, class_accuracies, class_counts)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}\n({count})', ha='center', va='bottom', fontsize=7)
            
            # 第9个位置：类别分布饼图
            plt.subplot(4, 3, 9)
            # 只显示有样本的类别
            non_zero_counts = [(name.split('\n')[0], count) for name, count in zip(class_names, class_counts) if count > 0]
            if non_zero_counts:
                labels, counts = zip(*non_zero_counts)
                plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
                plt.title('Class Distribution')
            
            # 第10个位置：类别准确率与样本数量的关系
            plt.subplot(4, 3, 10)
            plt.scatter(class_counts, class_accuracies, s=100, alpha=0.7, c=range(len(class_counts)), cmap='viridis')
            plt.xlabel('Sample Count')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Sample Count by Class')
            
            # 添加类别标签
            for i, (count, acc, name) in enumerate(zip(class_counts, class_accuracies, class_names)):
                if count > 0:  # 只标注有样本的类别
                    plt.annotate(f'C{i}', (count, acc), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
            
            # 第11个位置：总体分类准确率对比
            plt.subplot(4, 3, 11)
            accuracy_types = ['Overall\nClass Acc', 'Direction\nAcc', 'Positive\nAcc', 'Negative\nAcc']
            accuracy_values = [
                class_metrics['overall_class_accuracy'],
                results['direction_accuracy'],
                results['positive_accuracy'],
                results['negative_accuracy']
            ]
            
            colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']
            bars = plt.bar(accuracy_types, accuracy_values, color=colors, alpha=0.8)
            plt.ylabel('Accuracy')
            plt.title('Different Types of Accuracy')
            plt.ylim(0, 1)
            
            # 添加数值标签
            for bar, val in zip(bars, accuracy_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_predictions(self, results):
        # 根据模型类型选择合适的键值
        if self.loss_type == 'cross_entropy':
            # 分类模型
            actual_values = results['actual_continuous']
            predicted_values = results['predicted_continuous']
            predicted_classes = results['predicted_classes']
            actual_classes = results['actual_classes']
            
            # 保存分类模型的预测结果
            predictions_df = pd.DataFrame({
                'actual_continuous': actual_values,
                'predicted_continuous': predicted_values,
                'actual_classes': actual_classes,
                'predicted_classes': predicted_classes,
                'residual_continuous': predicted_values - actual_values
            })
            
            predictions_df.to_csv('test_predictions.csv', index=False)
            print("Classification prediction results saved to test_predictions.csv")
            
            # 保存分类模型的评估指标
            metrics_df = pd.DataFrame({
                'metric': ['MSE', 'MAE', 'RMSE', 'Classification_Accuracy', 'Top3_Accuracy',
                          'Direction_Accuracy', 'IC', 'IC_Shape'],
                'value': [results['mse'], results['mae'], results['rmse'], 
                         results['classification_accuracy'], results['top3_accuracy'],
                         results['direction_accuracy'], results['ic'], results['ic_shape']]
            })
            
        else:
            # 回归模型
            predictions_df = pd.DataFrame({
                'actual': results['actuals'],
                'predicted_original': results['predictions_original'],
                'predicted_constrained': results['predictions_constrained'],
                'residual_original': results['predictions_original'] - results['actuals'],
                'residual_constrained': results['predictions_constrained'] - results['actuals']
            })
            
            predictions_df.to_csv('test_predictions.csv', index=False)
            print("Regression prediction results saved to test_predictions.csv")
            
            # 保存回归模型的评估指标
            metrics_df = pd.DataFrame({
                'metric': ['MSE', 'MAE', 'RMSE', 'R²', 'Overall_Direction_Accuracy', 
                          'Positive_Prediction_Accuracy', 'Negative_Prediction_Accuracy',
                          'Positive_Predictions_Count', 'Negative_Predictions_Count',
                          'Positive_Correct_Count', 'Negative_Correct_Count',
                          'Sharpe_Ratio', 'IC', 'IC_Shape'],
                'value': [results['mse'], results['mae'], results['rmse'], results['r2'], 
                         results['direction_accuracy'], results['positive_accuracy'], 
                         results['negative_accuracy'], results['positive_predictions'],
                         results['negative_predictions'], results['positive_correct'],
                         results['negative_correct'], results['sharpe_ratio'], 
                         results['ic'], results['ic_shape']]
            })
        
        metrics_df.to_csv('evaluation_metrics.csv', index=False)
        print("Evaluation metrics saved to evaluation_metrics.csv")

def main():
    data_path = "/Project/python project/mx/ru2005_ModelData"
    
    # 可选择的损失函数类型: 'mse', 'smoothl1', 'focal', 'soft_classification', 'cross_entropy_regression', 'hybrid', 'gaussian_nll', 'classification_regression_hybrid', 'cross_entropy'
    predictor = MarketPredictor(data_path, sequence_length=30, test_days=10, encoding_size=10, loss_type='cross_entropy')
    
    predictor.load_data()
    
    predictor.prepare_data()
    
    predictor.create_model()
    
    train_losses, ic_values = predictor.train_model(epochs=10, batch_size=64, learning_rate=0.001)
    
    results = predictor.evaluate_model()
    results['ic_values'] = ic_values  
    
    predictor.plot_results(results)
    
    predictor.save_predictions(results)
    
    print("Prediction task completed!")

if __name__ == "__main__":
    main()