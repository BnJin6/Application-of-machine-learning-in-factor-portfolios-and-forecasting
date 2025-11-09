#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalRegressionLoss(nn.Module):
    """
    Focal Loss for Regression
    基于Focal Loss的思想，对难预测的样本给予更多关注
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalRegressionLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        # 计算基础损失 (L1 loss)
        base_loss = F.l1_loss(predictions, targets, reduction='none')
        
        # 计算权重：损失越大，权重越大
        # 使用sigmoid函数将损失映射到[0,1]，然后应用focal weight
        normalized_loss = torch.sigmoid(base_loss)
        focal_weight = self.alpha * (normalized_loss ** self.gamma)
        
        # 应用focal权重
        focal_loss = focal_weight * base_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SoftClassificationLoss(nn.Module):
    """
    Soft Classification Loss for Regression
    将回归问题转换为软分类问题，使用交叉熵损失
    """
    def __init__(self, allowed_values, temperature=1.0, reduction='mean'):
        super(SoftClassificationLoss, self).__init__()
        self.allowed_values = torch.FloatTensor(allowed_values)
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        device = predictions.device
        self.allowed_values = self.allowed_values.to(device)
        
        # 确保输入是正确的形状
        predictions = predictions.flatten()
        targets = targets.flatten()
        batch_size = predictions.size(0)
        num_classes = len(self.allowed_values)
        
        # 为每个预测值计算到所有允许值的距离
        pred_expanded = predictions.unsqueeze(1).expand(batch_size, num_classes)
        allowed_expanded = self.allowed_values.unsqueeze(0).expand(batch_size, num_classes)
        
        # 计算距离（负距离，因为距离越小概率越大）
        distances = -torch.abs(pred_expanded - allowed_expanded)
        
        # 转换为概率分布（softmax with temperature）
        pred_probs = F.softmax(distances / self.temperature, dim=1)
        
        # 为目标值创建软标签
        target_expanded = targets.unsqueeze(1).expand(batch_size, num_classes)
        target_distances = -torch.abs(target_expanded - allowed_expanded)
        target_probs = F.softmax(target_distances / self.temperature, dim=1)
        
        # 计算KL散度损失（类似交叉熵）
        kl_loss = F.kl_div(pred_probs.log(), target_probs, reduction='none').sum(dim=1)
        
        if self.reduction == 'mean':
            return kl_loss.mean()
        elif self.reduction == 'sum':
            return kl_loss.sum()
        else:
            return kl_loss

class CrossEntropyRegressionLoss(nn.Module):
    """
    Cross-Entropy-like Regression Loss
    将连续值离散化，然后使用交叉熵损失
    """
    def __init__(self, allowed_values, sigma=0.5, reduction='mean'):
        super(CrossEntropyRegressionLoss, self).__init__()
        self.allowed_values = torch.FloatTensor(allowed_values)
        self.sigma = sigma  # 控制软化程度
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        device = predictions.device
        self.allowed_values = self.allowed_values.to(device)
        
        # 确保输入是正确的形状
        predictions = predictions.flatten()
        targets = targets.flatten()
        batch_size = predictions.size(0)
        num_classes = len(self.allowed_values)
        
        # 计算预测值的概率分布
        pred_expanded = predictions.unsqueeze(1).expand(batch_size, num_classes)
        allowed_expanded = self.allowed_values.unsqueeze(0).expand(batch_size, num_classes)
        
        # 使用高斯核计算概率
        pred_distances = torch.exp(-((pred_expanded - allowed_expanded) ** 2) / (2 * self.sigma ** 2))
        pred_probs = pred_distances / (pred_distances.sum(dim=1, keepdim=True) + 1e-8)
        
        # 计算目标值的概率分布
        target_expanded = targets.unsqueeze(1).expand(batch_size, num_classes)
        target_distances = torch.exp(-((target_expanded - allowed_expanded) ** 2) / (2 * self.sigma ** 2))
        target_probs = target_distances / (target_distances.sum(dim=1, keepdim=True) + 1e-8)
        
        # 计算交叉熵损失
        cross_entropy = -(target_probs * torch.log(pred_probs + 1e-8)).sum(dim=1)
        
        if self.reduction == 'mean':
            return cross_entropy.mean()
        elif self.reduction == 'sum':
            return cross_entropy.sum()
        else:
            return cross_entropy

class HybridLoss(nn.Module):
    """
    混合损失函数：结合MSE和类交叉熵损失
    """
    def __init__(self, allowed_values, mse_weight=0.5, ce_weight=0.5, sigma=0.5):
        super(HybridLoss, self).__init__()
        self.mse_weight = mse_weight
        self.ce_weight = ce_weight
        self.mse_loss = nn.MSELoss()
        self.ce_loss = CrossEntropyRegressionLoss(allowed_values, sigma=sigma)
        
    def forward(self, predictions, targets):
        mse = self.mse_loss(predictions, targets)
        ce = self.ce_loss(predictions, targets)
        
        return self.mse_weight * mse + self.ce_weight * ce

class GaussianNLLLoss(nn.Module):
    """
    高斯负对数似然损失函数
    L = (y - ŷ)² / (2σ²) + 1/2 * log(σ²)
    """
    def __init__(self, eps=1e-6):
        super(GaussianNLLLoss, self).__init__()
        self.eps = eps
    
    def forward(self, predictions, targets, variance=None):
        """
        Args:
            predictions: 预测值 (batch_size, 1)
            targets: 真实值 (batch_size, 1)
            variance: 方差 (batch_size, 1) 或标量，如果为None则使用预测误差的方差
        """
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        if variance is None:
            # 使用预测误差的方差作为估计
            residuals = predictions - targets
            variance = torch.var(residuals) + self.eps
            variance = variance.expand_as(predictions)
        else:
            variance = variance.flatten()
            variance = torch.clamp(variance, min=self.eps)  # 防止方差为0
        
        # 高斯负对数似然: (y - ŷ)² / (2σ²) + 1/2 * log(σ²)
        nll = 0.5 * ((predictions - targets) ** 2) / variance + 0.5 * torch.log(variance)
        return torch.mean(nll)

class ClassificationRegressionHybridLoss(nn.Module):
    """
    分类化+回归混合损失函数
    L = α * CrossEntropy(y_class, p) + (1 - α) * MSE(y, ŷ)
    
    将回归问题部分转化为分类问题，然后与回归损失结合
    """
    def __init__(self, num_classes=20, value_range=(-0.1, 0.1), alpha=0.5):
        super(ClassificationRegressionHybridLoss, self).__init__()
        self.num_classes = num_classes
        self.value_range = value_range
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 创建分类边界
        self.class_boundaries = torch.linspace(value_range[0], value_range[1], num_classes + 1)
    
    def _values_to_classes(self, values):
        """将连续值转换为类别标签"""
        values = values.flatten()
        classes = torch.zeros_like(values, dtype=torch.long)
        
        for i in range(self.num_classes):
            mask = (values >= self.class_boundaries[i]) & (values < self.class_boundaries[i + 1])
            classes[mask] = i
        
        # 处理边界情况
        classes[values >= self.class_boundaries[-1]] = self.num_classes - 1
        classes[values < self.class_boundaries[0]] = 0
        
        return classes
    
    def _predictions_to_class_probs(self, predictions):
        """将预测值转换为类别概率分布"""
        predictions = predictions.flatten()
        batch_size = predictions.size(0)
        
        # 使用高斯分布将预测值转换为类别概率
        class_centers = (self.class_boundaries[:-1] + self.class_boundaries[1:]) / 2
        class_centers = class_centers.to(predictions.device)
        
        # 计算每个预测值到各个类别中心的距离
        distances = torch.abs(predictions.unsqueeze(1) - class_centers.unsqueeze(0))
        
        # 使用softmax将距离转换为概率（距离越小概率越大）
        sigma = (self.class_boundaries[1] - self.class_boundaries[0]) / 2  # 类别宽度的一半作为sigma
        probs = torch.exp(-distances ** 2 / (2 * sigma ** 2))
        probs = probs / torch.sum(probs, dim=1, keepdim=True)  # 归一化
        
        return probs
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: 预测值 (batch_size, 1)
            targets: 真实值 (batch_size, 1)
        """
        # 回归损失
        regression_loss = self.mse_loss(predictions, targets)
        
        # 分类损失
        target_classes = self._values_to_classes(targets)
        pred_class_probs = self._predictions_to_class_probs(predictions)
        classification_loss = self.ce_loss(pred_class_probs, target_classes)
        
        # 混合损失
        total_loss = self.alpha * classification_loss + (1 - self.alpha) * regression_loss
        
        return total_loss

def test_loss_functions():
    """测试损失函数"""
    # 模拟数据
    batch_size = 32
    allowed_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    predictions = torch.randn(batch_size, 1) * 2
    targets = torch.randn(batch_size, 1) * 2
    
    print(f"预测值范围: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"目标值范围: [{targets.min():.3f}, {targets.max():.3f}]")
    print()
    
    # 测试不同损失函数
    losses = {
        'MSE': nn.MSELoss(),
        'SmoothL1': nn.SmoothL1Loss(),
        'Focal': FocalRegressionLoss(alpha=1.0, gamma=2.0),
        'SoftClassification': SoftClassificationLoss(allowed_values, temperature=1.0),
        'CrossEntropyRegression': CrossEntropyRegressionLoss(allowed_values, sigma=0.5),
        'Hybrid': HybridLoss(allowed_values, mse_weight=0.3, ce_weight=0.7)
    }
    
    for name, loss_fn in losses.items():
        loss_value = loss_fn(predictions, targets)
        print(f"{name:20s}: {loss_value.item():.6f}")
    
    print("\n 损失函数测试完成！")

if __name__ == "__main__":
    test_loss_functions()