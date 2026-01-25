"""
特征重要性评估模块
实验二中级要求 - 任务3：特征重要性评估模块

包括：
1. 自注意力机制计算基因级注意力权重
2. 集成梯度方法计算特征贡献度
3. 重要基因筛选（Top 50）
4. 注意力与梯度方法的结果对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import seaborn as sns


class SelfAttentionLayer(nn.Module):
    """
    自注意力层
    用于计算基因级别的注意力权重
    """
    def __init__(self, input_dim, hidden_dim=128, num_heads=4):
        super(SelfAttentionLayer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 多头注意力的Query, Key, Value投影
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        """
        前向传播
        x: (batch_size, input_dim)
        返回: output, attention_weights
        """
        batch_size = x.shape[0]
        
        # 添加序列维度 (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        
        # 计算 Q, K, V
        Q = self.query(x)  # (batch_size, 1, hidden_dim)
        K = self.key(x)
        V = self.value(x)
        
        # 重塑为多头形式
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力
        attended = torch.matmul(attention_weights, V)
        
        # 合并多头
        attended = attended.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_dim)
        
        # 输出投影
        output = self.out_proj(attended)
        output = output.squeeze(1)  # (batch_size, input_dim)
        
        # 残差连接和层归一化
        output = self.layer_norm(x.squeeze(1) + output)
        
        # 返回输出和注意力权重（取平均）
        attention_weights = attention_weights.mean(dim=1).squeeze(1)  # (batch_size, 1, 1)
        
        return output, attention_weights


class EncoderWithAttention(nn.Module):
    """
    带自注意力的编码器
    在预训练编码器后添加注意力层
    """
    def __init__(self, pretrained_encoder, input_dim, attention_hidden_dim=128):
        super(EncoderWithAttention, self).__init__()
        
        self.encoder = pretrained_encoder
        # 冻结编码器参数
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 添加注意力层（作用在输入上）
        self.attention = SelfAttentionLayer(
            input_dim=input_dim,
            hidden_dim=attention_hidden_dim,
            num_heads=4
        )
        
    def forward(self, x):
        """
        前向传播
        返回: encoded_features, attention_weights
        """
        # 先通过注意力层
        attended_x, attention_weights = self.attention(x)
        
        # 再通过编码器（编码器参数已冻结，但保留计算图以便梯度回传到注意力层）
        encoded = self.encoder(attended_x)
        
        return encoded, attention_weights
    
    def get_gene_attention_weights(self, x):
        """
        获取基因级别的注意力权重
        x: (batch_size, n_genes)
        返回: (batch_size, n_genes) 的注意力权重
        """
        _, attention_weights = self.attention(x)
        
        # 由于我们的注意力是在特征维度上，我们需要计算每个特征的重要性
        # 这里使用注意力权重作为特征重要性的代理
        with torch.no_grad():
            # 计算每个输入特征对输出的贡献
            x_attended, _ = self.attention(x)
            feature_importance = torch.abs(x_attended - x).mean(dim=0)
            
        return feature_importance


class IntegratedGradients:
    """
    集成梯度方法
    计算输入特征对模型输出的贡献度
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        
    def compute_gradients(self, x, target_class=None, n_steps=50):
        """
        计算集成梯度
        
        Args:
            x: 输入样本 (batch_size, n_features)
            target_class: 目标类别（如果是分类任务）
            n_steps: 积分步数
        
        Returns:
            integrated_gradients: (batch_size, n_features)
        """
        self.model.eval()
        
        # 确保输入不需要梯度
        x = x.detach()
        
        # 基线：全零输入
        baseline = torch.zeros_like(x).to(self.device)
        
        # 生成插值路径
        alphas = torch.linspace(0, 1, n_steps).to(self.device)
        
        # 存储梯度
        gradients = []
        
        for alpha in alphas:
            # 插值
            interpolated = baseline + alpha * (x - baseline)
            interpolated.requires_grad = True
            
            # 前向传播
            if isinstance(self.model, EncoderWithAttention):
                output, _ = self.model(interpolated)
            else:
                output = self.model(interpolated)
            
            # 如果是分类任务，选择目标类别
            if target_class is not None:
                if len(output.shape) > 1 and output.shape[1] > 1:
                    output = output[:, target_class].sum()
                else:
                    output = output.sum()
            else:
                output = output.sum()
            
            # 计算梯度
            gradient = torch.autograd.grad(output, interpolated, create_graph=False)[0]
            gradients.append(gradient)
        
        # 计算平均梯度
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # 集成梯度 = (x - baseline) * avg_gradients
        integrated_gradients = (x - baseline) * avg_gradients
        
        return integrated_gradients.detach()
    
    def get_feature_importance(self, dataloader, top_k=50):
        """
        计算所有样本的平均特征重要性
        
        Returns:
            importance_scores: (n_features,)
            top_features: top_k个最重要特征的索引
        """
        all_importances = []
        
        for batch_data in dataloader:
            if isinstance(batch_data, (list, tuple)):
                batch_data = batch_data[0]
            
            batch_data = batch_data.to(self.device)
            
            # 计算集成梯度
            ig = self.compute_gradients(batch_data)
            
            # 取绝对值作为重要性
            importance = torch.abs(ig)
            all_importances.append(importance.cpu())
        
        # 计算平均重要性
        all_importances = torch.cat(all_importances, dim=0)
        importance_scores = all_importances.mean(dim=0).numpy()
        
        # 获取top-k特征
        top_features = np.argsort(importance_scores)[-top_k:][::-1]
        
        return importance_scores, top_features


class FeatureImportanceEvaluator:
    """
    特征重要性评估器
    整合注意力和集成梯度两种方法
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.ig_calculator = IntegratedGradients(model, device)
        
    def evaluate_attention_importance(self, dataloader, top_k=50):
        """
        使用注意力机制评估特征重要性
        
        Returns:
            attention_scores: (n_features,)
            top_features: top_k个最重要特征的索引
        """
        if not isinstance(self.model, EncoderWithAttention):
            raise ValueError("模型必须是 EncoderWithAttention 类型")
        
        all_attention_weights = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_data in dataloader:
                if isinstance(batch_data, (list, tuple)):
                    batch_data = batch_data[0]
                
                batch_data = batch_data.to(self.device)
                
                # 获取注意力权重
                feature_importance = self.model.get_gene_attention_weights(batch_data)
                all_attention_weights.append(feature_importance.cpu())
        
        # 计算平均注意力权重
        attention_scores = torch.stack(all_attention_weights).mean(dim=0).numpy()
        
        # 获取top-k特征
        top_features = np.argsort(attention_scores)[-top_k:][::-1]
        
        return attention_scores, top_features
    
    def evaluate_gradient_importance(self, dataloader, top_k=50):
        """
        使用集成梯度评估特征重要性
        
        Returns:
            gradient_scores: (n_features,)
            top_features: top_k个最重要特征的索引
        """
        return self.ig_calculator.get_feature_importance(dataloader, top_k)
    
    def compare_methods(self, attention_top_features, gradient_top_features):
        """
        比较两种方法的结果
        
        Returns:
            overlap_rate: 重叠率
            overlap_features: 重叠的特征索引
        """
        attention_set = set(attention_top_features)
        gradient_set = set(gradient_top_features)
        
        overlap_features = attention_set & gradient_set
        overlap_rate = len(overlap_features) / len(attention_set)
        
        return overlap_rate, list(overlap_features)
    
    def visualize_importance(self, attention_scores, gradient_scores, 
                            top_k=50, save_path=None):
        """
        可视化特征重要性
        """
        # 获取top-k特征
        attention_top_k = np.argsort(attention_scores)[-top_k:][::-1]
        gradient_top_k = np.argsort(gradient_scores)[-top_k:][::-1]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 注意力权重分布
        axes[0, 0].hist(attention_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Attention Weight', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Attention Weight Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 梯度重要性分布
        axes[0, 1].hist(gradient_scores, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].set_xlabel('Gradient Importance', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Gradient Importance Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Top-k特征比较（条形图）
        x = np.arange(min(20, top_k))
        axes[1, 0].bar(x - 0.2, attention_scores[attention_top_k[:20]], 
                      0.4, label='Attention', alpha=0.7, color='blue')
        axes[1, 0].bar(x + 0.2, gradient_scores[gradient_top_k[:20]], 
                      0.4, label='Gradient', alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Feature Rank', fontsize=12)
        axes[1, 0].set_ylabel('Importance Score', fontsize=12)
        axes[1, 0].set_title('Top 20 Features Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 两种方法的相关性散点图
        axes[1, 1].scatter(attention_scores, gradient_scores, alpha=0.5, s=20)
        axes[1, 1].set_xlabel('Attention Weight', fontsize=12)
        axes[1, 1].set_ylabel('Gradient Importance', fontsize=12)
        axes[1, 1].set_title('Attention vs Gradient Correlation', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 计算相关系数
        correlation = np.corrcoef(attention_scores, gradient_scores)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[1, 1].transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征重要性可视化已保存: {save_path}")
        
        plt.close()
    
    def comprehensive_evaluation(self, dataloader, top_k=50, save_dir=None):
        """
        综合评估特征重要性
        
        Returns:
            results: 包含所有评估结果的字典
        """
        print("\n" + "="*60)
        print("特征重要性综合评估")
        print("="*60)
        
        # 1. 注意力方法
        print("\n1. 评估注意力权重...")
        attention_scores, attention_top_features = self.evaluate_attention_importance(
            dataloader, top_k
        )
        
        # 2. 集成梯度方法
        print("2. 评估集成梯度...")
        gradient_scores, gradient_top_features = self.evaluate_gradient_importance(
            dataloader, top_k
        )
        
        # 3. 比较两种方法
        print("3. 比较两种方法...")
        overlap_rate, overlap_features = self.compare_methods(
            attention_top_features, gradient_top_features
        )
        
        # 4. 可视化
        if save_dir:
            from pathlib import Path
            save_path = Path(save_dir) / 'feature_importance_comparison.png'
            self.visualize_importance(
                attention_scores, gradient_scores, top_k, save_path
            )
        
        # 打印结果
        print("\n" + "="*60)
        print("评估结果")
        print("="*60)
        print(f"\nTop-{top_k} 特征统计:")
        print(f"  注意力方法 - 平均重要性: {attention_scores[attention_top_features].mean():.4f}")
        print(f"  梯度方法 - 平均重要性: {gradient_scores[gradient_top_features].mean():.4f}")
        print(f"\n两种方法的重叠情况:")
        print(f"  重叠特征数: {len(overlap_features)}")
        print(f"  重叠率: {overlap_rate:.2%}")
        print(f"\n相关性:")
        correlation = np.corrcoef(attention_scores, gradient_scores)[0, 1]
        print(f"  Pearson相关系数: {correlation:.4f}")
        
        print("\n" + "="*60)
        
        results = {
            'attention_scores': attention_scores,
            'attention_top_features': attention_top_features,
            'gradient_scores': gradient_scores,
            'gradient_top_features': gradient_top_features,
            'overlap_rate': overlap_rate,
            'overlap_features': overlap_features,
            'correlation': correlation
        }
        
        return results


def train_encoder_with_attention(encoder, attention_model, dataloader, 
                                 optimizer, device, epochs=50):
    """
    训练带注意力的编码器
    只训练注意力层，编码器保持冻结
    """
    attention_model.train()
    criterion = nn.MSELoss()
    
    history = {'loss': []}
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_data in dataloader:
            if isinstance(batch_data, (list, tuple)):
                batch_data = batch_data[0]
            
            batch_data = batch_data.to(device)
            
            # 前向传播
            encoded, _ = attention_model(batch_data)
            
            # 重构损失（简单起见，使用编码特征的自监督任务）
            # 这里我们最小化编码表示的方差，促进注意力学习
            loss = -encoded.var(dim=0).mean()  # 最大化方差
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        history['loss'].append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return history
