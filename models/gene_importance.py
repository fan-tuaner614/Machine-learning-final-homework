"""
基因级特征重要性评估模块
Gene-Level Feature Importance Evaluation

实现指导书要求的：
1. 基因级注意力权重评估
2. 基因级集成梯度评估
3. Top-50基因筛选与重叠率计算
4. 多随机种子下的Jaccard稳定性评估
5. UMAP/t-SNE可视化
6. 注意力可视化与热图生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 中文字体支持
from utils.chinese_font import setup_chinese_font, get_chinese_font
setup_chinese_font()


class GeneAttentionLayer(nn.Module):
    """
    基因级自注意力层
    直接在2000维高变基因空间上计算注意力权重
    """
    def __init__(self, n_genes: int, hidden_dim: int = 256, num_heads: int = 4):
        super(GeneAttentionLayer, self).__init__()
        
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 基因嵌入层：将每个基因表达值映射到隐藏空间
        self.gene_embedding = nn.Linear(1, hidden_dim)
        
        # 多头自注意力
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, 1)
        
        # 基因级注意力权重（可学习）
        self.gene_attention_weights = nn.Parameter(torch.randn(n_genes) * 0.01)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        x: (batch_size, n_genes) - 基因表达矩阵
        返回: (weighted_output, attention_weights)
        """
        batch_size = x.shape[0]
        
        # 计算注意力权重（softmax归一化）
        attention_weights = F.softmax(self.gene_attention_weights, dim=0)  # (n_genes,)
        
        # 加权基因表达
        weighted_x = x * attention_weights.unsqueeze(0)  # (batch_size, n_genes)
        
        return weighted_x, attention_weights
    
    def get_attention_weights(self) -> np.ndarray:
        """获取归一化的注意力权重"""
        with torch.no_grad():
            weights = F.softmax(self.gene_attention_weights, dim=0)
            return weights.cpu().numpy()


class GeneEncoder(nn.Module):
    """
    基因级编码器
    输入：2000维高变基因表达
    输出：64维潜在表示
    """
    def __init__(self, n_genes: int = 2000, hidden_dims: List[int] = [512, 256, 128], 
                 latent_dim: int = 64, dropout: float = 0.1):
        super(GeneEncoder, self).__init__()
        
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        
        # 基因注意力层
        self.attention = GeneAttentionLayer(n_genes, hidden_dim=256)
        
        # 编码器网络
        layers = []
        prev_dim = n_genes
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        返回: (encoded_features, attention_weights)
        """
        # 应用基因注意力
        weighted_x, attention_weights = self.attention(x)
        
        # 编码
        encoded = self.encoder(weighted_x)
        
        return encoded, attention_weights
    
    def get_gene_attention_weights(self) -> np.ndarray:
        """获取基因级注意力权重"""
        return self.attention.get_attention_weights()


class GeneIntegratedGradients:
    """
    基因级集成梯度
    计算每个基因对模型输出的贡献度
    """
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        
    def compute_gradients(self, x: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        """
        计算集成梯度
        
        Args:
            x: (batch_size, n_genes) 输入基因表达
            n_steps: 积分步数
        
        Returns:
            integrated_gradients: (batch_size, n_genes)
        """
        self.model.eval()
        x = x.detach().to(self.device)
        
        # 基线：全零
        baseline = torch.zeros_like(x).to(self.device)
        
        # 积分路径
        alphas = torch.linspace(0, 1, n_steps).to(self.device)
        
        gradients = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (x - baseline)
            interpolated.requires_grad = True
            
            # 前向传播
            if hasattr(self.model, 'attention'):
                output, _ = self.model(interpolated)
            else:
                output = self.model(interpolated)
            
            # 对输出求和（用于计算标量梯度）
            output_sum = output.sum()
            
            # 计算梯度
            gradient = torch.autograd.grad(output_sum, interpolated, 
                                          create_graph=False, retain_graph=False)[0]
            gradients.append(gradient.detach())
        
        # 平均梯度
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # 集成梯度 = (x - baseline) * avg_gradients
        integrated_gradients = (x - baseline) * avg_gradients
        
        return integrated_gradients.cpu()
    
    def get_gene_importance(self, dataloader, top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算基因级重要性分数
        
        Returns:
            importance_scores: (n_genes,)
            top_genes: Top-k基因索引
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
            all_importances.append(importance)
        
        # 汇总所有样本
        all_importances = torch.cat(all_importances, dim=0)
        importance_scores = all_importances.mean(dim=0).numpy()
        
        # Top-k基因
        top_genes = np.argsort(importance_scores)[-top_k:][::-1]
        
        return importance_scores, top_genes


class GeneFeatureEvaluator:
    """
    基因级特征重要性评估器
    整合注意力、集成梯度、稳定性评估、可视化
    """
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.font_prop = get_chinese_font()
        
    def evaluate_with_attention(self, model: GeneEncoder, dataloader, 
                                top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用注意力机制评估基因重要性
        
        Returns:
            attention_scores: (n_genes,)
            top_genes: Top-k基因索引
        """
        attention_scores = model.get_gene_attention_weights()
        top_genes = np.argsort(attention_scores)[-top_k:][::-1]
        
        return attention_scores, top_genes
    
    def evaluate_with_gradient(self, model: GeneEncoder, dataloader,
                               top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用集成梯度评估基因重要性
        
        Returns:
            gradient_scores: (n_genes,)
            top_genes: Top-k基因索引
        """
        ig_calculator = GeneIntegratedGradients(model, self.device)
        return ig_calculator.get_gene_importance(dataloader, top_k)
    
    def compute_jaccard(self, set1: np.ndarray, set2: np.ndarray) -> float:
        """计算Jaccard相似度"""
        s1, s2 = set(set1), set(set2)
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return intersection / union if union > 0 else 0.0
    
    def evaluate_stability_multi_seed(self, model_class, X_train: np.ndarray, 
                                      seeds: List[int] = [0, 1, 2, 3, 4],
                                      top_k: int = 50, epochs: int = 30,
                                      method: str = 'attention') -> Dict:
        """
        多随机种子下的稳定性评估
        
        Args:
            model_class: 模型类
            X_train: 训练数据 (n_samples, n_genes)
            seeds: 随机种子列表
            top_k: Top-k基因数量
            epochs: 训练轮数
            method: 'attention' 或 'gradient'
        
        Returns:
            stability_results: 包含Jaccard矩阵、均值、标准差的字典
        """
        from torch.utils.data import DataLoader, TensorDataset
        
        all_top_genes = []
        
        for seed in seeds:
            # 设置随机种子
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # 创建模型
            n_genes = X_train.shape[1]
            model = model_class(n_genes=n_genes).to(self.device)
            
            # 准备数据
            X_tensor = torch.FloatTensor(X_train)
            dataset = TensorDataset(X_tensor)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            # 训练模型
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model.train()
            
            for epoch in range(epochs):
                for batch_data, in dataloader:
                    batch_data = batch_data.to(self.device)
                    
                    # 前向传播
                    encoded, _ = model(batch_data)
                    
                    # 训练目标：最大化编码特征的方差
                    loss = -encoded.var(dim=0).mean()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # 评估Top-k基因
            model.eval()
            if method == 'attention':
                _, top_genes = self.evaluate_with_attention(model, dataloader, top_k)
            else:
                _, top_genes = self.evaluate_with_gradient(model, dataloader, top_k)
            
            all_top_genes.append(top_genes)
        
        # 计算两两Jaccard
        n_seeds = len(seeds)
        jaccard_matrix = np.zeros((n_seeds, n_seeds))
        
        for i in range(n_seeds):
            for j in range(n_seeds):
                jaccard_matrix[i, j] = self.compute_jaccard(
                    all_top_genes[i], all_top_genes[j]
                )
        
        # 取上三角（不含对角线）的Jaccard值
        upper_triangle = jaccard_matrix[np.triu_indices(n_seeds, k=1)]
        
        return {
            'jaccard_matrix': jaccard_matrix,
            'jaccard_mean': float(upper_triangle.mean()),
            'jaccard_std': float(upper_triangle.std()),
            'all_top_genes': all_top_genes,
            'seeds': seeds
        }
    
    def visualize_attention_weights(self, attention_scores: np.ndarray, 
                                    gene_names: Optional[List[str]] = None,
                                    top_k: int = 50, save_path: Optional[str] = None):
        """
        可视化注意力权重分布
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. 全部基因的注意力权重分布
        axes[0].hist(attention_scores, bins=50, color='steelblue', 
                    edgecolor='white', alpha=0.8)
        axes[0].set_xlabel('注意力权重', fontsize=12, fontproperties=self.font_prop)
        axes[0].set_ylabel('基因数量', fontsize=12, fontproperties=self.font_prop)
        axes[0].set_title('基因注意力权重分布', fontsize=14, 
                         fontweight='bold', fontproperties=self.font_prop)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Top-k基因的权重条形图
        top_indices = np.argsort(attention_scores)[-top_k:][::-1]
        top_scores = attention_scores[top_indices]
        
        if gene_names is not None:
            top_names = [gene_names[i] for i in top_indices[:20]]
        else:
            top_names = [f'Gene_{i}' for i in top_indices[:20]]
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, 20))
        axes[1].barh(range(20), top_scores[:20][::-1], color=colors[::-1])
        axes[1].set_yticks(range(20))
        axes[1].set_yticklabels(top_names[::-1], fontsize=8)
        axes[1].set_xlabel('注意力权重', fontsize=12, fontproperties=self.font_prop)
        axes[1].set_title('Top-20 重要基因', fontsize=14, 
                         fontweight='bold', fontproperties=self.font_prop)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # 3. 累积重要性曲线
        sorted_scores = np.sort(attention_scores)[::-1]
        cumsum = np.cumsum(sorted_scores)
        cumsum_normalized = cumsum / cumsum[-1]
        
        axes[2].plot(range(1, len(cumsum)+1), cumsum_normalized, 
                    color='darkblue', linewidth=2)
        axes[2].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                       label='50% 累积权重')
        axes[2].axhline(y=0.8, color='orange', linestyle='--', alpha=0.7,
                       label='80% 累积权重')
        
        # 找到达到50%和80%的基因数
        n_50 = np.searchsorted(cumsum_normalized, 0.5) + 1
        n_80 = np.searchsorted(cumsum_normalized, 0.8) + 1
        axes[2].axvline(x=n_50, color='red', linestyle=':', alpha=0.5)
        axes[2].axvline(x=n_80, color='orange', linestyle=':', alpha=0.5)
        
        axes[2].set_xlabel('基因数量（按重要性排序）', fontsize=12, fontproperties=self.font_prop)
        axes[2].set_ylabel('累积权重比例', fontsize=12, fontproperties=self.font_prop)
        axes[2].set_title(f'累积注意力分布\n(Top-{n_50}基因占50%, Top-{n_80}基因占80%)', 
                         fontsize=14, fontweight='bold', fontproperties=self.font_prop)
        axes[2].legend(fontsize=10, prop=self.font_prop)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"注意力可视化已保存: {save_path}")
        plt.close()
    
    def visualize_importance_heatmap(self, attention_scores: np.ndarray,
                                     gradient_scores: np.ndarray,
                                     gene_names: Optional[List[str]] = None,
                                     top_k: int = 50, save_path: Optional[str] = None):
        """
        可视化Top基因重要性热图
        """
        # 获取两种方法的Top基因
        attn_top = np.argsort(attention_scores)[-top_k:][::-1]
        grad_top = np.argsort(gradient_scores)[-top_k:][::-1]
        
        # 合并所有重要基因（取并集）
        all_important = list(set(attn_top) | set(grad_top))
        all_important = sorted(all_important, 
                              key=lambda x: attention_scores[x] + gradient_scores[x], 
                              reverse=True)[:top_k]
        
        # 创建热图数据
        heatmap_data = np.zeros((len(all_important), 2))
        for i, gene_idx in enumerate(all_important):
            heatmap_data[i, 0] = attention_scores[gene_idx]
            heatmap_data[i, 1] = gradient_scores[gene_idx]
        
        # 归一化
        heatmap_data[:, 0] = heatmap_data[:, 0] / heatmap_data[:, 0].max()
        heatmap_data[:, 1] = heatmap_data[:, 1] / heatmap_data[:, 1].max()
        
        # 基因名称
        if gene_names is not None:
            row_labels = [gene_names[i] for i in all_important]
        else:
            row_labels = [f'Gene_{i}' for i in all_important]
        
        # 绘制热图
        fig, ax = plt.subplots(figsize=(8, max(12, top_k * 0.3)))
        
        sns.heatmap(heatmap_data, annot=False, cmap='YlOrRd',
                   xticklabels=['注意力权重', '梯度重要性'],
                   yticklabels=row_labels, ax=ax,
                   cbar_kws={'label': '归一化重要性分数'})
        
        ax.set_title(f'Top-{top_k} 重要基因热图', fontsize=14, 
                    fontweight='bold', fontproperties=self.font_prop)
        ax.set_xlabel('评估方法', fontsize=12, fontproperties=self.font_prop)
        ax.set_ylabel('基因', fontsize=12, fontproperties=self.font_prop)
        
        # 标记两种方法都识别的基因
        overlap = set(attn_top) & set(grad_top)
        for i, gene_idx in enumerate(all_important):
            if gene_idx in overlap:
                ax.get_yticklabels()[i].set_color('red')
                ax.get_yticklabels()[i].set_fontweight('bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"重要性热图已保存: {save_path}")
        plt.close()
    
    def visualize_umap_tsne(self, features: np.ndarray, labels: np.ndarray,
                           method: str = 'tsne', label_names: Optional[List[str]] = None,
                           perplexity: int = 30, n_neighbors: int = 15,
                           save_path: Optional[str] = None, title_suffix: str = ''):
        """
        UMAP/t-SNE降维可视化
        
        Args:
            features: (n_samples, n_features) 特征矩阵
            labels: (n_samples,) 标签
            method: 'tsne' 或 'umap'
            label_names: 标签名称列表
            perplexity: t-SNE的perplexity参数
            n_neighbors: UMAP的n_neighbors参数
            save_path: 保存路径
            title_suffix: 标题后缀
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, perplexity=perplexity, 
                          random_state=42, n_iter=1000)
            embedding = reducer.fit_transform(features)
            method_name = f't-SNE (perplexity={perplexity})'
        else:  # UMAP
            try:
                import umap
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, 
                                   random_state=42)
                embedding = reducer.fit_transform(features)
                method_name = f'UMAP (n_neighbors={n_neighbors}, min_dist=0.1)'
            except ImportError:
                print("UMAP未安装，使用t-SNE替代")
                reducer = TSNE(n_components=2, perplexity=perplexity, 
                              random_state=42, n_iter=1000)
                embedding = reducer.fit_transform(features)
                method_name = f't-SNE (perplexity={perplexity})'
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_str = label_names[label] if label_names else f'类别 {label}'
            ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                      c=[colors[i]], label=label_str, alpha=0.7, s=30)
        
        ax.set_xlabel(f'{method.upper()} 1', fontsize=12, fontproperties=self.font_prop)
        ax.set_ylabel(f'{method.upper()} 2', fontsize=12, fontproperties=self.font_prop)
        ax.set_title(f'细胞表征{method.upper()}可视化{title_suffix}\n({method_name})', 
                    fontsize=14, fontweight='bold', fontproperties=self.font_prop)
        ax.legend(fontsize=10, prop=self.font_prop, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"{method.upper()}可视化已保存: {save_path}")
        plt.close()
        
        return embedding
    
    def visualize_stability_boxplot(self, stability_results: Dict, 
                                   save_path: Optional[str] = None):
        """
        可视化稳定性评估结果（箱线图）
        """
        jaccard_matrix = stability_results['jaccard_matrix']
        seeds = stability_results['seeds']
        n_seeds = len(seeds)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Jaccard矩阵热图
        mask = np.triu(np.ones_like(jaccard_matrix, dtype=bool), k=1)
        sns.heatmap(jaccard_matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                   xticklabels=[f'Seed {s}' for s in seeds],
                   yticklabels=[f'Seed {s}' for s in seeds],
                   ax=axes[0], vmin=0, vmax=1, mask=~mask & ~np.eye(n_seeds, dtype=bool))
        axes[0].set_title('Top-50基因Jaccard相似度矩阵', fontsize=14, 
                         fontweight='bold', fontproperties=self.font_prop)
        
        # 2. 箱线图
        upper_triangle = jaccard_matrix[np.triu_indices(n_seeds, k=1)]
        axes[1].boxplot(upper_triangle, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', color='blue'),
                       medianprops=dict(color='red', linewidth=2))
        axes[1].axhline(y=stability_results['jaccard_mean'], color='green', 
                       linestyle='--', linewidth=2, label=f"均值: {stability_results['jaccard_mean']:.3f}")
        axes[1].set_ylabel('Jaccard相似度', fontsize=12, fontproperties=self.font_prop)
        axes[1].set_title(f"稳定性评估: Jaccard = {stability_results['jaccard_mean']:.3f} ± {stability_results['jaccard_std']:.3f}",
                         fontsize=14, fontweight='bold', fontproperties=self.font_prop)
        axes[1].legend(fontsize=10, prop=self.font_prop)
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"稳定性可视化已保存: {save_path}")
        plt.close()


def comprehensive_gene_evaluation(X_genes: np.ndarray, labels: np.ndarray,
                                 gene_names: Optional[List[str]] = None,
                                 label_names: Optional[List[str]] = None,
                                 top_k: int = 50, seeds: List[int] = [0, 1, 2, 3, 4],
                                 output_dir: str = 'outputs_intermediate/plots',
                                 device: str = 'cpu') -> Dict:
    """
    基因级特征重要性综合评估
    
    Args:
        X_genes: (n_samples, n_genes) 高变基因表达矩阵
        labels: (n_samples,) 细胞标签
        gene_names: 基因名称列表
        label_names: 标签名称列表
        top_k: Top-k基因数量
        seeds: 随机种子列表
        output_dir: 输出目录
        device: 计算设备
    
    Returns:
        results: 综合评估结果字典
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples, n_genes = X_genes.shape
    print(f"\n基因级特征重要性评估")
    print(f"  样本数: {n_samples}")
    print(f"  基因数: {n_genes}")
    print(f"  Top-k: {top_k}")
    print(f"  随机种子: {seeds}")
    
    # 初始化评估器
    evaluator = GeneFeatureEvaluator(device=device)
    
    # 1. 训练基因编码器（使用第一个种子）
    print("\n[1] 训练基因级编码器...")
    torch.manual_seed(seeds[0])
    np.random.seed(seeds[0])
    
    model = GeneEncoder(n_genes=n_genes).to(device)
    
    X_tensor = torch.FloatTensor(X_genes)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    for epoch in range(50):
        total_loss = 0
        for batch_data, in dataloader:
            batch_data = batch_data.to(device)
            encoded, _ = model(batch_data)
            loss = -encoded.var(dim=0).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/50], Loss: {total_loss/len(dataloader):.4f}")
    
    model.eval()
    
    # 2. 计算注意力权重
    print("\n[2] 计算注意力权重...")
    attention_scores, attention_top = evaluator.evaluate_with_attention(model, dataloader, top_k)
    
    # 3. 计算集成梯度
    print("[3] 计算集成梯度...")
    gradient_scores, gradient_top = evaluator.evaluate_with_gradient(model, dataloader, top_k)
    
    # 4. 计算重叠率
    overlap = set(attention_top) & set(gradient_top)
    jaccard_overlap = evaluator.compute_jaccard(attention_top, gradient_top)
    
    print(f"\n[4] Top-{top_k}基因重叠分析:")
    print(f"  注意力方法: {attention_top[:10]}... (前10个)")
    print(f"  梯度方法: {gradient_top[:10]}... (前10个)")
    print(f"  重叠基因数: {len(overlap)}")
    print(f"  Jaccard重叠率: {jaccard_overlap:.3f}")
    
    # 5. 多种子稳定性评估
    print(f"\n[5] 多随机种子稳定性评估 (seeds={seeds})...")
    stability_attention = evaluator.evaluate_stability_multi_seed(
        GeneEncoder, X_genes, seeds=seeds, top_k=top_k, epochs=30, method='attention'
    )
    stability_gradient = evaluator.evaluate_stability_multi_seed(
        GeneEncoder, X_genes, seeds=seeds, top_k=top_k, epochs=30, method='gradient'
    )
    
    print(f"  注意力方法稳定性: Jaccard = {stability_attention['jaccard_mean']:.3f} ± {stability_attention['jaccard_std']:.3f}")
    print(f"  梯度方法稳定性: Jaccard = {stability_gradient['jaccard_mean']:.3f} ± {stability_gradient['jaccard_std']:.3f}")
    
    # 6. 可视化
    print("\n[6] 生成可视化...")
    
    # 6.1 注意力权重可视化
    evaluator.visualize_attention_weights(
        attention_scores, gene_names, top_k,
        save_path=output_dir / 'gene_attention_weights.png'
    )
    
    # 6.2 重要性热图
    evaluator.visualize_importance_heatmap(
        attention_scores, gradient_scores, gene_names, top_k,
        save_path=output_dir / 'gene_importance_heatmap.png'
    )
    
    # 6.3 稳定性箱线图
    evaluator.visualize_stability_boxplot(
        stability_attention,
        save_path=output_dir / 'stability_attention_boxplot.png'
    )
    evaluator.visualize_stability_boxplot(
        stability_gradient,
        save_path=output_dir / 'stability_gradient_boxplot.png'
    )
    
    # 6.4 t-SNE可视化（使用编码器输出）
    with torch.no_grad():
        X_tensor_full = torch.FloatTensor(X_genes).to(device)
        features, _ = model(X_tensor_full)
        features = features.cpu().numpy()
    
    evaluator.visualize_umap_tsne(
        features, labels, method='tsne', label_names=label_names,
        perplexity=30, save_path=output_dir / 'gene_encoder_tsne.png',
        title_suffix=' (基因编码器)'
    )
    
    # 汇总结果
    results = {
        'attention_scores': attention_scores,
        'attention_top_genes': attention_top,
        'gradient_scores': gradient_scores,
        'gradient_top_genes': gradient_top,
        'overlap_genes': list(overlap),
        'jaccard_overlap': jaccard_overlap,
        'stability_attention': stability_attention,
        'stability_gradient': stability_gradient,
        'n_genes': n_genes,
        'n_samples': n_samples,
        'top_k': top_k,
        'seeds': seeds
    }
    
    # 保存结果 - 确保所有数据类型可序列化
    results_to_save = {
        'attention_top_genes': [int(g) for g in attention_top.tolist()],
        'gradient_top_genes': [int(g) for g in gradient_top.tolist()],
        'overlap_genes': [int(g) for g in overlap],  # 转换为Python int
        'jaccard_overlap': float(jaccard_overlap),
        'stability_attention_mean': float(stability_attention['jaccard_mean']),
        'stability_attention_std': float(stability_attention['jaccard_std']),
        'stability_gradient_mean': float(stability_gradient['jaccard_mean']),
        'stability_gradient_std': float(stability_gradient['jaccard_std']),
        'n_genes': int(n_genes),
        'n_samples': int(n_samples),
        'top_k': int(top_k),
        'seeds': [int(s) for s in seeds]
    }
    
    import json
    with open(output_dir / 'gene_importance_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=4, ensure_ascii=False)
    
    print(f"\n基因级评估结果已保存: {output_dir / 'gene_importance_results.json'}")
    
    return results
