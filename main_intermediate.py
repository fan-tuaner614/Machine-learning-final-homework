"""
中级实验主脚本 (完整版)
实验二的中级要求：对比学习与特征重要性评估

实现内容：
1. SimCLR对比学习预训练
2. 基因级特征重要性评估（注意力 + 集成梯度）
3. Top-50基因筛选与重叠率计算
4. 多随机种子Jaccard稳定性评估
5. UMAP/t-SNE可视化
6. 注意力权重分布可视化
7. 特征重要性热图

输入空间说明：
- PCA特征（50维）：用于SimCLR对比学习
- 基因表达（2000维HVG）：用于基因级特征重要性评估
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体支持
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from utils.chinese_font import setup_chinese_font, get_chinese_font
setup_chinese_font()

# 导入模块
from src.data_loader import SingleCellDataLoader
from src.preprocessing import SingleCellPreprocessor
from models.pretrain import SimCLR, SingleCellDataset, train_simclr
from models.feature_importance import (
    EncoderWithAttention,
    FeatureImportanceEvaluator,
    train_encoder_with_attention
)
from models.gene_importance import (
    GeneEncoder,
    GeneFeatureEvaluator,
    comprehensive_gene_evaluation
)
from utils.visualization import Visualizer
from config import Config
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


class IntermediateExperiment:
    """中级实验类（完整版）"""
    
    def __init__(self, config=None):
        self.config = config if config else Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path('outputs_intermediate')
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        
        # 初始化模块
        self.data_loader = SingleCellDataLoader(
            self.config.data_dir if hasattr(self.config, 'data_dir') else 'data'
        )
        self.preprocessor = SingleCellPreprocessor(
            min_genes=self.config.min_genes if hasattr(self.config, 'min_genes') else 200,
            min_cells=self.config.min_cells if hasattr(self.config, 'min_cells') else 3,
            n_top_genes=self.config.n_top_genes if hasattr(self.config, 'n_top_genes') else 2000,
            n_pca_components=self.config.n_pca_components if hasattr(self.config, 'n_pca_components') else 50
        )
        self.visualizer = Visualizer(self.output_dir / 'plots')
        self.font_prop = get_chinese_font()
        
        # 结果存储
        self.results = {}
    
    def step1_simclr_pretrain(self):
        """
        步骤1：SimCLR对比学习预训练
        使用PCA特征（50维）进行对比学习
        """
        print("\n" + "="*70)
        print("步骤1: SimCLR对比学习预训练")
        print("="*70)
        
        # 加载SC-2数据
        datasets = self.data_loader.load_all_data()
        
        if 'SC-2_dense' not in datasets:
            raise ValueError("未找到SC-2_dense数据集")
        
        adata = datasets['SC-2_dense']
        
        # 预处理
        print("\n预处理数据...")
        adata_processed = self.preprocessor.preprocess(adata)
        
        # 获取PCA特征（用于SimCLR）
        X_pca = adata_processed.obsm['X_pca']
        pca_dim = X_pca.shape[1]
        
        # 获取高变基因表达（用于基因级分析）
        X_hvg = adata_processed.X
        n_genes = X_hvg.shape[1]
        
        # 获取标签
        if 'subtype' in adata_processed.obs.columns:
            le = LabelEncoder()
            labels = le.fit_transform(adata_processed.obs['subtype'])
            label_names = list(le.classes_)
        else:
            labels = np.zeros(len(X_pca), dtype=int)
            label_names = ['Unknown']
        
        # 获取基因名称
        if hasattr(adata_processed, 'var_names'):
            gene_names = list(adata_processed.var_names)
        else:
            gene_names = [f'Gene_{i}' for i in range(n_genes)]
        
        print(f"\n数据信息:")
        print(f"  样本数量: {len(X_pca)}")
        print(f"  PCA维度: {pca_dim} (用于SimCLR)")
        print(f"  高变基因数: {n_genes} (用于基因级分析)")
        print(f"  细胞亚型数: {len(np.unique(labels))}")
        
        # 保存数据
        self.X_pca = X_pca
        self.X_hvg = X_hvg
        self.labels = labels
        self.label_names = label_names
        self.gene_names = gene_names
        self.adata_processed = adata_processed
        self.pca_dim = pca_dim
        self.n_genes = n_genes
        
        # 创建数据集和数据加载器（带数据增强）
        dataset = SingleCellDataset(X_pca, augment=True)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # 创建SimCLR模型
        model = SimCLR(
            input_dim=pca_dim,
            hidden_dims=[256, 128],
            latent_dim=64,
            projection_dim=64
        ).to(self.device)
        
        # 训练
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print("\n开始训练SimCLR...")
        history = train_simclr(
            model, dataloader, optimizer, self.device, 
            epochs=100, temperature=0.5
        )
        
        # 保存模型
        model_path = self.output_dir / 'models' / 'pretrained_simclr.pth'
        torch.save(model.state_dict(), model_path)
        print(f"\nSimCLR模型已保存: {model_path}")
        
        # 保存训练历史
        history_path = self.output_dir / 'simclr_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        # 绘制训练曲线
        self._plot_simclr_loss(history)
        
        self.simclr_model = model
        self.simclr_encoder = model.encoder
        
        return model
    
    def step2_pca_feature_importance(self):
        """
        步骤2：PCA维度上的特征重要性评估
        （保留原有功能，作为对比）
        """
        print("\n" + "="*70)
        print("步骤2: PCA特征重要性评估 (50维)")
        print("="*70)
        
        # 创建带注意力的编码器
        attention_model = EncoderWithAttention(
            pretrained_encoder=self.simclr_encoder,
            input_dim=self.pca_dim,
            attention_hidden_dim=128
        ).to(self.device)
        
        print(f"\n模型结构:")
        print(f"  编码器参数（冻结）: {sum(p.numel() for p in attention_model.encoder.parameters()):,}")
        print(f"  注意力层参数（可训练）: {sum(p.numel() for p in attention_model.attention.parameters()):,}")
        
        # 创建数据加载器
        dataset = SingleCellDataset(self.X_pca, augment=False)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # 训练注意力层
        optimizer = optim.Adam(attention_model.attention.parameters(), lr=0.001)
        print("\n训练注意力层...")
        history = train_encoder_with_attention(
            self.simclr_encoder, attention_model, dataloader,
            optimizer, self.device, epochs=50
        )
        
        # 保存模型
        model_path = self.output_dir / 'models' / 'encoder_with_attention.pth'
        torch.save(attention_model.state_dict(), model_path)
        
        # 特征重要性评估
        eval_dataloader = DataLoader(
            SingleCellDataset(self.X_pca, augment=False), 
            batch_size=32, shuffle=False
        )
        
        evaluator = FeatureImportanceEvaluator(
            model=attention_model,
            device=self.device
        )
        
        results = evaluator.comprehensive_evaluation(
            dataloader=eval_dataloader,
            top_k=50,
            save_dir=self.output_dir / 'plots'
        )
        
        self.pca_importance_results = results
        self.attention_model = attention_model
        
        return results
    
    def step3_gene_level_importance(self):
        """
        步骤3：基因级特征重要性评估（核心任务）
        
        输入：2000维高变基因表达
        输出：
        - Top-50基因列表（注意力 + 集成梯度）
        - 两种方法的Jaccard重叠率
        - 多随机种子稳定性评估
        """
        print("\n" + "="*70)
        print("步骤3: 基因级特征重要性评估 (2000维高变基因)")
        print("="*70)
        
        # 运行综合基因级评估
        results = comprehensive_gene_evaluation(
            X_genes=self.X_hvg,
            labels=self.labels,
            gene_names=self.gene_names,
            label_names=self.label_names,
            top_k=50,
            seeds=[0, 1, 2, 3, 4],
            output_dir=str(self.output_dir / 'plots'),
            device=str(self.device)
        )
        
        self.gene_importance_results = results
        
        # 打印Top-50基因
        print("\n" + "="*60)
        print("Top-50 重要基因 (注意力方法)")
        print("="*60)
        for i, gene_idx in enumerate(results['attention_top_genes'][:20]):
            gene_name = self.gene_names[gene_idx] if self.gene_names else f'Gene_{gene_idx}'
            score = results['attention_scores'][gene_idx]
            print(f"  {i+1:2d}. {gene_name:<20s} 权重: {score:.6f}")
        print("  ...")
        
        print("\n" + "="*60)
        print("Top-50 重要基因 (集成梯度方法)")
        print("="*60)
        for i, gene_idx in enumerate(results['gradient_top_genes'][:20]):
            gene_name = self.gene_names[gene_idx] if self.gene_names else f'Gene_{gene_idx}'
            score = results['gradient_scores'][gene_idx]
            print(f"  {i+1:2d}. {gene_name:<20s} 梯度: {score:.6f}")
        print("  ...")
        
        return results
    
    def step4_tsne_visualization(self):
        """
        步骤4：UMAP/t-SNE降维可视化
        在不同特征表示上进行可视化
        """
        print("\n" + "="*70)
        print("步骤4: 特征表示降维可视化 (t-SNE)")
        print("="*70)
        
        evaluator = GeneFeatureEvaluator(device=str(self.device))
        
        # 1. SimCLR编码器输出的t-SNE
        print("\n生成SimCLR编码器特征的t-SNE...")
        self.simclr_encoder.eval()
        with torch.no_grad():
            X_pca_tensor = torch.FloatTensor(self.X_pca).to(self.device)
            simclr_features = self.simclr_encoder(X_pca_tensor).cpu().numpy()
        
        evaluator.visualize_umap_tsne(
            simclr_features, self.labels, method='tsne',
            label_names=self.label_names, perplexity=30,
            save_path=self.output_dir / 'plots' / 'simclr_encoder_tsne.png',
            title_suffix=' (SimCLR编码器, 64维)'
        )
        
        # 2. 原始PCA特征的t-SNE
        print("生成原始PCA特征的t-SNE...")
        evaluator.visualize_umap_tsne(
            self.X_pca, self.labels, method='tsne',
            label_names=self.label_names, perplexity=30,
            save_path=self.output_dir / 'plots' / 'pca_features_tsne.png',
            title_suffix=' (PCA, 50维)'
        )
        
        # 3. 原始HVG特征的t-SNE（降采样以加速）
        print("生成原始高变基因特征的t-SNE...")
        evaluator.visualize_umap_tsne(
            self.X_hvg, self.labels, method='tsne',
            label_names=self.label_names, perplexity=30,
            save_path=self.output_dir / 'plots' / 'hvg_features_tsne.png',
            title_suffix=' (高变基因, 2000维)'
        )
        
        print("\nt-SNE可视化完成！")
    
    def step5_save_summary(self):
        """
        步骤5：保存综合结果摘要
        """
        print("\n" + "="*70)
        print("步骤5: 保存实验结果摘要")
        print("="*70)
        
        summary = {
            'experiment': '中级实验：对比学习与基因级特征重要性评估',
            'data': {
                'n_samples': len(self.X_pca),
                'n_pca_dims': self.pca_dim,
                'n_hvg_genes': self.n_genes,
                'n_cell_types': len(np.unique(self.labels))
            },
            'pca_importance': {
                'top_50_pca_features_attention': self.pca_importance_results['attention_top_features'].tolist(),
                'top_50_pca_features_gradient': self.pca_importance_results['gradient_top_features'].tolist(),
                'overlap_rate': self.pca_importance_results['overlap_rate'],
                'correlation': self.pca_importance_results['correlation']
            },
            'gene_importance': {
                'top_50_genes_attention': self.gene_importance_results['attention_top_genes'].tolist(),
                'top_50_genes_gradient': self.gene_importance_results['gradient_top_genes'].tolist(),
                'jaccard_overlap': self.gene_importance_results['jaccard_overlap'],
                'n_overlap_genes': len(self.gene_importance_results['overlap_genes']),
                'stability_attention_mean': self.gene_importance_results['stability_attention']['jaccard_mean'],
                'stability_attention_std': self.gene_importance_results['stability_attention']['jaccard_std'],
                'stability_gradient_mean': self.gene_importance_results['stability_gradient']['jaccard_mean'],
                'stability_gradient_std': self.gene_importance_results['stability_gradient']['jaccard_std']
            }
        }
        
        # 保存摘要
        summary_path = self.output_dir / 'experiment_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        print(f"\n实验摘要已保存: {summary_path}")
        
        # 打印关键结果
        print("\n" + "="*60)
        print("关键实验结果")
        print("="*60)
        print(f"\n【基因级特征重要性】")
        print(f"  Top-50基因Jaccard重叠率: {summary['gene_importance']['jaccard_overlap']:.3f}")
        print(f"  重叠基因数: {summary['gene_importance']['n_overlap_genes']}")
        print(f"\n【多随机种子稳定性 (Jaccard)】")
        print(f"  注意力方法: {summary['gene_importance']['stability_attention_mean']:.3f} ± {summary['gene_importance']['stability_attention_std']:.3f}")
        print(f"  梯度方法: {summary['gene_importance']['stability_gradient_mean']:.3f} ± {summary['gene_importance']['stability_gradient_std']:.3f}")
        
        return summary
    
    def _plot_simclr_loss(self, history):
        """绘制SimCLR损失曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history['loss']) + 1)
        ax.plot(epochs, history['loss'], 'b-', linewidth=2, label='对比损失 (InfoNCE)')
        
        ax.set_xlabel('Epoch', fontsize=12, fontproperties=self.font_prop)
        ax.set_ylabel('对比损失', fontsize=12, fontproperties=self.font_prop)
        ax.set_title('SimCLR对比学习训练曲线', fontsize=14, 
                    fontweight='bold', fontproperties=self.font_prop)
        ax.legend(fontsize=10, prop=self.font_prop)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'plots' / 'simclr_training_loss.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线已保存: {save_path}")
    
    def run_full_experiment(self):
        """
        运行完整的中级实验
        """
        print("\n" + "="*70)
        print(" " * 10 + "中级实验：对比学习与基因级特征重要性评估")
        print("="*70)
        
        # 步骤1：SimCLR预训练
        self.step1_simclr_pretrain()
        
        # 步骤2：PCA特征重要性（保留作为对比）
        self.step2_pca_feature_importance()
        
        # 步骤3：基因级特征重要性（核心任务）
        self.step3_gene_level_importance()
        
        # 步骤4：t-SNE可视化
        self.step4_tsne_visualization()
        
        # 步骤5：保存摘要
        self.step5_save_summary()
        
        print("\n" + "="*70)
        print(" " * 20 + "实验完成！")
        print(f" " * 15 + f"所有结果保存在: {self.output_dir}")
        print("="*70)
        
        # 输出文件列表
        print("\n生成的文件:")
        print("  模型:")
        print("    - models/pretrained_simclr.pth")
        print("    - models/encoder_with_attention.pth")
        print("  可视化:")
        print("    - plots/simclr_training_loss.png")
        print("    - plots/feature_importance_comparison.png")
        print("    - plots/gene_attention_weights.png")
        print("    - plots/gene_importance_heatmap.png")
        print("    - plots/stability_attention_boxplot.png")
        print("    - plots/stability_gradient_boxplot.png")
        print("    - plots/gene_encoder_tsne.png")
        print("    - plots/simclr_encoder_tsne.png")
        print("    - plots/pca_features_tsne.png")
        print("    - plots/hvg_features_tsne.png")
        print("  结果:")
        print("    - simclr_history.json")
        print("    - feature_importance_results.json")
        print("    - gene_importance_results.json")
        print("    - experiment_summary.json")


def main():
    """主函数"""
    # 创建配置
    config = Config()
    
    # 创建实验实例
    experiment = IntermediateExperiment(config)
    
    # 运行完整实验
    experiment.run_full_experiment()


if __name__ == '__main__':
    main()
