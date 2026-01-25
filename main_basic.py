"""
初级实验主脚本
实验二的初级要求：自监督预训练模型构建（30分）

实验流程：
任务1：自编码器设计与实现
1. 构建标准的自编码器模型
2. 设计合适的编码器和解码器结构
3. 实现重构损失函数并训练
4. 保存训练好的编码器模型用于特征提取
5. 实现基因表达数据的增强方法（高斯噪声添加和随机掩码）
6. 设计增强参数的可调节性
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 导入模块
from src.data_loader import SingleCellDataLoader
from src.preprocessing import SingleCellPreprocessor
from models.pretrain import AutoEncoder, SingleCellDataset, train_autoencoder
from utils.visualization import Visualizer
from config import Config


class BasicExperiment:
    """初级实验类：自监督预训练模型构建"""
    
    def __init__(self, config=None):
        self.config = config if config else Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path('outputs_basic')
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        
        # 初始化组件
        self.data_loader = SingleCellDataLoader(self.config.data_dir if hasattr(self.config, 'data_dir') else 'data')
        self.preprocessor = SingleCellPreprocessor(
            min_genes=self.config.min_genes if hasattr(self.config, 'min_genes') else 200,
            min_cells=self.config.min_cells if hasattr(self.config, 'min_cells') else 3,
            n_top_genes=self.config.n_top_genes if hasattr(self.config, 'n_top_genes') else 2000,
            n_pca_components=self.config.n_pca_components if hasattr(self.config, 'n_pca_components') else 50
        )
        self.visualizer = Visualizer(self.output_dir / 'plots')
        
        # 存储结果
        self.results = {}
        
    def step1_load_and_preprocess(self):
        """步骤1: 数据加载与预处理"""
        print("\n" + "="*60)
        print("步骤1: 数据加载与预处理")
        print("="*60)
        
        # 1.1 加载数据
        print("\n[1.1] 数据加载")
        print("-"*60)
        datasets = self.data_loader.load_all_data()
        
        # 1.2 预处理SC-2数据（用于预训练）
        print("\n[1.2] 数据预处理 - SC-2_dense")
        print("-"*60)
        adata = datasets['SC-2_dense']
        adata_processed = self.preprocessor.preprocess(adata, dataset_name='SC-2_dense')
        
        # 1.3 保存预处理后的数据信息
        print("\n[1.3] 预处理结果")
        print("-"*60)
        print(f"细胞数量: {adata_processed.n_obs}")
        print(f"特征维度: {adata_processed.n_vars}")
        print(f"数据形状: {adata_processed.X.shape}")
        
        # 存储
        self.adata_processed = adata_processed
        self.X_data = torch.FloatTensor(adata_processed.X)
        self.input_dim = adata_processed.X.shape[1]
        
        print(f"\n[OK] 数据准备完成，输入维度: {self.input_dim}")
        
        return adata_processed
    
    def step2_build_autoencoder(self):
        """步骤2: 构建自编码器模型"""
        print("\n" + "="*60)
        print("步骤2: 构建自编码器模型")
        print("="*60)
        
        # 2.1 模型架构设计
        print("\n[2.1] 模型架构设计")
        print("-"*60)
        
        hidden_dims = [256, 128]
        latent_dim = 64
        
        self.autoencoder = AutoEncoder(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim
        ).to(self.device)
        
        # 2.2 打印模型信息
        print(f"\n自编码器架构:")
        print(f"  输入维度: {self.input_dim}")
        print(f"  编码器隐藏层: {hidden_dims}")
        print(f"  潜在空间维度: {latent_dim}")
        print(f"  解码器隐藏层: {hidden_dims[::-1]}")
        print(f"  输出维度: {self.input_dim}")
        
        total_params = sum(p.numel() for p in self.autoencoder.parameters())
        encoder_params = sum(p.numel() for p in self.autoencoder.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.autoencoder.decoder.parameters())
        
        print(f"\n参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  编码器参数: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
        print(f"  解码器参数: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
        
        # 2.3 打印详细结构
        print(f"\n编码器结构:")
        print(self.autoencoder.encoder)
        print(f"\n解码器结构:")
        print(self.autoencoder.decoder)
        
        print("\n[OK] 自编码器模型构建完成")
        
        return self.autoencoder
    
    def step3_data_augmentation(self):
        """步骤3: 实现数据增强方法"""
        print("\n" + "="*60)
        print("步骤3: 数据增强方法实现")
        print("="*60)
        
        print("\n[3.1] 数据增强策略")
        print("-"*60)
        print("实现的增强方法:")
        print("  1. 高斯噪声添加 (Gaussian Noise)")
        print("     - 参数: noise_std (默认0.1)")
        print("     - 作用: 增加表达值的随机扰动，提高模型鲁棒性")
        print("  2. 随机掩码 (Random Masking)")
        print("     - 参数: mask_prob (默认0.15)")
        print("     - 作用: 随机遮盖部分基因表达，学习更好的表征")
        
        # 3.2 展示数据增强效果
        print("\n[3.2] 数据增强效果展示")
        print("-"*60)
        
        # 获取一个样本
        sample = self.X_data[0].to(self.device)
        
        # 手动应用不同的增强
        augmentations = []
        
        # 原始数据
        augmentations.append(('原始数据', sample))
        
        # 高斯噪声(0.1)
        aug1 = sample + torch.randn_like(sample) * 0.1
        augmentations.append(('高斯噪声(0.1)', aug1))
        
        # 高斯噪声(0.2)
        aug2 = sample + torch.randn_like(sample) * 0.2
        augmentations.append(('高斯噪声(0.2)', aug2))
        
        # 随机掩码(15%)
        mask1 = (torch.rand_like(sample) > 0.15).float()
        aug3 = sample * mask1
        augmentations.append(('随机掩码(15%)', aug3))
        
        # 随机掩码(30%)
        mask2 = (torch.rand_like(sample) > 0.30).float()
        aug4 = sample * mask2
        augmentations.append(('随机掩码(30%)', aug4))
        
        # 混合增强
        aug5 = (sample + torch.randn_like(sample) * 0.1) * mask1
        augmentations.append(('混合增强', aug5))
        
        print("\n样本统计信息对比:")
        print(f"{'增强方法':<20} {'均值':<12} {'标准差':<12} {'非零比例':<12}")
        print("-"*60)
        
        for name, aug_data in augmentations:
            mean_val = aug_data.mean().item()
            std_val = aug_data.std().item()
            nonzero_ratio = (aug_data != 0).float().mean().item()
            
            print(f"{name:<20} {mean_val:<12.4f} {std_val:<12.4f} {nonzero_ratio:<12.4f}")
        
        print("\n[OK] 数据增强方法已实现并验证")
        
        return augmentations
    
    def step4_train_autoencoder(self, n_epochs=50, batch_size=32, learning_rate=0.001):
        """步骤4: 训练自编码器"""
        print("\n" + "="*60)
        print("步骤4: 训练自编码器")
        print("="*60)
        
        # 4.1 训练配置
        print("\n[4.1] 训练配置")
        print("-"*60)
        print(f"训练轮数: {n_epochs}")
        print(f"批次大小: {batch_size}")
        print(f"学习率: {learning_rate}")
        print(f"优化器: Adam")
        print(f"损失函数: MSE (均方误差)")
        print(f"数据增强: 在训练循环中动态应用")
        
        # 4.2 准备数据
        dataset = SingleCellDataset(self.X_data.numpy(), augment=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 4.3 准备优化器
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        
        # 4.4 开始训练
        print("\n[4.2] 开始训练...")
        print("-"*60)
        
        from models.pretrain import train_autoencoder
        history = train_autoencoder(
            self.autoencoder,
            dataloader,
            optimizer,
            self.device,
            epochs=n_epochs
        )
        
        # 4.5 训练结果
        print("\n[4.3] 训练结果")
        print("-"*60)
        print(f"最终训练损失: {history['loss'][-1]:.6f}")
        print(f"最低训练损失: {min(history['loss']):.6f} (Epoch {np.argmin(history['loss'])+1})")
        
        # 4.6 保存训练历史
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"\n[OK] 训练历史已保存: {history_path}")
        
        self.history = history
        
        return history
    
    def step5_save_model(self):
        """步骤5: 保存模型"""
        print("\n" + "="*60)
        print("步骤5: 保存模型")
        print("="*60)
        
        # 5.1 保存完整自编码器
        autoencoder_path = self.output_dir / 'models' / 'autoencoder_full.pth'
        torch.save({
            'model_state_dict': self.autoencoder.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': [256, 128],
            'latent_dim': 64
        }, autoencoder_path)
        print(f"\n[5.1] 完整自编码器已保存: {autoencoder_path}")
        
        # 5.2 保存编码器（用于特征提取）
        encoder_path = self.output_dir / 'models' / 'encoder.pth'
        torch.save({
            'encoder_state_dict': self.autoencoder.encoder.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': 64
        }, encoder_path)
        print(f"[5.2] 编码器已保存: {encoder_path}")
        
        # 5.3 保存解码器
        decoder_path = self.output_dir / 'models' / 'decoder.pth'
        torch.save({
            'decoder_state_dict': self.autoencoder.decoder.state_dict(),
            'latent_dim': 64,
            'output_dim': self.input_dim
        }, decoder_path)
        print(f"[5.3] 解码器已保存: {decoder_path}")
        
        print("\n[OK] 所有模型已保存")
    
    def step6_evaluation(self):
        """步骤6: 模型评估"""
        print("\n" + "="*60)
        print("步骤6: 模型评估")
        print("="*60)
        
        self.autoencoder.eval()
        
        # 6.1 重构误差评估
        print("\n[6.1] 重构误差评估")
        print("-"*60)
        
        with torch.no_grad():
            X_input = self.X_data.to(self.device)
            X_reconstructed, _ = self.autoencoder(X_input)  # 解包元组，取重构结果
            
            # 计算各种误差指标
            mse = nn.MSELoss()(X_reconstructed, X_input).item()
            mae = torch.mean(torch.abs(X_reconstructed - X_input)).item()
            
            # 相对误差
            relative_error = torch.mean(
                torch.abs(X_reconstructed - X_input) / (torch.abs(X_input) + 1e-8)
            ).item()
            
            # 相关系数
            correlation = np.corrcoef(
                X_input.cpu().numpy().flatten(),
                X_reconstructed.cpu().numpy().flatten()
            )[0, 1]
        
        print(f"均方误差 (MSE): {mse:.6f}")
        print(f"平均绝对误差 (MAE): {mae:.6f}")
        print(f"相对误差: {relative_error:.6f}")
        print(f"相关系数: {correlation:.6f}")
        
        # 6.2 潜在空间分析
        print("\n[6.2] 潜在空间分析")
        print("-"*60)
        
        with torch.no_grad():
            latent = self.autoencoder.encoder(X_input)
            
            latent_mean = latent.mean(dim=0).cpu().numpy()
            latent_std = latent.std(dim=0).cpu().numpy()
        
        print(f"潜在空间维度: {latent.shape[1]}")
        print(f"潜在向量均值范围: [{latent_mean.min():.4f}, {latent_mean.max():.4f}]")
        print(f"潜在向量标准差范围: [{latent_std.min():.4f}, {latent_std.max():.4f}]")
        print(f"潜在空间平均激活度: {(latent != 0).float().mean().item():.4f}")
        
        # 保存评估结果
        evaluation_results = {
            'reconstruction': {
                'mse': float(mse),
                'mae': float(mae),
                'relative_error': float(relative_error),
                'correlation': float(correlation)
            },
            'latent_space': {
                'dimension': int(latent.shape[1]),
                'mean_range': [float(latent_mean.min()), float(latent_mean.max())],
                'std_range': [float(latent_std.min()), float(latent_std.max())],
                'activation_rate': float((latent != 0).float().mean().item())
            }
        }
        
        results_path = self.output_dir / 'evaluation_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=4, ensure_ascii=False)
        print(f"\n[OK] 评估结果已保存: {results_path}")
        
        return evaluation_results
    
    def step7_visualization(self):
        """步骤7: 结果可视化"""
        print("\n" + "="*60)
        print("步骤7: 结果可视化")
        print("="*60)
        
        # 7.1 训练损失曲线
        print("\n[7.1] 绘制训练损失曲线")
        print("-"*60)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['loss'], linewidth=2, color='#2E86AB')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('重构损失 (MSE)', fontsize=12)
        plt.title('自编码器训练曲线', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        loss_curve_path = self.output_dir / 'plots' / 'training_loss_curve.png'
        plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] 训练损失曲线已保存: {loss_curve_path}")
        
        # 7.2 重构效果对比
        print("\n[7.2] 绘制重构效果对比")
        print("-"*60)
        
        self.autoencoder.eval()
        with torch.no_grad():
            # 选择几个样本
            n_samples = 5
            X_samples = self.X_data[:n_samples].to(self.device)
            X_recon, _ = self.autoencoder(X_samples)  # 解包元组，取重构结果
            X_recon = X_recon.cpu().numpy()
            X_original = X_samples.cpu().numpy()
            
            fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3*n_samples))
            
            for i in range(n_samples):
                # 原始数据
                axes[i, 0].plot(X_original[i], linewidth=1, alpha=0.7)
                axes[i, 0].set_title(f'样本 {i+1} - 原始数据', fontsize=11)
                axes[i, 0].set_ylabel('表达值', fontsize=10)
                axes[i, 0].grid(True, alpha=0.3)
                
                # 重构数据
                axes[i, 1].plot(X_recon[i], linewidth=1, alpha=0.7, color='orange')
                axes[i, 1].set_title(f'样本 {i+1} - 重构数据', fontsize=11)
                axes[i, 1].set_ylabel('表达值', fontsize=10)
                axes[i, 1].grid(True, alpha=0.3)
                
                if i == n_samples - 1:
                    axes[i, 0].set_xlabel('特征索引', fontsize=10)
                    axes[i, 1].set_xlabel('特征索引', fontsize=10)
            
            plt.tight_layout()
            recon_path = self.output_dir / 'plots' / 'reconstruction_comparison.png'
            plt.savefig(recon_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OK] 重构效果对比已保存: {recon_path}")
        
        # 7.3 潜在空间分布
        print("\n[7.3] 绘制潜在空间分布")
        print("-"*60)
        
        with torch.no_grad():
            latent = self.autoencoder.encoder(self.X_data.to(self.device)).cpu().numpy()
            
            # 绘制前两个维度
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # 散点图
            axes[0].scatter(latent[:, 0], latent[:, 1], alpha=0.5, s=10)
            axes[0].set_xlabel('潜在维度 1', fontsize=12)
            axes[0].set_ylabel('潜在维度 2', fontsize=12)
            axes[0].set_title('潜在空间分布 (维度1 vs 维度2)', fontsize=13, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # 维度激活热图
            latent_mean = np.abs(latent).mean(axis=0)
            axes[1].bar(range(len(latent_mean)), latent_mean, color='steelblue', alpha=0.7)
            axes[1].set_xlabel('潜在维度', fontsize=12)
            axes[1].set_ylabel('平均激活值', fontsize=12)
            axes[1].set_title('潜在空间各维度激活强度', fontsize=13, fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            latent_path = self.output_dir / 'plots' / 'latent_space_analysis.png'
            plt.savefig(latent_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OK] 潜在空间分布已保存: {latent_path}")
        
        print("\n[OK] 所有可视化已完成")
    
    def step8_summary(self):
        """步骤8: 实验总结"""
        print("\n" + "="*70)
        print(" "*20 + "实验完成摘要")
        print("="*70)
        
        print("\n【初级要求完成情况】")
        print("-"*70)
        print("[OK] 任务1.1: 自编码器模型构建")
        print("  - 编码器: 输入 -> [256, 128] -> 64维潜在空间")
        print("  - 解码器: 64维潜在空间 -> [128, 256] -> 输出")
        print("  - 重构损失函数: MSE (均方误差)")
        
        print("\n[OK] 任务1.2: 数据增强方法实现")
        print("  - 高斯噪声添加: 可调节标准差 (默认0.1)")
        print("  - 随机掩码: 可调节掩码比例 (默认15%)")
        print("  - 参数可调节性: 通过函数参数灵活配置")
        
        print("\n【生成的文件】")
        print("-"*70)
        print("模型文件:")
        models_dir = self.output_dir / 'models'
        for model_file in sorted(models_dir.glob('*.pth')):
            print(f"  - {model_file.name}")
        
        print("\n可视化图表:")
        plots_dir = self.output_dir / 'plots'
        for plot_file in sorted(plots_dir.glob('*.png')):
            print(f"  - {plot_file.name}")
        
        print("\n结果文件:")
        print(f"  - training_history.json")
        print(f"  - evaluation_results.json")
        
        print("\n【性能指标】")
        print("-"*70)
        if hasattr(self, 'history'):
            print(f"最终训练损失: {self.history['loss'][-1]:.6f}")
            print(f"最低训练损失: {min(self.history['loss']):.6f}")
        
        print("\n" + "="*70)
        print("初级实验已完成！编码器模型已保存，可用于后续的中级和高级实验。")
        print("="*70)
    
    def run_full_experiment(self):
        """运行完整的初级实验"""
        print("\n" + "="*70)
        print(" "*10 + "初级实验：自监督预训练模型构建（30分）")
        print("="*70)
        
        try:
            # 步骤1: 数据加载与预处理
            self.step1_load_and_preprocess()
            
            # 步骤2: 构建自编码器模型
            self.step2_build_autoencoder()
            
            # 步骤3: 实现数据增强方法
            self.step3_data_augmentation()
            
            # 步骤4: 训练自编码器
            self.step4_train_autoencoder(n_epochs=50, batch_size=32, learning_rate=0.001)
            
            # 步骤5: 保存模型
            self.step5_save_model()
            
            # 步骤6: 模型评估
            self.step6_evaluation()
            
            # 步骤7: 结果可视化
            self.step7_visualization()
            
            # 步骤8: 实验总结
            self.step8_summary()
            
            print("\n" + "="*70)
            print("实验成功完成！")
            print("="*70)
            
        except Exception as e:
            print(f"\n[ERROR] 实验出错: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    # 创建配置
    config = Config()
    
    # 创建实验实例
    experiment = BasicExperiment(config)
    
    # 运行完整实验
    experiment.run_full_experiment()


if __name__ == '__main__':
    main()
