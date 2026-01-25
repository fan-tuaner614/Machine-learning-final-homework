"""
高级实验主脚本
实验二的高级要求：提示学习与端到端评估

实验流程：
1. 在SC-2全量数据上进行自监督预训练（AutoEncoder）
2. 使用预训练编码器进行提示学习
3. 在4个标签比例上训练（10%, 30%, 50%, 100%）
4. 进行多维度评估（分类性能、聚类质量、特征稳定性、计算效率）
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

# 导入模块
from src.data_loader import SingleCellDataLoader
from src.preprocessing import SingleCellPreprocessor
from models.pretrain import AutoEncoder, SingleCellDataset, train_autoencoder
from models.prompt_learning import PromptLearningModel, train_prompt_model
from src.advanced_evaluator import AdvancedEvaluator
from utils.visualization import Visualizer
from config import Config


class AdvancedExperiment:
    """高级实验类"""
    
    def __init__(self, config=None):
        self.config = config if config else Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path('outputs_advanced')
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
        self.evaluator = AdvancedEvaluator(self.device)
        
        # 存储结果
        self.results = {}
        
    def step1_pretrain(self):
        """步骤1: 自监督预训练"""
        print("\n" + "="*60)
        print("步骤1: AutoEncoder预训练")
        print("="*60)
        
        # 1.1 加载数据
        print("\n步骤1.1: 数据加载")
        print("-"*60)
        datasets = self.data_loader.load_all_data()
        
        # 1.2 预处理SC-2数据（用于预训练）
        print("\n步骤1.2: 数据预处理 - SC-2_dense")
        print("-"*60)
        adata = datasets['SC-2_dense']
        adata_processed = self.preprocessor.preprocess(adata, dataset_name='SC-2_dense')
        
        # 编码标签
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        adata_processed.obs['subtype_encoded'] = le.fit_transform(adata_processed.obs['subtype'])
        
        # 1.3 准备预训练数据
        X_pretrain = torch.FloatTensor(adata_processed.X)
        input_dim = X_pretrain.shape[1]
        
        print(f"\n预训练数据维度: {X_pretrain.shape}")
        
        # 1.4 创建AutoEncoder
        print("\n步骤1.3: 创建AutoEncoder模型")
        print("-"*60)
        autoencoder = AutoEncoder(
            input_dim=input_dim,
            hidden_dims=[256, 128],
            latent_dim=64
        ).to(self.device)
        
        print(f"AutoEncoder架构:")
        print(f"  输入维度: {input_dim}")
        print(f"  隐藏层: [256, 128]")
        print(f"  潜在维度: 64")
        print(f"  总参数: {sum(p.numel() for p in autoencoder.parameters()):,}")
        
        # 1.5 训练AutoEncoder
        print("\n步骤1.4: 训练AutoEncoder")
        print("-"*60)
        
        # 创建数据加载器和优化器
        dataset = SingleCellDataset(X_pretrain, augment=False)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
        
        history = train_autoencoder(
            autoencoder,
            dataloader,
            optimizer,
            self.device,
            epochs=50
        )
        
        # 1.6 保存预训练模型
        model_path = self.output_dir / 'models' / 'pretrained_autoencoder.pth'
        torch.save(autoencoder.state_dict(), model_path)
        print(f"\n[OK] 预训练模型已保存: {model_path}")
        
        # 1.7 保存训练历史
        history_path = self.output_dir / 'pretrain_history_autoencoder.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        # 1.8 可视化训练历史
        self._plot_pretrain_history(history, 'autoencoder')
        
        # 存储结果
        self.pretrained_encoder = autoencoder.encoder
        self.input_dim = input_dim
        self.adata_processed = adata_processed
        
        return autoencoder, adata_processed
    
    def step2_prompt_learning(self, pretrained_encoder, adata_processed):
        """步骤2: 提示学习（在不同标签比例上）"""
        print("\n" + "="*60)
        print("步骤2: 提示学习")
        print("="*60)
        
        # 准备数据
        X_all = torch.FloatTensor(adata_processed.X)
        y_all = torch.LongTensor(adata_processed.obs['subtype_encoded'].values)
        n_classes = len(adata_processed.obs['subtype'].unique())
        
        # 加载标签索引
        label_ratios = ['10_percent', '30_percent', '50_percent', 'all']
        
        results = {}
        
        for ratio_name in label_ratios:
            print(f"\n{'='*60}")
            print(f"训练提示学习模型 - {ratio_name}")
            print(f"{'='*60}")
            
            # 加载索引
            if ratio_name == 'all':
                indices_path = Path('data') / 'all.npy'
            else:
                indices_path = Path('data') / f'{ratio_name}.npy'
            
            raw_indices = np.load(indices_path)
            
            # 过滤掉超出范围的索引（因为预处理可能删除了一些细胞）
            valid_indices = raw_indices[raw_indices < len(X_all)]
            print(f"使用 {len(valid_indices)} 个标注样本 ({len(valid_indices)/len(X_all)*100:.1f}%)")
            
            # 创建提示学习模型
            prompt_model = PromptLearningModel(
                pretrained_encoder=pretrained_encoder,
                latent_dim=64,  # 编码器输出维度
                num_classes=n_classes,
                prompt_length=10,
                prompt_dim=64
            ).to(self.device)
            
            print(f"\n提示学习模型:")
            total_params = sum(p.numel() for p in prompt_model.parameters())
            trainable_params = sum(p.numel() for p in prompt_model.parameters() if p.requires_grad)
            print(f"  总参数: {total_params:,}")
            print(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
            
            # 准备训练和验证数据
            from sklearn.model_selection import train_test_split
            
            # 获取标注数据
            X_labeled = X_all[valid_indices]
            y_labeled = y_all[valid_indices]
            
            # 划分训练和验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
            )
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # 创建优化器
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, prompt_model.parameters()),
                lr=0.001
            )
            
            # 训练提示学习模型
            history, best_model_state = train_prompt_model(
                prompt_model,
                train_loader,
                val_loader,
                optimizer,
                self.device,
                epochs=100,
                patience=10
            )
            
            # 保存模型
            model_path = self.output_dir / 'models' / f'prompt_model_{ratio_name}.pth'
            torch.save(prompt_model.state_dict(), model_path)
            print(f"\n[OK] 模型已保存: {model_path}")
            
            # 可视化训练历史
            self._plot_training_history(history, f'Prompt_Learning_{ratio_name}')
            
            # 存储结果
            results[ratio_name] = {
                'model': prompt_model,
                'history': history,
                'n_labeled': len(valid_indices)
            }
        
        self.prompt_results = results
        return results
    
    def step3_evaluation(self, adata_processed):
        """步骤3: 多维度评估"""
        print("\n" + "="*60)
        print("步骤3: 多维度评估")
        print("="*60)
        
        X_all = torch.FloatTensor(adata_processed.X)
        y_all = adata_processed.obs['subtype_encoded'].values
        
        all_evaluation_results = {}
        
        for ratio_name, result in self.prompt_results.items():
            print(f"\n{'='*60}")
            print(f"评估 {ratio_name} 模型")
            print(f"{'='*60}")
            
            model = result['model']
            model.eval()
            
            # 获取模型预测和特征
            with torch.no_grad():
                X_all_tensor = X_all.to(self.device)
                
                # 获取预测结果
                logits = model(X_all_tensor)
                y_pred = torch.argmax(logits, dim=1).cpu().numpy()
                
                # 获取特征表示（编码器输出）
                features = model.encoder(X_all_tensor).cpu().numpy()
            
            # 1. 分类性能评估
            print("\n1. 分类性能评估")
            print("-"*40)
            classification_results = self.evaluator.evaluate_classification(
                y_all, y_pred
            )
            
            print(f"准确率: {classification_results['accuracy']:.4f}")
            print(f"精确率: {classification_results['precision']:.4f}")
            print(f"召回率: {classification_results['recall']:.4f}")
            print(f"F1分数: {classification_results['f1_score']:.4f}")
            
            # 1.1 交叉验证评估（可选，更全面的评估）
            print("\n1.1 交叉验证评估")
            print("-"*40)
            cv_results = self.evaluator.evaluate_with_cross_validation(
                model, adata_processed.X, y_all, cv=5, device=self.device
            )
            
            print(f"5折交叉验证结果:")
            print(f"  准确率: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
            print(f"  精确率: {cv_results['precision_mean']:.4f} ± {cv_results['precision_std']:.4f}")
            print(f"  召回率: {cv_results['recall_mean']:.4f} ± {cv_results['recall_std']:.4f}")
            print(f"  F1分数: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
            
            # 将交叉验证结果添加到分类结果中
            classification_results['cv_results'] = cv_results
            
            # 2. 聚类质量评估
            print("\n2. 聚类质量评估")
            print("-"*40)
            clustering_results = self.evaluator.evaluate_clustering_quality(
                features, y_all
            )
            
            print(f"调整兰德指数 (ARI): {clustering_results['adjusted_rand_index']:.4f}")
            print(f"标准化互信息 (NMI): {clustering_results['normalized_mutual_info']:.4f}")
            print(f"轮廓系数: {clustering_results['silhouette_score']:.4f}")
            
            # 3. 特征稳定性评估
            print("\n3. 特征稳定性评估")
            print("-"*40)
            # 创建数据加载器（需要包含标签以匹配dataloader接口）
            from torch.utils.data import TensorDataset, DataLoader
            y_all_tensor = torch.LongTensor(y_all)
            dataset = TensorDataset(X_all, y_all_tensor)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
            
            stability_results = self.evaluator.evaluate_feature_stability(
                model, dataloader, self.device, n_runs=5
            )
            
            print(f"平均Jaccard相似度: {stability_results['jaccard_similarity_mean']:.4f}")
            print(f"标准差: {stability_results['jaccard_similarity_std']:.4f}")
            print(f"平均余弦相似度: {stability_results['cosine_similarity_mean']:.4f}")
            
            # 4. 计算效率评估
            print("\n4. 计算效率评估")
            print("-"*40)
            efficiency_results = self.evaluator.evaluate_computational_efficiency(
                model, dataloader, self.device
            )
            
            print(f"推理时间 (平均): {efficiency_results['inference_time_mean']:.4f} 秒")
            print(f"每样本推理时间: {efficiency_results['inference_time_per_sample']*1000:.2f} ms")
            print(f"总参数: {efficiency_results['total_parameters']:,}")
            print(f"可训练参数: {efficiency_results['trainable_parameters']:,} ({efficiency_results['trainable_ratio']:.1%})")
            
            # 汇总结果
            evaluation_results = {
                'classification': classification_results,
                'clustering': clustering_results,
                'stability': stability_results,
                'efficiency': efficiency_results,
                'n_labeled': result['n_labeled']
            }
            
            all_evaluation_results[ratio_name] = evaluation_results
            
            # 打印摘要
            print(f"\n{'='*60}")
            print(f"{ratio_name} 评估摘要")
            print(f"{'='*60}")
            print(f"分类准确率: {classification_results['accuracy']:.4f}")
            print(f"F1分数: {classification_results['f1_score']:.4f}")
            print(f"ARI: {clustering_results['adjusted_rand_index']:.4f}")
            print(f"NMI: {clustering_results['normalized_mutual_info']:.4f}")
            print(f"特征稳定性 (Jaccard): {stability_results['jaccard_similarity_mean']:.4f}")
            print(f"推理时间: {efficiency_results['inference_time_per_sample']*1000:.2f} ms")
        
        self.evaluation_results = all_evaluation_results
        return all_evaluation_results
    
    def step4_visualization(self):
        """步骤4: 结果可视化"""
        print("\n" + "="*60)
        print("步骤4: 结果可视化")
        print("="*60)
        
        # 准备数据
        ratios = list(self.evaluation_results.keys())
        metrics = {
            'accuracy': [],
            'f1_score': [],
            'ari': [],
            'nmi': [],
            'jaccard': [],
            'inference_time': []
        }
        
        for ratio in ratios:
            results = self.evaluation_results[ratio]
            metrics['accuracy'].append(results['classification']['accuracy'])
            metrics['f1_score'].append(results['classification']['f1_score'])
            metrics['ari'].append(results['clustering']['adjusted_rand_index'])
            metrics['nmi'].append(results['clustering']['normalized_mutual_info'])
            metrics['jaccard'].append(results['stability']['jaccard_similarity_mean'])
            metrics['inference_time'].append(results['efficiency']['inference_time_per_sample'] * 1000)  # 转换为ms
        
        # 创建性能对比图
        self._plot_performance_comparison(ratios, metrics)
        
        print("\n[OK] 所有可视化图表已生成")
    
    def step5_save_results(self):
        """步骤5: 保存完整结果"""
        print("\n" + "="*60)
        print("步骤5: 保存实验结果")
        print("="*60)
        
        # 辅助函数：递归转换numpy类型为Python原生类型
        def convert_to_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # 准备保存的结果（移除不能序列化的对象）
        save_results = {}
        
        for ratio_name, results in self.evaluation_results.items():
            save_results[ratio_name] = {
                'classification': convert_to_json_serializable(results['classification']),
                'clustering': convert_to_json_serializable(results['clustering']),
                'stability': convert_to_json_serializable(results['stability']),
                'efficiency': convert_to_json_serializable(results['efficiency']),
                'n_labeled': int(results['n_labeled'])
            }
        
        # 保存为JSON
        results_path = self.output_dir / 'all_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, indent=4, ensure_ascii=False)
        
        print(f"[OK] 完整结果已保存: {results_path}")
        
        # 打印最终摘要
        self._print_final_summary()
    
    def _plot_pretrain_history(self, history, model_name):
        """绘制预训练历史"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.plot(history['loss'], label='训练损失', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('重构损失', fontsize=12)
        plt.title(f'{model_name.upper()} 预训练曲线', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        save_path = self.output_dir / 'plots' / f'pretrain_{model_name}_loss.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] 预训练曲线已保存: {save_path}")
    
    def _plot_training_history(self, history, model_name):
        """绘制训练历史"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        axes[0].plot(history['train_loss'], label='训练损失', linewidth=2)
        axes[0].plot(history['val_loss'], label='验证损失', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('损失', fontsize=12)
        axes[0].set_title('训练和验证损失', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[1].plot(history['train_acc'], label='训练准确率', linewidth=2)
        axes[1].plot(history['val_acc'], label='验证准确率', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('准确率', fontsize=12)
        axes[1].set_title('训练和验证准确率', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'plots' / f'{model_name}_training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] 训练曲线已保存: {save_path}")
    
    def _plot_performance_comparison(self, ratios, metrics):
        """绘制性能对比图"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        ratio_labels = [r.replace('_', ' ').title() for r in ratios]
        
        # 1. 分类性能
        ax = axes[0, 0]
        x = np.arange(len(ratios))
        width = 0.35
        ax.bar(x - width/2, metrics['accuracy'], width, label='准确率', alpha=0.8)
        ax.bar(x + width/2, metrics['f1_score'], width, label='F1分数', alpha=0.8)
        ax.set_xlabel('标签比例', fontsize=12)
        ax.set_ylabel('得分', fontsize=12)
        ax.set_title('分类性能对比', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(ratio_labels, rotation=15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. 聚类质量
        ax = axes[0, 1]
        ax.bar(x - width/2, metrics['ari'], width, label='ARI', alpha=0.8)
        ax.bar(x + width/2, metrics['nmi'], width, label='NMI', alpha=0.8)
        ax.set_xlabel('标签比例', fontsize=12)
        ax.set_ylabel('得分', fontsize=12)
        ax.set_title('聚类质量对比', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(ratio_labels, rotation=15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. 特征稳定性
        ax = axes[1, 0]
        ax.plot(ratio_labels, metrics['jaccard'], marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('标签比例', fontsize=12)
        ax.set_ylabel('Jaccard相似度', fontsize=12)
        ax.set_title('特征稳定性', fontsize=13, fontweight='bold')
        ax.set_xticklabels(ratio_labels, rotation=15)
        ax.grid(True, alpha=0.3)
        
        # 4. 计算效率
        ax = axes[1, 1]
        ax.bar(ratio_labels, metrics['inference_time'], alpha=0.8, color='coral')
        ax.set_xlabel('标签比例', fontsize=12)
        ax.set_ylabel('推理时间 (ms)', fontsize=12)
        ax.set_title('计算效率', fontsize=13, fontweight='bold')
        ax.set_xticklabels(ratio_labels, rotation=15)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / 'plots' / 'performance_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] 性能对比图已保存: {save_path}")
    
    def _print_final_summary(self):
        """打印最终摘要"""
        print("\n" + "="*70)
        print(" "*20 + "实验完成摘要")
        print("="*70)
        
        print("\n模型文件:")
        models_dir = self.output_dir / 'models'
        for model_file in sorted(models_dir.glob('*.pth')):
            print(f"  - {model_file.name}")
        
        print("\n可视化图表:")
        plots_dir = self.output_dir / 'plots'
        for plot_file in sorted(plots_dir.glob('*.png')):
            print(f"  - {plot_file.name}")
        
        print("\n性能摘要:")
        print(f"{'标签比例':<15} {'准确率':<10} {'F1分数':<10} {'ARI':<10} {'NMI':<10}")
        print("-"*70)
        
        for ratio_name, results in self.evaluation_results.items():
            ratio_display = ratio_name.replace('_', ' ').title()
            acc = results['classification']['accuracy']
            f1 = results['classification']['f1_score']
            ari = results['clustering']['adjusted_rand_index']
            nmi = results['clustering']['normalized_mutual_info']
            print(f"{ratio_display:<15} {acc:<10.4f} {f1:<10.4f} {ari:<10.4f} {nmi:<10.4f}")
        
        print("\n" + "="*70)
        print("所有实验已完成！")
        print("="*70)
    
    def run_full_experiment(self):
        """运行完整实验流程"""
        print("\n" + "="*70)
        print(" "*15 + "高级实验：提示学习与端到端评估")
        print("="*70)
        
        try:
            # 步骤1: 预训练
            autoencoder, adata_processed = self.step1_pretrain()
            
            # 步骤2: 提示学习
            prompt_results = self.step2_prompt_learning(
                autoencoder.encoder, 
                adata_processed
            )
            
            # 步骤3: 多维度评估
            evaluation_results = self.step3_evaluation(adata_processed)
            
            # 步骤4: 可视化
            self.step4_visualization()
            
            # 步骤5: 保存结果
            self.step5_save_results()
            
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
    experiment = AdvancedExperiment(config)
    
    # 运行完整实验
    experiment.run_full_experiment()


if __name__ == '__main__':
    main()
