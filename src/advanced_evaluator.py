"""
高级评估模块
实现多维度评估：分类性能、聚类质量、特征稳定性、计算效率
"""

import numpy as np
import time
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.cluster import KMeans
import torch


class AdvancedEvaluator:
    """
    高级评估器
    提供分类性能、聚类质量、特征稳定性、计算效率的全面评估
    """
    
    def __init__(self, device):
        self.device = device
        self.results = {}
    
    def evaluate_classification(self, y_true, y_pred, y_prob=None, average='macro'):
        """
        评估分类性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率 (n_samples, n_classes)，用于计算 AUC
            average: 平均方式 ('macro', 'micro', 'weighted')
        
        Returns:
            dict: 分类性能指标（包含 ARI/NMI/F1/AUC）
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        
        # 每个类别的指标
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 计算 AUC（如果提供了概率）
        auc_score = None
        auc_per_class = None
        if y_prob is not None:
            try:
                n_classes = y_prob.shape[1] if len(y_prob.shape) > 1 else 2
                
                if n_classes == 2:
                    # 二分类
                    auc_score = roc_auc_score(y_true, y_prob[:, 1] if len(y_prob.shape) > 1 else y_prob)
                else:
                    # 多分类：使用 macro-average 和 ovr (one-vs-rest) 策略
                    auc_score = roc_auc_score(
                        y_true, y_prob, 
                        multi_class='ovr',
                        average='macro'
                    )
                    # 每个类别的 AUC
                    y_true_bin = label_binarize(y_true, classes=range(n_classes))
                    auc_per_class = []
                    for i in range(n_classes):
                        try:
                            auc_i = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                            auc_per_class.append(auc_i)
                        except:
                            auc_per_class.append(0.0)
            except Exception as e:
                print(f"  [Warning] AUC 计算失败: {e}")
                auc_score = None
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc_score,  # 添加 AUC 指标
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'auc_per_class': auc_per_class if auc_per_class else None,
            'support_per_class': support_per_class.tolist(),
            'confusion_matrix': cm.tolist()
        }
        
        return results
    
    def evaluate_with_cross_validation(self, model, X, y, cv=5, device='cpu'):
        """
        使用交叉验证评估模型
        
        Args:
            model: 模型实例
            X: 特征数据
            y: 标签
            cv: 交叉验证折数
            device: 设备
        
        Returns:
            dict: 交叉验证结果
        """
        from torch.utils.data import DataLoader, TensorDataset
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        fold_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 创建数据加载器
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # 训练模型（这里简化，实际应该调用训练函数）
            # 假设model已经训练好，这里只做评估
            model.eval()
            y_pred = []
            y_prob_list = []
            
            with torch.no_grad():
                for batch_X, _ in val_loader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    y_pred.extend(predicted.cpu().numpy())
                    y_prob_list.append(probs.cpu().numpy())
            
            y_pred = np.array(y_pred)
            y_prob = np.vstack(y_prob_list)
            
            # 计算指标
            accuracy = accuracy_score(y_val, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average='macro', zero_division=0
            )
            
            # 计算 AUC
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_val, y_prob, multi_class='ovr', average='macro')
                fold_scores['auc'].append(auc)
            except:
                fold_scores['auc'].append(0.0)
            
            fold_scores['accuracy'].append(accuracy)
            fold_scores['precision'].append(precision)
            fold_scores['recall'].append(recall)
            fold_scores['f1'].append(f1)
        
        # 计算平均和标准差
        results = {
            'accuracy_mean': np.mean(fold_scores['accuracy']),
            'accuracy_std': np.std(fold_scores['accuracy']),
            'precision_mean': np.mean(fold_scores['precision']),
            'precision_std': np.std(fold_scores['precision']),
            'recall_mean': np.mean(fold_scores['recall']),
            'recall_std': np.std(fold_scores['recall']),
            'f1_mean': np.mean(fold_scores['f1']),
            'f1_std': np.std(fold_scores['f1']),
            'auc_mean': np.mean(fold_scores['auc']),
            'auc_std': np.std(fold_scores['auc']),
            'fold_scores': fold_scores
        }
        
        return results
    
    def evaluate_clustering_quality(self, features, true_labels, n_clusters=None):
        """
        评估聚类质量
        
        Args:
            features: 特征数据
            true_labels: 真实标签
            n_clusters: 聚类数量（如果为None，使用真实标签的类别数）
        
        Returns:
            dict: 聚类质量指标
        """
        if n_clusters is None:
            n_clusters = len(np.unique(true_labels))
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # 计算调整兰德指数 (ARI)
        ari = adjusted_rand_score(true_labels, cluster_labels)
        
        # 计算标准化互信息 (NMI)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        
        # 计算轮廓系数 (Silhouette Score)
        if len(features) < 10000:  # 对于大数据集，轮廓系数计算很慢
            silhouette = silhouette_score(features, cluster_labels)
        else:
            # 随机采样计算
            indices = np.random.choice(len(features), 10000, replace=False)
            silhouette = silhouette_score(features[indices], cluster_labels[indices])
        
        results = {
            'adjusted_rand_index': ari,
            'normalized_mutual_info': nmi,
            'silhouette_score': silhouette,
            'n_clusters': n_clusters
        }
        
        return results
    
    def evaluate_feature_stability(self, model, dataloader, device, n_runs=5):
        """
        评估特征稳定性
        通过不同随机种子下提取的特征计算Jaccard相似度
        
        Args:
            model: 模型实例
            dataloader: 数据加载器
            device: 设备
            n_runs: 运行次数
        
        Returns:
            dict: 特征稳定性指标
        """
        from models.prompt_learning import extract_prompt_features
        
        # 提取多次特征
        all_features = []
        
        for run in range(n_runs):
            # 设置不同的随机种子
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)
            
            # 提取特征
            features = extract_prompt_features(model, dataloader, device)
            all_features.append(features)
        
        # 计算Jaccard相似度
        # 对于连续特征，我们使用top-k特征的交集
        jaccard_scores = []
        k = min(100, features.shape[1])  # 选择top-k个特征
        
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                # 对每个样本，找到top-k特征的索引
                features_i = all_features[i]
                features_j = all_features[j]
                
                sample_jaccards = []
                for sample_idx in range(len(features_i)):
                    top_k_i = set(np.argsort(features_i[sample_idx])[-k:])
                    top_k_j = set(np.argsort(features_j[sample_idx])[-k:])
                    
                    intersection = len(top_k_i & top_k_j)
                    union = len(top_k_i | top_k_j)
                    
                    jaccard = intersection / union if union > 0 else 0
                    sample_jaccards.append(jaccard)
                
                jaccard_scores.append(np.mean(sample_jaccards))
        
        # 特征相似度（使用余弦相似度）
        cosine_similarities = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                # 计算每个样本的余弦相似度
                features_i = all_features[i]
                features_j = all_features[j]
                
                # 归一化
                features_i_norm = features_i / (np.linalg.norm(features_i, axis=1, keepdims=True) + 1e-8)
                features_j_norm = features_j / (np.linalg.norm(features_j, axis=1, keepdims=True) + 1e-8)
                
                # 计算余弦相似度
                cosine_sim = np.sum(features_i_norm * features_j_norm, axis=1).mean()
                cosine_similarities.append(cosine_sim)
        
        results = {
            'jaccard_similarity_mean': np.mean(jaccard_scores),
            'jaccard_similarity_std': np.std(jaccard_scores),
            'cosine_similarity_mean': np.mean(cosine_similarities),
            'cosine_similarity_std': np.std(cosine_similarities),
            'n_runs': n_runs
        }
        
        return results
    
    def evaluate_computational_efficiency(self, model, dataloader, device, n_iterations=10):
        """
        评估计算效率
        
        Args:
            model: 模型实例
            dataloader: 数据加载器
            device: 设备
            n_iterations: 迭代次数
        
        Returns:
            dict: 计算效率指标
        """
        model.eval()
        
        # 预热
        with torch.no_grad():
            for batch_data, _ in dataloader:
                batch_data = batch_data.to(device)
                _ = model(batch_data)
                break
        
        # 测量推理时间
        inference_times = []
        
        with torch.no_grad():
            for iteration in range(n_iterations):
                start_time = time.time()
                
                for batch_data, _ in dataloader:
                    batch_data = batch_data.to(device)
                    _ = model(batch_data)
                
                end_time = time.time()
                inference_times.append(end_time - start_time)
        
        # 计算参数量
        param_stats = model.count_parameters()
        
        # 计算内存占用
        if device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'):
            torch.cuda.synchronize()
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        results = {
            'inference_time_mean': np.mean(inference_times),
            'inference_time_std': np.std(inference_times),
            'inference_time_per_sample': np.mean(inference_times) / len(dataloader.dataset),
            'total_parameters': param_stats['total'],
            'trainable_parameters': param_stats['trainable'],
            'frozen_parameters': param_stats['frozen'],
            'trainable_ratio': param_stats['trainable_ratio'],
            'memory_allocated_mb': memory_allocated,
            'memory_reserved_mb': memory_reserved
        }
        
        return results
    
    def comprehensive_evaluation(self, model, dataloader, y_true, y_pred, 
                                 features, device, n_runs=5):
        """
        综合评估
        
        Args:
            model: 模型实例
            dataloader: 数据加载器
            y_true: 真实标签
            y_pred: 预测标签
            features: 特征数据
            device: 设备
            n_runs: 特征稳定性评估的运行次数
        
        Returns:
            dict: 综合评估结果
        """
        print("\n" + "="*60)
        print("开始综合评估...")
        print("="*60)
        
        # 1. 分类性能评估
        print("\n1. 评估分类性能...")
        classification_results = self.evaluate_classification(y_true, y_pred)
        
        # 2. 聚类质量评估
        print("2. 评估聚类质量...")
        clustering_results = self.evaluate_clustering_quality(features, y_true)
        
        # 3. 特征稳定性评估
        print("3. 评估特征稳定性...")
        stability_results = self.evaluate_feature_stability(model, dataloader, device, n_runs)
        
        # 4. 计算效率评估
        print("4. 评估计算效率...")
        efficiency_results = self.evaluate_computational_efficiency(model, dataloader, device)
        
        # 整合结果
        comprehensive_results = {
            'classification': classification_results,
            'clustering': clustering_results,
            'stability': stability_results,
            'efficiency': efficiency_results
        }
        
        self.results = comprehensive_results
        
        # 打印摘要
        self.print_summary()
        
        return comprehensive_results
    
    def print_summary(self):
        """打印评估摘要"""
        if not self.results:
            print("还没有评估结果")
            return
        
        print("\n" + "="*60)
        print("评估结果摘要")
        print("="*60)
        
        # 分类性能
        if 'classification' in self.results:
            cls_results = self.results['classification']
            print("\n[分类性能]")
            print(f"  准确率 (Accuracy):  {cls_results['accuracy']:.4f}")
            print(f"  精确率 (Precision): {cls_results['precision']:.4f}")
            print(f"  召回率 (Recall):    {cls_results['recall']:.4f}")
            print(f"  F1分数 (F1-Score):  {cls_results['f1_score']:.4f}")
            if cls_results.get('auc') is not None:
                print(f"  AUC (Area Under Curve): {cls_results['auc']:.4f}")
        
        # 聚类质量
        if 'clustering' in self.results:
            cluster_results = self.results['clustering']
            print("\n[聚类质量]")
            print(f"  调整兰德指数 (ARI):         {cluster_results['adjusted_rand_index']:.4f}")
            print(f"  标准化互信息 (NMI):         {cluster_results['normalized_mutual_info']:.4f}")
            print(f"  轮廓系数 (Silhouette):      {cluster_results['silhouette_score']:.4f}")
        
        # 特征稳定性
        if 'stability' in self.results:
            stab_results = self.results['stability']
            print("\n[特征稳定性]")
            print(f"  Jaccard相似度: {stab_results['jaccard_similarity_mean']:.4f} ± {stab_results['jaccard_similarity_std']:.4f}")
            print(f"  余弦相似度:    {stab_results['cosine_similarity_mean']:.4f} ± {stab_results['cosine_similarity_std']:.4f}")
        
        # 计算效率
        if 'efficiency' in self.results:
            eff_results = self.results['efficiency']
            print("\n[计算效率]")
            print(f"  推理时间: {eff_results['inference_time_mean']:.4f} ± {eff_results['inference_time_std']:.4f} 秒")
            print(f"  每样本推理时间: {eff_results['inference_time_per_sample']*1000:.2f} 毫秒")
            print(f"  总参数量: {eff_results['total_parameters']:,}")
            print(f"  可训练参数: {eff_results['trainable_parameters']:,} ({eff_results['trainable_ratio']:.2%})")
            print(f"  冻结参数: {eff_results['frozen_parameters']:,}")
            if eff_results['memory_allocated_mb'] > 0:
                print(f"  GPU内存占用: {eff_results['memory_allocated_mb']:.2f} MB")
        
        print("\n" + "="*60)
