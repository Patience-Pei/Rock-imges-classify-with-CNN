# -*- coding: utf-8 -*-
"""
集成学习模块
"""

import torch
import torch.nn as nn
import numpy as np
from models import get_model
from trainer import Trainer
from config import Config
import os

class EnsembleTrainer:
    """集成学习训练器"""
    
    def __init__(self, model_types, train_loader, valid_loader, test_loader, config=Config):
        self.model_types = model_types
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config.DEVICE
        
        self.models = {}
        self.trainers = {}
        self.histories = {}
        
        # 创建保存目录
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    def train_individual_models(self):
        """训练各个单独的模型"""
        print("开始训练集成模型的各个组件...")
        print("=" * 60)
        
        for i, model_type in enumerate(self.model_types):
            print(f"\n训练模型 {i+1}/{len(self.model_types)}: {model_type}")
            print("-" * 40)
            
            # 创建模型
            model = get_model(
                model_type=model_type,
                num_classes=self.config.NUM_CLASSES,
                pretrained=self.config.PRETRAINED,
                dropout_rate=self.config.DROPOUT_RATE
            )
            
            # 创建训练器
            trainer = Trainer(
                model=model,
                train_loader=self.train_loader,
                valid_loader=self.valid_loader,
                test_loader=self.test_loader,
                config=self.config
            )
            
            # 训练模型
            history, test_acc, predictions, targets = trainer.train()
            
            # 保存结果
            self.models[model_type] = model
            self.trainers[model_type] = trainer
            self.histories[model_type] = history
            
            # 保存单个模型
            model_save_path = os.path.join(
                self.config.MODEL_SAVE_DIR, 
                f'best_model_{model_type}.pth'
            )
            trainer.save_model(f'best_model_{model_type}.pth')
            
            print(f"{model_type} 训练完成，测试准确率: {test_acc:.4f}")
        
        print("\n所有单独模型训练完成!")
        return self.models, self.histories
    
    def create_ensemble(self, weights=None):
        """创建集成模型"""
        if not self.models:
            raise ValueError("请先训练单独的模型")
        
        # 加载最佳模型权重
        for model_type in self.model_types:
            model_path = os.path.join(
                self.config.MODEL_SAVE_DIR, 
                f'best_model_{model_type}.pth'
            )
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.models[model_type].load_state_dict(checkpoint['model_state_dict'])
                print(f"已加载 {model_type} 的最佳权重")
        
        # 创建集成模型
        ensemble_model = ModelEnsemble(
            models=list(self.models.values()),
            weights=weights
        )
        
        return ensemble_model
    
    def evaluate_ensemble(self, ensemble_model=None, weights=None):
        """评估集成模型"""
        if ensemble_model is None:
            ensemble_model = self.create_ensemble(weights)
        
        ensemble_model.eval()
        ensemble_model.to(self.device)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 集成预测
                output = ensemble_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        # 计算准确率
        correct = sum(p == t for p, t in zip(all_predictions, all_targets))
        accuracy = correct / len(all_targets)
        
        return accuracy, all_predictions, all_targets
    
    def voting_ensemble(self, method='hard'):
        """投票集成"""
        print(f"使用{method}投票进行集成预测...")
        
        all_model_predictions = {}
        all_targets = []
        
        # 获取每个模型的预测
        for model_type, model in self.models.items():
            model.eval()
            model.to(self.device)
            
            predictions = []
            probabilities = []
            
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    output = model(data)
                    
                    if method == 'hard':
                        pred = output.argmax(dim=1, keepdim=True)
                        predictions.extend(pred.cpu().numpy().flatten())
                    else:  # soft voting
                        prob = torch.softmax(output, dim=1)
                        probabilities.extend(prob.cpu().numpy())
                    
                    if model_type == self.model_types[0]:  # 只需要保存一次targets
                        all_targets.extend(target.cpu().numpy())
            
            if method == 'hard':
                all_model_predictions[model_type] = predictions
            else:
                all_model_predictions[model_type] = probabilities
        
        # 进行投票
        if method == 'hard':
            # 硬投票
            ensemble_predictions = []
            for i in range(len(all_targets)):
                votes = [all_model_predictions[model_type][i] for model_type in self.model_types]
                # 选择得票最多的类别
                ensemble_pred = max(set(votes), key=votes.count)
                ensemble_predictions.append(ensemble_pred)
        else:
            # 软投票
            ensemble_predictions = []
            num_samples = len(all_targets)
            num_classes = self.config.NUM_CLASSES
            
            for i in range(num_samples):
                avg_probs = np.zeros(num_classes)
                for model_type in self.model_types:
                    avg_probs += all_model_predictions[model_type][i]
                avg_probs /= len(self.model_types)
                ensemble_pred = np.argmax(avg_probs)
                ensemble_predictions.append(ensemble_pred)
        
        # 计算准确率
        correct = sum(p == t for p, t in zip(ensemble_predictions, all_targets))
        accuracy = correct / len(all_targets)
        
        print(f"{method}投票集成准确率: {accuracy:.4f}")
        
        return accuracy, ensemble_predictions, all_targets
    
    def weighted_ensemble(self, weights=None):
        """加权集成"""
        if weights is None:
            # 使用验证集准确率作为权重
            weights = []
            for model_type in self.model_types:
                if model_type in self.histories:
                    val_acc = max(self.histories[model_type]['val_acc'])
                    weights.append(val_acc)
                else:
                    weights.append(1.0)
            
            # 归一化权重
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        print(f"使用权重: {dict(zip(self.model_types, weights))}")
        
        # 创建加权集成模型
        ensemble_model = self.create_ensemble(weights)
        accuracy, predictions, targets = self.evaluate_ensemble(ensemble_model)
        
        print(f"加权集成准确率: {accuracy:.4f}")
        
        return accuracy, predictions, targets
    
    def compare_methods(self):
        """比较不同集成方法"""
        print("\n比较不同集成方法:")
        print("=" * 50)
        
        results = {}
        
        # 单个模型结果
        print("\n单个模型结果:")
        for model_type in self.model_types:
            if model_type in self.histories:
                best_val_acc = max(self.histories[model_type]['val_acc'])
                print(f"{model_type:15s}: {best_val_acc:.4f}")
                results[model_type] = best_val_acc
        
        # 硬投票
        hard_acc, _, _ = self.voting_ensemble('hard')
        results['hard_voting'] = hard_acc
        
        # 软投票
        soft_acc, _, _ = self.voting_ensemble('soft')
        results['soft_voting'] = soft_acc
        
        # 加权集成
        weighted_acc, _, _ = self.weighted_ensemble()
        results['weighted_ensemble'] = weighted_acc
        
        # 找出最佳方法
        best_method = max(results, key=results.get)
        best_acc = results[best_method]
        
        print(f"\n最佳方法: {best_method}, 准确率: {best_acc:.4f}")
        
        return results, best_method

class ModelEnsemble(nn.Module):
    """模型集成类"""
    
    def __init__(self, models, weights=None):
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights is not None else [1.0] * len(models)
        
        # 归一化权重
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # 加权平均
        ensemble_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            ensemble_output += self.weights[i] * output
        
        return ensemble_output
