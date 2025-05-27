# -*- coding: utf-8 -*-
"""
数据加载和预处理模块
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
from config import Config

class DataManager:
    """数据管理类，负责数据加载、预处理和增强"""
    
    def __init__(self, config=Config):
        self.config = config
        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()
        
    def _get_train_transform(self):
        """获取训练数据的变换"""
        transform_list = [
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.RandomRotation(self.config.AUGMENTATION['rotation']),
            transforms.RandomHorizontalFlip(p=self.config.AUGMENTATION['horizontal_flip']),
            transforms.RandomVerticalFlip(p=self.config.AUGMENTATION['vertical_flip']),
            transforms.ColorJitter(
                brightness=self.config.AUGMENTATION['brightness'],
                contrast=self.config.AUGMENTATION['contrast'],
                saturation=self.config.AUGMENTATION['saturation'],
                hue=self.config.AUGMENTATION['hue']
            ),
            transforms.RandomResizedCrop(
                self.config.IMAGE_SIZE, 
                scale=(self.config.AUGMENTATION['random_crop'], 1.0)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet标准化
                std=[0.229, 0.224, 0.225]
            )
        ]
        
        # 添加高斯模糊（随机应用）
        if self.config.AUGMENTATION['gaussian_blur'] > 0:
            transform_list.insert(-2, transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=self.config.AUGMENTATION['gaussian_blur']))
        
        return transforms.Compose(transform_list)
    
    def _get_test_transform(self):
        """获取测试数据的变换"""
        return transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def get_data_loaders(self):
        """获取数据加载器"""
        # 检查是否存在单独的验证集
        if os.path.exists(self.config.VALID_DIR):
            # 使用单独的验证集
            train_dataset = datasets.ImageFolder(
                root=self.config.TRAIN_DIR, 
                transform=self.train_transform
            )
            valid_dataset = datasets.ImageFolder(
                root=self.config.VALID_DIR, 
                transform=self.test_transform
            )
            test_dataset = datasets.ImageFolder(
                root=self.config.TEST_DIR, 
                transform=self.test_transform
            )
        else:
            # 从训练集中分割验证集
            full_train_dataset = datasets.ImageFolder(
                root=self.config.TRAIN_DIR, 
                transform=self.train_transform
            )
            
            # 计算分割大小
            total_size = len(full_train_dataset)
            valid_size = int(total_size * self.config.VALIDATION_SPLIT)
            train_size = total_size - valid_size
            
            # 分割数据集
            train_dataset, valid_dataset = random_split(
                full_train_dataset, 
                [train_size, valid_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # 为验证集设置不同的变换
            valid_dataset.dataset = datasets.ImageFolder(
                root=self.config.TRAIN_DIR, 
                transform=self.test_transform
            )
            
            test_dataset = datasets.ImageFolder(
                root=self.config.TEST_DIR, 
                transform=self.test_transform
            )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if self.config.DEVICE.type == 'cuda' else False
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if self.config.DEVICE.type == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if self.config.DEVICE.type == 'cuda' else False
        )
        
        return train_loader, valid_loader, test_loader
    
    def get_class_names(self):
        """获取类别名称"""
        dataset = datasets.ImageFolder(root=self.config.TRAIN_DIR)
        return dataset.classes
    
    def get_dataset_info(self):
        """获取数据集信息"""
        train_dataset = datasets.ImageFolder(root=self.config.TRAIN_DIR)
        test_dataset = datasets.ImageFolder(root=self.config.TEST_DIR)
        
        info = {
            'num_classes': len(train_dataset.classes),
            'class_names': train_dataset.classes,
            'train_size': len(train_dataset),
            'test_size': len(test_dataset)
        }
        
        if os.path.exists(self.config.VALID_DIR):
            valid_dataset = datasets.ImageFolder(root=self.config.VALID_DIR)
            info['valid_size'] = len(valid_dataset)
        
        return info
