import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, 
                out_channels=8, 
                kernel_size=5, 
                stride=1, 
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        ) 
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out = nn.Linear(40 * 40 * 64, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def main():
    # 数据预处理
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
    ])
    #训练数据集
    train_dataset = datasets.ImageFolder(root='Rock Data/train', transform=data_transforms)
    #测试数据集
    test_dataset = datasets.ImageFolder(root='Rock Data/test', transform=data_transforms)
    #数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnn = CNN()
    #选择交叉熵为损失函数
    loss_func = nn.CrossEntropyLoss()
    #优化器，学习率为0.002
    optimizer = optim.Adam(cnn.parameters(), lr = 0.002)
    #选择运行设备，在gpu可用时使用gpu，否则使用cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn.to(device)

    # 训练参数
    num_epochs = 30
    correct, total = 0, 0
    accuracy_list = []
    loss_list = []
    test_accuracy_list = []
    max_test_accuracy = 0
    
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()    # 清理未使用的显存缓存
        running_loss = 0.0
        correct, total = 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = cnn(inputs)            
            loss = loss_func(outputs, labels)           
            loss.backward()          
            optimizer.step()       
            running_loss += loss.item() * inputs.size(0)

            # 统计训练准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        loss_list.append(running_loss)
        accuracy_list.append(accuracy)
        # 清零正确数与总数，统计测试数据集的准确率
        correct, total = 0, 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = cnn(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # 输出统计信息
        test_accuracy = correct / total
        test_accuracy_list.append(test_accuracy)
        max_test_accuracy = test_accuracy if test_accuracy > max_test_accuracy else max_test_accuracy
        print('epoch: ', epoch+1, '\t|\t', 
              'train accuracy: ', percent(accuracy) , '\t|\t', 
              'loss: ', running_loss, '\t|\t', 
              'test accuracy: ', percent(test_accuracy))


    fig = plt.figure('Visual Data', figsize=(16,8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.set_title('Loss Tendency')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot([i for i in range(1, len(loss_list)+1)], loss_list, 'b-')

    ax2.set_title('Accuracy Tendency')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.plot([i for i in range(1, len(accuracy_list)+1)], accuracy_list, 'b-', label='Train Accuracy')
    ax2.plot([i for i in range(1, len(test_accuracy_list)+1)], test_accuracy_list, 'g-', label='Test Accuracy')
    ax2.legend(loc='best')

    plt.show()



def percent(value):
    s = str(value * 100)
    if len(s) > 5: s = s[:5]
    return s + '%'

if __name__ == "__main__":
    main()