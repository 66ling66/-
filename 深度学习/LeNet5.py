import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def plot_losses(losses,name):
    plt.plot(range(len(losses)),losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'Test/{name}.png')
    print(f'Loss plot saved to Test/{name}.png')
#
# 定义 LeNet-5 模型
# class LeNet5(nn.Module):
#     def __init__(self):
#         super(LeNet5, self).__init__()
#         # 定义卷积层
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
#         self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
#         self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
#
#         # 定义全连接层
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = torch.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = x.view(-1, 16 * 5 * 5)  # Flatten the output
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#
# # 数据集的预处理
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据集的均值和标准差
# ])
#
# # 加载 MNIST 数据集
# trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
#
# testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# testloader = DataLoader(testset, batch_size=1000, shuffle=False)
#
# # 实例化模型、定义损失函数和优化器
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = LeNet5().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#
#
# # 训练模型
# def train_model(model, trainloader, criterion, optimizer, epochs=10):
#     model.train()
#     losses = []
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for inputs, labels in trainloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         losses.append(running_loss / len(trainloader) )
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}")
#     return losses
#
#
# # 测试模型
# def test_model(model, testloader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in testloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
#
#     print(f"Test Accuracy: {100 * correct / total:.2f}%")
#
#
# # 执行训练和测试
# plot_losses(train_model(model, trainloader, criterion, optimizer, epochs=10),'lenet5mnist')
# test_model(model, testloader)



# ---------------------------LeNet5 cifar10 -------------------------
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
#
#
# # 修改后的 LeNet-5 模型
# class LeNet5(nn.Module):
#     def __init__(self):
#         super(LeNet5, self).__init__()
#         # 修改第一个卷积层的输入通道数为 3
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
#         self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
#         self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
#
#         # 全连接层保持不变
#         self.fc1 = nn.Linear(16 * 6 * 6, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = torch.relu(self.conv2(x))
#         x = self.pool2(x)
#
#         x = x.view(-1, 16 * 6 * 6)  # Flatten the output
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#
# # 数据集的预处理
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#
# # 加载 CIFAR-10 数据集
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = DataLoader(testset, batch_size=1000, shuffle=False)
#
# # 实例化模型、定义损失函数和优化器
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = LeNet5().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
#
#
# # 训练模型
# def train_model(model, trainloader, criterion, optimizer, epochs=10):
#     model.train()
#     losses = []
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for inputs, labels in trainloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         losses.append(running_loss / len(trainloader) )
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}")
#     return losses
#
# # 测试模型
# def test_model(model, testloader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in testloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
#
#     print(f"Test Accuracy: {100 * correct / total:.2f}%")
#
#
# # 执行训练和测试
# plot_losses(train_model(model, trainloader, criterion, optimizer, epochs=50),'LeNet5cifar10.png')
# test_model(model, testloader)

# ---------------------------AlexNet Mnist -------------------------

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
#
#
# # 定义适用于 MNIST 的 AlexNet 模型
# class AlexNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),  # 将输入通道数改为 1
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#
#         # 全连接层适配 MNIST 数据
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),  # 如果输入图像大小调整到 224x224，这里可以保持不变
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)  # Flatten the output
#         x = self.classifier(x)
#         return x
#
#
# # 数据集的预处理
# transform = transforms.Compose([
#     transforms.Resize(224),  # 将 MNIST 图像大小调整为 224x224
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 的均值和标准差
# ])
#
# # 加载 MNIST 数据集
# trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
#
# testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# testloader = DataLoader(testset, batch_size=1000, shuffle=False)
#
# # 实例化模型、定义损失函数和优化器
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AlexNet(num_classes=10).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
#
#
# # 训练模型
# def train_model(model, trainloader, criterion, optimizer, epochs=10):
#     model.train()
#     losses = []
#     for epoch in range(epochs):
#         running_loss = 0.0
#
#         for inputs, labels in trainloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         losses.append(running_loss / len(trainloader))
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}")
#     return losses
#
# # 测试模型
# def test_model(model, testloader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in testloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
#
#     print(f"Test Accuracy: {100 * correct / total:.2f}%")
#
#
# # 执行训练和测试
# plot_losses(train_model(model, trainloader, criterion, optimizer, epochs=10),'Alexnetmnist')
# test_model(model, testloader)



# ---------------------------AlexNet Cifar10-------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# 定义适合 CIFAR-10 的 AlexNet 模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 使用 3x3 卷积核和较小的步幅
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 修改全连接层适配 CIFAR-10 的 32x32 输入
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 1024),  # 扁平化后维度为 256 * 2 * 2
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output

        x = self.classifier(x)
        return x


# 数据集的预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1000, shuffle=False)

# 实例化模型、定义损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)


# 训练模型
def train_model(model, trainloader, criterion, optimizer, epochs=10):
    model.train()
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        losses.append(running_loss / len(trainloader))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}")
    return losses

# 测试模型
def test_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


# 执行训练和测试
plot_losses(train_model(model, trainloader, criterion, optimizer, epochs=50),'AlexNetcifar10')
test_model(model, testloader)
