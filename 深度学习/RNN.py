# 导入必要的库
import torch  # PyTorch库，用于张量操作和自动求导
import torchvision  # 包含MNIST数据集的库
import torchvision.transforms as transforms  # 图像预处理模块
import torch.nn.functional as F  # 常用的激活函数和损失函数

# 设置设备，如果有GPU可用则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义超参数
input_size = 28  # 每个时间步的输入维度（MNIST每行有28个像素）
hidden_size = 128  # 隐藏层神经元数量
output_size = 10  # 输出类别数量（数字0到9）
sequence_length = 28  # 序列长度（MNIST每张图片有28行）
num_epochs = 2  # 训练轮数
batch_size = 100  # 批次大小
learning_rate = 0.001  # 学习率

# 下载并加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# 创建数据加载器，用于批量加载数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# 定义自定义的RNN类
class CustomRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化模型参数
        # 输入到隐藏层的权重和偏置
        self.W_ih = torch.randn(input_size, hidden_size, device=device, requires_grad=True) * 0.01
        self.b_ih = torch.zeros(hidden_size, device=device, requires_grad=True)

        # 隐藏层到隐藏层的权重和偏置
        self.W_hh = torch.randn(hidden_size, hidden_size, device=device, requires_grad=True) * 0.01
        self.b_hh = torch.zeros(hidden_size, device=device, requires_grad=True)

        # 隐藏层到输出层的权重和偏置
        self.W_ho = torch.randn(hidden_size, output_size, device=device, requires_grad=True) * 0.01
        self.b_ho = torch.zeros(output_size, device=device, requires_grad=True)

    def forward(self, x):
        # 前向传播函数
        # x的形状为(batch_size, sequence_length, input_size)
        batch_size = x.size(0)  # 获取批次大小
        h_t = torch.zeros(batch_size, hidden_size, device=device)  # 初始化隐藏状态为全零

        # 遍历序列中的每个时间步
        for t in range(sequence_length):
            x_t = x[:, t, :]  # 取出第t个时间步的输入，形状为(batch_size, input_size)
            # 计算隐藏状态
            h_t = torch.tanh(x_t @ self.W_ih + self.b_ih + h_t @ self.W_hh + self.b_hh)
            # torch.tanh是tanh激活函数，@表示矩阵乘法

        # 使用最后一个时间步的隐藏状态来计算输出
        output = h_t @ self.W_ho + self.b_ho  # 形状为(batch_size, output_size)
        return output  # 返回模型的输出

    def parameters(self):
        # 返回模型的参数列表，用于优化器更新参数
        return [self.W_ih, self.b_ih, self.W_hh, self.b_hh, self.W_ho, self.b_ho]


# 创建模型实例
model = CustomRNN(input_size, hidden_size, output_size)


# 定义损失函数，这里使用交叉熵损失函数
def cross_entropy_loss(outputs, labels):
    # outputs: 模型的输出，形状为(batch_size, output_size)
    # labels: 真实标签，形状为(batch_size)
    # 计算log_softmax
    log_probs = F.log_softmax(outputs, dim=1)
    # 选择正确类别的log概率
    selected_log_probs = log_probs[range(labels.size(0)), labels]
    # 返回负的平均log概率作为损失
    return -selected_log_probs.mean()


# 定义优化器，这里使用简单的随机梯度下降（SGD）
def sgd(params, lr):
    # params: 模型的参数列表
    # lr: 学习率
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad  # 更新参数
            param.grad.zero_()  # 清空梯度


# 获取模型的参数列表
params = model.parameters()

# 开始训练模型
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # images的原始形状为(batch_size, 1, 28, 28)
        # 需要将images调整为(batch_size, sequence_length, input_size)
        images = images.squeeze(1).to(device)  # 去除通道维度，形状变为(batch_size, 28, 28)
        images = images.reshape(-1, sequence_length, input_size)  # 调整形状
        labels = labels.to(device)  # 将标签移动到设备上

        # 前向传播
        outputs = model.forward(images)  # 得到模型输出，形状为(batch_size, output_size)
        loss = cross_entropy_loss(outputs, labels)  # 计算损失

        # 反向传播
        loss.backward()  # 计算梯度

        # 更新参数
        sgd(params, learning_rate)  # 使用SGD优化器更新模型参数

        # 每隔100个批次打印一次训练信息
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

# 在测试集上评估模型性能
with torch.no_grad():  # 禁用梯度计算
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.squeeze(1).to(device)
        images = images.reshape(-1, sequence_length, input_size)
        labels = labels.to(device)
        outputs = model.forward(images)
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
        total += labels.size(0)  # 累计样本数量
        correct += (predicted == labels).sum().item()  # 计算预测正确的数量

    print('测试集上的准确率为: {:.2f} %'.format(100 * correct / total))
