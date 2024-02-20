import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import os
import numpy as np
from network import DeepSeparator
from torch.optim.lr_scheduler import StepLR

# 设定工作目录为当前文件所在目录
os.chdir(os.path.dirname(__file__))

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# 定义超参数
BATCH_SIZE = 1000
learning_rate = 1e-4
epochs = 1000
mini_loss = float('inf')  # 初始化为无穷大

# 加载数据集
raw_eeg = np.load('../data/train_input.npy')
clean_eeg = np.load('../data/train_output.npy')
val_input = np.load('../data/val_input.npy')  # 加载验证集输入
val_output = np.load('../data/val_output.npy')  # 加载验证集输出
test_input = np.load('../data/test_input.npy')
test_output = np.load('../data/test_output.npy')

# 数据预处理
val_input = torch.from_numpy(val_input).float()
val_output = torch.from_numpy(val_output).float()

# 创建DataLoader
train_input = torch.from_numpy(raw_eeg).float()
train_output = torch.from_numpy(clean_eeg).float()
train_torch_dataset = Data.TensorDataset(train_input, train_output)
train_loader = Data.DataLoader(dataset=train_torch_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_torch_dataset = Data.TensorDataset(val_input, val_output)
val_loader = Data.DataLoader(dataset=val_torch_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型、损失函数和优化器
model = DeepSeparator().to(device)
loss = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # 学习率调度器

# 模型训练循环
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        train_loss = loss(outputs, targets)
        train_loss.backward()
        optimizer.step()
        total_train_loss += train_loss.item()
    
    # 计算平均训练损失
    avg_train_loss = total_train_loss / len(train_loader)
    
    # 模型验证
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss = loss(outputs, targets)
            total_val_loss += val_loss.item()
    
    # 计算平均验证损失
    avg_val_loss = total_val_loss / len(val_loader)
    
    # 打印训练和验证损失
    if epoch % print_loss_frequency == 0:
        print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # 保存性能最佳的模型
    if avg_val_loss < mini_loss:
        print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(mini_loss, avg_val_loss))
        torch.save(model.state_dict(), 'checkpoint/' + model_name + '.pt')
        mini_loss = avg_val_loss
    
    scheduler.step()

