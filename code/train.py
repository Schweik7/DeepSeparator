import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import os
import numpy as np
from network import DeepSeparator
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
os.chdir(os.path.dirname(__file__))

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


BATCH_SIZE = 1000
learning_rate = 1e-4
epochs = 1000

mini_loss = 1

print_loss_frequency = 1
print_train_accuracy_frequency = 1
val_frequency = 1


model_name = 'DeepSeparator'
model = DeepSeparator()
loss = nn.MSELoss(reduction='mean')

raw_eeg = np.load('../data/train_input.npy')
clean_eeg = np.load('../data/train_output.npy')

artifact1 = np.load('../data/EOG_all_epochs.npy')
artifact2 = np.load('../data/EMG_all_epochs.npy')

val_input = np.load('../data/val_input.npy')
val_output = np.load('../data/val_output.npy')

artifact1 = standardization(artifact1)
artifact2 = standardization(artifact2)
 # 合并两种伪迹为一个数组，并创建对应的指示器
artifact = np.concatenate((artifact1, artifact2), axis=0)
indicator_artifact = np.concatenate((np.ones(artifact1.shape[0]), np.ones(artifact2.shape[0]) * 2), axis=0)

# 创建训练数据的指示器，其中0表示原始EEG，1和2分别表示两种伪迹
indicator = np.concatenate((np.zeros(raw_eeg.shape[0]), indicator_artifact), axis=0)


train_input = np.concatenate((raw_eeg, artifact, clean_eeg), axis=0)
train_output = np.concatenate((clean_eeg, artifact, clean_eeg), axis=0)

indicator = torch.from_numpy(indicator)
indicator = indicator.unsqueeze(1)

train_input = torch.from_numpy(train_input)
train_output = torch.from_numpy(train_output)

train_torch_dataset = Data.TensorDataset(train_input, indicator, train_output)

train_loader = Data.DataLoader(
    dataset=train_torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_input = torch.from_numpy(val_input)
val_output = torch.from_numpy(val_output)

val_indicator = np.zeros(val_input.shape[0])
val_indicator = torch.from_numpy(val_indicator)
val_indicator = val_indicator.unsqueeze(1)

val_torch_dataset = Data.TensorDataset(val_input, val_indicator, val_output)

val_loader = Data.DataLoader(
    dataset=val_torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("torch.cuda.is_available() = ", torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1) #定义学习率调度器
if os.path.exists('checkpoint/' + model_name + '.pkl'):
    print('load model')
    model.load_state_dict(torch.load('checkpoint/' + model_name + '.pkl'))

for epoch in range(epochs):
    train_acc = 0
    train_loss = 0
    total_train_loss_per_epoch = 0
    average_train_loss_per_epoch = 0
    train_step_num = 0
    for step, (train_input, indicator, train_output) in enumerate(train_loader):
        train_step_num += 1
        indicator = indicator.float().to(device)
        train_input = train_input.float().to(device)
        train_output = train_output.float().to(device)
        optimizer.zero_grad()
        train_preds = model(train_input, indicator)
        train_loss = loss(train_preds, train_output)
        total_train_loss_per_epoch += train_loss.item()
        train_loss.backward()
        optimizer.step()
    average_train_loss_per_epoch = total_train_loss_per_epoch / train_step_num
    if epoch % print_loss_frequency == 0:
        print('train loss: ', average_train_loss_per_epoch)
    val_step_num = 0
    total_val_loss_per_epoch = 0
    average_val_loss_per_epoch = 0

    if epoch % val_frequency == 0:
        for step, (val_input, val_indicator, val_output) in enumerate(val_loader):
            val_step_num += 1
            val_indicator = val_indicator.float().to(device)
            val_input = val_input.float().to(device)
            val_output = val_output.float().to(device)
            val_preds = model(val_input, val_indicator)
            val_loss = loss(val_preds, val_output)
            total_val_loss_per_epoch += val_loss.item()

        average_val_loss_per_epoch = total_val_loss_per_epoch / val_step_num

        print('--------------val loss: ', average_val_loss_per_epoch)

        if average_val_loss_per_epoch < mini_loss:
            print('save model')
            torch.save(model.state_dict(), 'checkpoint/' + model_name + '.pkl')
            mini_loss = average_val_loss_per_epoch
    scheduler.step()