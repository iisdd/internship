# 包装一下训练的函数
import cv2
import matplotlib.pyplot as plt
import os

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from utils import MyDataset, save_models, load_models
from model import CNN
import torch.utils.data as data
import torch
import torch.nn as nn

def train(train_x, test_x, train_y, test_y, pic_px, n_o,
          lr=0.001, batch_size=16, epoch=5, type='plr'):
    # 前4个输入为划分出的训练&测试集,pic_px为输入图片的像素宽度,n_o为分类的数量
    # lr为学习率,batch_size为每次训练抽取的批数量,epoch为训练轮数,type为模型类别
    # 定个种子复现
    torch.manual_seed(123)
    print('训练集大小: ', train_x.shape, train_y.shape)
    print('测试集大小: ', test_x.shape, test_y.shape)
    dataset = MyDataset(train_x, train_y)
    train_loader = data.DataLoader(dataset = dataset , batch_size = batch_size , shuffle = True)

    cnn = CNN(n_o=n_o, last_layer_px=(pic_px//2)//2)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    Error = []      # 训练过程中的错误率变化
    Loss = []       # 训练过程中的损失变化
    for e in range(epoch):
        for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data, normalize x when iterate train_loader
            output = cnn(b_x)[0]                                # cnn output
            loss = loss_func(output, b_y)                       # cross entropy loss
            Loss.append(loss)
            optimizer.zero_grad()                               # clear gradients for this training step
            loss.backward()                                     # backpropagation, compute gradients
            optimizer.step()                                    # apply gradients

            test_output, _, _ = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            Error.append((1- accuracy)*100)
            print('Epoch: ', e, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)
    save_models(cnn, type=type)


    ################################### 画中间特征图 #################################
    img = train_x[10]                       # (C, H, W)
    img = np.transpose(img, (1, 2, 0))      # (H, W, C)
    plt.imshow(img)
    plt.title('input img')
    plt.show()
    mid_1 = cnn(train_x[10:11])[2][0]
    mid_1 = torch.squeeze(mid_1, 0)
    mid_1 = mid_1.detach().numpy()
    mid_2 = cnn(train_x[10:11])[2][1]
    mid_2 = torch.squeeze(mid_2, 0)
    mid_2 = mid_2.detach().numpy()


    plt.figure()                            # 卷一次特征
    plt.title('mid features')
    for i in range(1, 17):
        ax = plt.subplot(4, 4, i)
        plt.imshow(mid_1[i-1, :, :], cmap='gray')
    plt.show()

    plt.figure()                            # 卷两次特征
    plt.title('mid features')
    for i in range(1, 33):
        ax = plt.subplot(4, 8, i)
        plt.imshow(mid_2[i-1, :, :], cmap='gray')
    plt.show()
    ################################### 画中间特征图 #################################

    ################################## 画训练指标曲线 ################################
    np.save('Loss_plr.npy', Loss)
    plt.plot(Loss)
    plt.title('Training convergence', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.savefig('Loss.jpg', dpi=1000)
    plt.show()

    np.save('Test_error_plr.npy', Error)
    plt.plot(Error)
    plt.xlabel('Training steps', fontsize=16)
    plt.ylabel('Test error(%)', fontsize=16)
    plt.savefig('Error.jpg', dpi=1000)
    plt.show()
    return cnn  # 返回训练好的网络
    ################################## 画训练指标曲线 ################################