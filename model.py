# 单独包装模型,方便所有程序调用
import torch
import torch.nn as nn

# 建立CNN网络
class CNN(nn.Module):
    def __init__(self, n_o, last_layer_px=5):
        super(CNN , self).__init__()
        self.n_o = n_o
        self.last_layer_px = last_layer_px  # 最后一层池化完像素变成多少
        # 第一层
        self.conv1 = nn.Sequential(
            nn.Conv2d(                      # 卷积层, (3, 20, 20), 厚度放前面了
                in_channels = 3,            # 输入的厚度为3
                out_channels = 16,          # 16个filters , 相当于把图片分成16个部分来提取特征
                kernel_size = 5,            # filter的宽度为5个像素点
                stride = 1,                 # 滑动步长为1个像素点
                padding = 2,                # 边界填充2圈0
                ),                          # -> (16, 20, 20)
            # torch.nn.Dropout(0.25),
            nn.ReLU(),                      # 激活层  -> (16, 20, 20)
            nn.MaxPool2d(kernel_size = 2),  # 池化层,选2*2的最大值, -> (16, 10, 10)
            )
        # 第二层
        self.conv2 = nn.Sequential(         # (16, 10, 10)
            nn.Conv2d(16, 32, 5, 1, 2),     # -> (32, 10, 10)
            # torch.nn.Dropout(0.25),
            nn.ReLU(),                      # -> (32, 10, 10)
            nn.MaxPool2d(2),                # -> (32, 5, 5)
            )
        # 输出层,全连接
        self.out = nn.Linear(32*self.last_layer_px*self.last_layer_px, self.n_o)

    def forward(self, x):
        per_out = []                        # 记录每层的特征输出
        x = self.conv1(x)
        per_out.append(x)
        x = self.conv2(x)                   # (batch, 32, 5, 5)
        per_out.append(x)
        x = x.view(x.size(0) , -1)          # (batch, 32 * 5 * 5), view():降维拉平
        output = self.out(x)
        return output, x, per_out           # 返回最后一层长向量x为了做聚类可视化