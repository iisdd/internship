# 生成训练集
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from sklearn.model_selection import train_test_split    # sklearn内置分割训练集
import pandas as pd
import torchvision.transforms as transforms # (H, W, C) -> (C, H, W)
import torch
import cv2
import numpy as np
from PIL import Image


# 图 -> tensor
def pic2tensor(pic_path, pic_num):
    # pic_path为小图存放的文件夹名, pic_num为小图数量
    delay_tensor = None
    plr_tensor = None
    transf = transforms.ToTensor()
    for i in range(1, pic_num+1):
        img_d = cv2.imread(pic_path + '/%s_d.jpg' % str(i))
        img_p = cv2.imread(pic_path + '/%s_p.jpg' % str(i))
        img_d = transf(img_d).unsqueeze(0)
        img_p = transf(img_p).unsqueeze(0)
        # 初始化
        if i == 1:
            delay_tensor = img_d
            plr_tensor = img_p
            continue
        # 堆叠
        delay_tensor = torch.cat((delay_tensor, img_d), 0)  # axis=0: 竖着堆叠
        plr_tensor = torch.cat((plr_tensor, img_p), 0)      # axis=0: 竖着堆叠
    return delay_tensor, plr_tensor

# 标签 -> tensor
def csv2tensor(filename):
    # filename为标签csv文件名
    data = pd.read_csv(filename)
    delay = data['delay']
    plr = data['plr']
    decode = data['decode']

    delay = delay.to_numpy()    # df -> np -> tensor
    delay = torch.tensor(delay)

    plr = plr.to_numpy()
    plr = torch.tensor(plr)

    decode = decode.to_numpy()
    decode = torch.tensor(decode)
    return delay, plr, decode           # 返回标签tensor

# 生成数据集
def gen_dataset(pic_path, pic_num, filename, type='plr', test_size=0.2):
    # 前三个参数喂入之前的两个函数,type选择返回的数据集类型,test_size代表测试集占总数据的比例
    x_d, x_p = pic2tensor(pic_path=pic_path, pic_num=pic_num)
    y_d, y_p, decode = csv2tensor(filename=filename)
    # 注意标签要取和图片一样多的
    if type == 'delay':                 # 返回的标签tensor和所有图片tensor用来输入给cnn出来长向量聚类
        # 保持一样长,不一定所有标签都用
        return x_d, y_d, decode, train_test_split(x_d, y_d[:x_d.shape[0]], test_size=test_size, random_state=14138)
    elif type == 'plr':
        return x_p, y_p, decode, train_test_split(x_p, y_p[:x_p.shape[0]], test_size=test_size, random_state=14138)
    elif type == 'both':                # 直接从偏振角图片中训练对应标签decode
        return x_p, y_p, decode, train_test_split(x_p, decode[:x_p.shape[0]], test_size=test_size, random_state=14138)
