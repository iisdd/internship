# 处理输入图片与标签数据,将其保存为tensor的.pt文件
import pandas as pd
import torchvision.transforms as transforms # (H, W, C) -> (C, H, W)
import torch
import cv2
import numpy as np
from PIL import Image
import os


# 尝试把两个图堆在一起
# pic_path = 'pic/0.jpg'
# img = cv2.imread(pic_path)
# print(img.shape)            # (20, 20, 3)
#
# transf = transforms.ToTensor()
# img_tensor = transf(img)    # tensor数据格式是torch(C,H,W)
# print(img_tensor.size())    # (3, 20, 20)
# print(img_tensor.unsqueeze(0).shape)
#
#
# pic_path = 'pic/1.jpg'
# img = cv2.imread(pic_path)
# print(img.shape)            # (20, 20, 3)
#
# transf = transforms.ToTensor()
# img_tensor2 = transf(img)    # tensor数据格式是torch(C,H,W)
# print(img_tensor2.size())    # (3, 20, 20)
# print(img_tensor2.unsqueeze(0).shape)
#
# print(torch.cat((img_tensor.unsqueeze(0), img_tensor2.unsqueeze(0)), 0).shape)


save_path = 'pt_file'
if not os.path.exists(save_path):
    os.mkdir(save_path)


# 标签保存
data = pd.read_csv('y.csv')
delay = data['delay']
plr = data['plr']

delay = delay.to_numpy()        # df -> np -> tensor
delay = torch.tensor(delay)
torch.save(delay, save_path + '/' + 'delay_label.pt')
print('延迟量标签pt保存完毕')

plr = plr.to_numpy()
plr = torch.tensor(plr)
torch.save(plr, save_path + '/' + 'plr_label.pt')
print('偏振角标签pt保存完毕')


# 彩色图读取,存储延迟量图片
# 原图
x = None
transf = transforms.ToTensor()
for i in range(1290):
    pic_path = 'pic' + '/%s_d.jpg' % str(i)
    tmp_img = cv2.imread(pic_path)
    tmp_img = transf(tmp_img).unsqueeze(0)
    if x == None:
        x = tmp_img
        continue
    x = torch.cat((x, tmp_img), 0)  # axis=0: 竖着堆叠
print('延迟量原图pt保存完毕')
torch.save(x, save_path + '/' + 'delay_image.pt')


# 水平翻转
x = None
transf = transforms.ToTensor()
for i in range(1290):
    pic_path = 'pic_aug' + '/%s_d_h.jpg' % str(i)
    tmp_img = cv2.imread(pic_path)
    tmp_img = transf(tmp_img).unsqueeze(0)
    if x == None:
        x = tmp_img
        continue
    x = torch.cat((x, tmp_img), 0)  # axis=0: 竖着堆叠
print('延迟量水平翻转pt保存完毕')
torch.save(x, save_path + '/' + 'delay_image_h.pt')


# 垂直翻转
x = None
transf = transforms.ToTensor()
for i in range(1290):
    pic_path = 'pic_aug' + '/%s_d_v.jpg' % str(i)
    tmp_img = cv2.imread(pic_path)
    tmp_img = transf(tmp_img).unsqueeze(0)
    if x == None:
        x = tmp_img
        continue
    x = torch.cat((x, tmp_img), 0)  # axis=0: 竖着堆叠
print('延迟量垂直翻转pt保存完毕')
torch.save(x, save_path + '/' + 'delay_image_v.pt')


# 灰度图
x = None
transf = transforms.ToTensor()
for i in range(1290):
    pic_path = 'pic_aug' + '/%s_d_g.jpg' % str(i)
    tmp_img = cv2.imread(pic_path)
    tmp_img = transf(tmp_img).unsqueeze(0)
    if x == None:
        x = tmp_img
        continue
    x = torch.cat((x, tmp_img), 0)  # axis=0: 竖着堆叠
print('延迟量灰度图pt保存完毕')
torch.save(x, save_path + '/' + 'delay_image_g.pt')


# 黑白图
x = None
transf = transforms.ToTensor()
for i in range(1290):
    pic_path = 'pic_aug' + '/%s_d_b.jpg' % str(i)
    tmp_img = cv2.imread(pic_path)
    tmp_img = transf(tmp_img).unsqueeze(0)
    if x == None:
        x = tmp_img
        continue
    x = torch.cat((x, tmp_img), 0)  # axis=0: 竖着堆叠
print('延迟量黑白图pt保存完毕')
torch.save(x, save_path + '/' + 'delay_image_b.pt')



# 彩色图读取,存储偏振角图片
# 原图
x = None
transf = transforms.ToTensor()
for i in range(1290):
    pic_path = 'pic' + '/%s_p.jpg' % str(i)
    tmp_img = cv2.imread(pic_path)
    tmp_img = transf(tmp_img).unsqueeze(0)
    if x == None:
        x = tmp_img
        continue
    x = torch.cat((x, tmp_img), 0)  # axis=0: 竖着堆叠
print('偏振角原图pt保存完毕')
torch.save(x, save_path + '/' + 'plr_image.pt')


# 水平翻转
x = None
transf = transforms.ToTensor()
for i in range(1290):
    pic_path = 'pic_aug' + '/%s_p_h.jpg' % str(i)
    tmp_img = cv2.imread(pic_path)
    tmp_img = transf(tmp_img).unsqueeze(0)
    if x == None:
        x = tmp_img
        continue
    x = torch.cat((x, tmp_img), 0)  # axis=0: 竖着堆叠
print('偏振角水平翻转pt保存完毕')
torch.save(x, save_path + '/' + 'plr_image_h.pt')


# 垂直翻转
x = None
transf = transforms.ToTensor()
for i in range(1290):
    pic_path = 'pic_aug' + '/%s_p_v.jpg' % str(i)
    tmp_img = cv2.imread(pic_path)
    tmp_img = transf(tmp_img).unsqueeze(0)
    if x == None:
        x = tmp_img
        continue
    x = torch.cat((x, tmp_img), 0)  # axis=0: 竖着堆叠
print('偏振角垂直翻转pt保存完毕')
torch.save(x, save_path + '/' + 'plr_image_v.pt')