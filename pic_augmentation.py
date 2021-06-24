# 把图片水平翻转,竖直翻转,灰度化,并保存
import pandas as pd
import torchvision.transforms as transforms # (H, W, C) -> (C, H, W)
import torch
import cv2
import numpy as np
from PIL import Image
import os

save_path = 'pic_aug'
if not os.path.exists(save_path):
    os.mkdir(save_path)

for i in range(1290):
    # 规定图片名字
    pic_name_d = 'pic/' + str(i) + '_d.jpg'                     # 延迟量原图
    pic_name_d_h = save_path + '/' + str(i) + '_d_h.jpg'        # 延迟量水平翻转
    pic_name_d_v = save_path + '/' + str(i) + '_d_v.jpg'        # 延迟量垂直翻转
    pic_name_d_g = save_path + '/' + str(i) + '_d_g.jpg'        # 延迟量灰度图
    pic_name_d_b = save_path + '/' + str(i) + '_d_b.jpg'        # 延迟量黑白图

    pic_name_p = 'pic/' + str(i) + '_p.jpg'                     # 偏振角原图
    pic_name_p_h = save_path + '/' + str(i) + '_p_h.jpg'        # 偏振角水平翻转
    pic_name_p_v = save_path + '/' + str(i) + '_p_v.jpg'        # 偏振角垂直翻转
    pic_name_p_g = save_path + '/' + str(i) + '_p_g.jpg'        # 偏振角灰度图
    pic_name_p_b = save_path + '/' + str(i) + '_p_b.jpg'        # 偏振角黑白图


    # Image库图片读取(用于翻转transforms)
    img_d = Image.open(pic_name_d)
    img_p = Image.open(pic_name_p)
    # 延迟量(水平,垂直翻转)
    aug_im_d_h = transforms.RandomHorizontalFlip(p=1)(img_d)    # p表示概率,水平翻转
    aug_im_d_v = transforms.RandomVerticalFlip(p=1)(img_d)
    aug_im_d_h.save(pic_name_d_h)
    aug_im_d_v.save(pic_name_d_v)
    # 偏振角(水平,垂直翻转)
    aug_im_p_h = transforms.RandomHorizontalFlip(p=1)(img_p)    # p表示概率,水平翻转
    aug_im_p_v = transforms.RandomVerticalFlip(p=1)(img_p)
    aug_im_p_h.save(pic_name_p_h)
    aug_im_p_v.save(pic_name_p_v)

    # cv2库图片读取(用于灰度图与黑白图)
    img_d_cv = cv2.imread(pic_name_d)
    img_p_cv = cv2.imread(pic_name_p)
    # 延迟量(灰度图,黑白图)
    gray_d = cv2.cvtColor(img_d_cv, cv2.COLOR_BGR2GRAY)
    blurred_d = cv2.GaussianBlur(gray_d, (5, 5), 0)
    img_d_binary = cv2.threshold(blurred_d, 60, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(pic_name_d_g, gray_d)
    cv2.imwrite(pic_name_d_b, img_d_binary)

    # 偏振角本来就是靠颜色区分的,所以不用灰度化和二值化了
    # # 偏振角(灰度图,黑白图)
    # gray_p = cv2.cvtColor(img_p_cv, cv2.COLOR_BGR2GRAY)
    # blurred_p = cv2.GaussianBlur(gray_p, (5, 5), 0)
    # img_p_binary = cv2.threshold(blurred_p, 60, 255, cv2.THRESH_BINARY)[1]
    # cv2.imwrite(pic_name_p_g, gray_p)
    # cv2.imwrite(pic_name_p_b, img_p_binary)
