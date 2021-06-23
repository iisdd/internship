# 找出图片中的轮廓,可以提取形状特征
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

img = cv2.imread('contour.png') # imread默认读出来格式是BGR
img_BGR = img.copy()            # 留一份全彩的备份
plt.imshow(img_BGR)
plt.title('BGR')
plt.show()

# RGB才是原图的格式
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_RGB)
plt.title('RGB')
plt.show()

# 灰度图+高斯模糊+二值图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.title('after gray')
plt.imshow(gray, cmap='gray')
plt.show()

blurred = cv2.GaussianBlur(gray, (5,5), 0)                          # 高斯矩阵形状为(5,5), 标准差取0
plt.title('blurred pic')
plt.imshow(blurred, cmap='gray')
plt.show()

# cv2.threshold(源图片, 阈值, 填充色, 阈值类型), 255是上限
img_binary = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]  # return的第一个不关键
plt.title('after binary')
plt.imshow(img_binary, cmap='gray')
plt.show()

# findContours这个函数返回两个值: contours, hierarchy  其中contours是轮廓的列表
contours = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
print('找到轮廓数: ', len(contours))                 # 17
print('轮廓shape: ', contours[0].shape)             # (90, 1, 2),每个轮廓离散成90个点
# print('找到的第一个轮廓点坐标列表: ', contours[0])    # 轮廓不是从上到下从左到右找的

# 计算轮廓的中心点坐标
x = int(np.mean([x for x in contours[0][:,:,0]]))
y = int(np.mean([x for x in contours[0][:,:,1]]))
print('第一个轮廓的中心点坐标: ', (x,y)) # 239.49, 424.53



# 切出来的图形
# split_img = img[y-32:y+32, x-32:x+32]   # 先y后x
# plt.imshow(split_img)
# plt.title('first contour')
# plt.show()

center_list = []

for i,c in enumerate(contours):
    x_tmp = int(np.mean([x for x in c[:, :, 0]]))
    y_tmp = int(np.mean([x for x in c[:, :, 1]]))
    center_list.append([x_tmp, y_tmp])
    # split_img = img[y_tmp-50:y_tmp+50, x_tmp-50:x_tmp+50]   # 先写y后写x
    # plt.imshow(split_img)
    # plt.title('contour %s' % str(i+1))
    # plt.show()
print('原始中心点列表: ', center_list)

# 排序
def sort_li(li):
    # 输入一个列表[[x1, y1], [x2, y2], ..., ], 输出有序列表, x,y都从大到小排
    res = []
    li = sorted(li, key=lambda x: -x[1])                        # 倒序,从大到小
    n = len(li)
    i = 0
    head = li[0][1]                                             # 这一行内最大的y
    tmp_li = []
    while i < n:
        tmp_li.append(li[i])
        if i < n - 1 and head - li[i + 1][1] < 30:              # 距离小于30就算是同一行
            i += 1
            continue
        else:                                                   # 一行的最后一个
            res += sorted(tmp_li, key=lambda x: -x[0])          # 从大到小排
            tmp_li = []
            if i < n - 1:
                head = li[i + 1][1]
        i += 1
    return res

# 先按y从小到大排
order_center = sort_li(center_list)
print('排序后的中心点坐标: ', order_center)

# 画图
img = cv2.imread('contour.png')
save_path = 'find_contour'
if not os.path.exists(save_path):
    os.mkdir(save_path)
for i, (x, y) in enumerate(order_center):
    split_img = img[y-50:y+50, x-50:x+50]   # 先写y后写x
    pic_name = save_path + '/' + 'order_contour%s' % str(i+1) + '.jpg'
    cv2.imwrite(pic_name, split_img)
    plt.imshow(split_img)
    plt.title('order_contour%s' % str(i+1))
    plt.show()
