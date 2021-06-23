# 把一张大图切成很多小图并保存,自定义像素切割大小
import cv2 as cv
import os
row = 6
column = 5

org_img = cv.imread('contour.png')
height, width = org_img.shape[:2]
# 切之前图片大小
print('height %d widht %d' % (height, width))

row_step = (int)(height/row)
column_step = (int)(width/column)

print('row step %d, col step %d'% (row_step, column_step))
# 切之后图片大小
print('height %d widht %d' % (row_step*row, column_step*column))

img = org_img[0:row_step*row, 0:column_step*column]

save_path = 'split'
if not os.path.exists(save_path):
    os.mkdir(save_path)
for i in range(row):
    for j in range(column):
        pic_name = save_path + '/' + str(i) + "_" + str(j) + ".jpg"
        tmp_img = img[(i*row_step):(i*row_step+row_step), (j*column_step):(j*column_step)+column_step]
        cv.imwrite(pic_name, tmp_img)
