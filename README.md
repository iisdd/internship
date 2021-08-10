核心函数: 

split.py(输入大图切成小图保存,输出切出的小图数量),

txt2csv.py(把包含label y的txt转成csv), 

gen_dataset.py(把图片和label都转成tensor,然后切成训练集和测试集), 

model.py(CNN模型), 

train.py(训练并画出中间层特征图), 

visualize.py(可视化模块,用TSN-E把CNN抽出来的特征降维成2维进行聚类展示), 

main.py(把以上功能集成到了主程序里,一键训练出图,只需要改文件地址).


## 聚类结果图
![image](https://github.com/iisdd/internship/blob/main/upload_pic/3%20bit(plr).jpg)




# 2021-06-10~
记录一下暑假实习做的项目

6-10 来公司报道,转到图像处理组

6-11 上午阅读5D玻璃存储技术,通过x,y,z坐标,光强和偏振角记录信息,下午参观量子实验室,明天放假(端午)

6-15 开始学习opencv图像操作,包括: 
cv2读取图片(cv2.imread),从图片中切出指定部分(img[y0:y1, x0:x1]),像素坐标都是从上到下从左到右,先y后x,图像保存与导出(cv2.imwrite),
图片格式转换(RGB,BGR,GRAY)(cv2.cvtColor),高斯模糊(cv2.GaussianBlur),二值化(cv2.threshold),找轮廓(cv2.findContours),画灰度图要加cmap='gray'

6-16 拿到真实数据(延迟量二分类),分割处理图片,先手动把左上角和右下角的无用信息截掉,然后把大图分割成很多小图,提取标签y,将其保存为.pt文件,为之后的网络训练准备数据
torchvision.transforms.ToTensor()可以把通道放第一位(H, W, C) -> (C, H, W)

6-18 搭建CNN模型,把数据集分为训练集与测试集,训练模型,尝试了图片增强(上下翻转,左右翻转)与灰度图,效果一般

6-21 新数据出现了一个问题,有极个别点用findContours找不出轮廓,尝试加入检验程序,如果相邻中心点距离超过75pt,就认为有点缺失,用线性插值补足

6-22 对新数据建模,发现偏振角模型准确率不足50%,手动排查数据后发现是图片和标签对不上,那没事了

6-23 整理之前的代码,简化上传github,制作整体流程图&CNN原理PPT

6-24 拿掉CNN最后一层,把原来FC的输入拿出来做降维,把分类结果可视化,偏振角的图可以分成8类(大小2 * 颜色4)




