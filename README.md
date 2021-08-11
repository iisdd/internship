核心函数: 

split.py(输入大图切成小图保存,输出切出的小图数量),

txt2csv.py(把包含label y的txt转成csv), 

gen_dataset.py(把图片和label都转成tensor,然后切成训练集和测试集), 

model.py(CNN模型), 

train.py(训练并画出中间层特征图), 

visualize.py(可视化模块,用TSN-E把CNN抽出来的特征降维成2维进行聚类展示), 

main.py(把以上功能集成到了主程序里,一键训练出图,只需要改文件地址).


## 聚类结果图
![image](https://github.com/iisdd/internship/blob/main/upload_pic/3-bits.jpg)




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

6-28 拿到一批带噪声的图片,噪声和小点太像了,滤掉噪声会把点也滤掉,需要按坐标补插值

6-29 去实验室看打片子与显微镜出图过程

7-6 把实现功能的模块封装成函数,把一个很大的程序拆成几个子函数加一个主程序,实现一键训练

########################### 至此python部分图像分类告一段落,开始C++ #########################

7-7 程序交接&浏览micromanager源码程序结构

7-8 看MM官网说明&软件下载&环境配置

7-9 读DemoCamera源码(4000行真有你的!!!)

7-12 看源码,记录一下看到的杂七杂八的常识
* using namspace:命名空间,一些规定好的类名,比如using namspace std 就像 from std import * ,本来是std.xxx，现在就能直接用xxx，但是如果用了多个命名空间,变量名就容易起冲突，所以最好是from XXX import xxx, 也就是std::cin
* 关于include:include<>是包含标准的系统头文件名,只在标准头文件目录里找,include""是包含自己写的头文件，优先在当前目录的头文件里找
* 片段注释：ctrl+k -> 选中段落 -> ctrl+c, 取消注释：ctrl+k -> 选中段落 -> ctrl+u
* const定义变量，保证变量永远不变，后面接的东西都不能改（比如接指针const int * p = &a）这就固定了p只能指向a的内存地址，不能改
* cout:console out,cout << XXX << XXX << endl; 后面加endl代表结束此行并换行
* 在C++中 9/5=1, 9.0/5.0=1.8, 加.自动转成double型
* signed:-128 ~ 127, unsigned:0 ~ 255
* &:取地址, *p:解引用,取地址的值

7-13 生成动态链接库(dll),替换软件中源文件部分,观察软件中变化

7-14 实现打印日志,SDK:里面包括你想要的功能的函数包(ex:抓取图片,设置曝光时间...),API:SDK中函数的接口
* ->:通过结构体里的指针可以找到结构体中的属性
* 指针指向数组其实就是指向了第一个元素的地址

7-16 删掉没用的属性(只保留DemoCamera那个类),继续看源码打日志

7-19 阅读相机TUCam的开发手册(Dhyana系列)

7-20 安装相机驱动,连通相机并拍照,int x(5)相当于int x = 5

7-21 python里dir(对象)可以看类的属性和方法

7-23 atof: ascii -> float 字符转数字
* int转string小技巧
  int x(5);

  std::stringstream ss;

  std::string str;

  ss << x;    // x数据倒给ss

  ss >> str;  // ss数据倒给str
* 查看数据类型

  #include <typeinfo>

  std::cout << typeid(查看对象).name() << std::endl;










