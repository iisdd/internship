# 核心函数: 

split.py(输入大图切成小图保存,输出切出的小图数量),

txt2csv.py(把包含label y的txt转成csv), 

gen_dataset.py(把图片和label都转成tensor,然后切成训练集和测试集), 

model.py(CNN模型), 

train.py(训练并画出中间层特征图), 

visualize.py(可视化模块,用TSN-E把CNN抽出来的特征降维成2维进行聚类展示), 

main.py(把以上功能集成到了主程序里,一键训练出图,只需要改文件地址).

utils.py(工具箱,包括模型保存&加载以及数据集生成)

## 整体框架图
![framework](https://github.com/iisdd/internship/blob/main/upload_pic/%E6%95%B4%E4%BD%93%E6%A1%86%E6%9E%B6.jpg)
1. 将需要存储的文本/视频/音频信息转化成01编码
2. 通过不同的功率以及偏振角的飞秒激光在玻璃片上打下体素
3. 双折射成像系统出来两类图,打点功率不同->延迟量不同->光点大小不同,打点偏振角不同->方位角不同->光点颜色不同
4. 通过神经网络分类光点图并转化为01编码
5. 加上纠错码的辅助还原成原本存储的信息

## 打点示意图
![save](https://github.com/iisdd/internship/blob/main/upload_pic/%E4%BA%94%E7%BB%B4%E5%AD%98%E5%82%A8.jpg)

## 网络结构
![NN](https://github.com/iisdd/internship/blob/main/upload_pic/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.jpg)

# 训练过程(3-bits)
## Loss曲线
![Loss](https://github.com/iisdd/internship/blob/main/upload_pic/Loss.jpg)

## 误差曲线
![error](https://github.com/iisdd/internship/blob/main/upload_pic/Error.jpg)


## 聚类结果图

### 1-bit(2类延迟量)
![1-bit](https://github.com/iisdd/internship/blob/main/upload_pic/1bit.jpg)
### 3-bits(2类延迟量4类偏振角)
![3-bits](https://github.com/iisdd/internship/blob/main/upload_pic/3-bits.jpg)
### 4-bits(2类延迟量8类偏振角)
![4-bits](https://github.com/iisdd/internship/blob/main/upload_pic/4-bits.jpg)
### 5-bits(2类延迟量16类偏振角)
![5-bits](https://github.com/iisdd/internship/blob/main/upload_pic/5-bits.jpg)


# 2021-06-10~2021-08-27
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

######################## 至此python部分图像分类告一段落,开始C++ ########################

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

  ss << x;&emsp;&emsp;&emsp;&emsp;//x数据倒给ss

  ss >> str;&emsp;&emsp;&emsp;//ss数据倒给str
* 查看数据类型

  #include \<typeinfo\>

  std::cout << typeid(查看对象).name() << std::endl;

7-26 台风,滞留学校

7-27 看DemoCam代码,打通拍照逻辑

7-28 看nnCam API手册,尝试接入MM

######################## 至此相机源码部分告一段落,开始部署模型 ########################

7-29 查找资料用C++调用pytorch模型

7-30 配置libtorch环境无限报错,改用cpu版本的libtorch

8-02 配置环境时有三个地方要改,同时运行时调试器要改成 debug x64
* 1.项目——属性——配置属性——C++——常规——附加包含目录(.h文件位置)
* 2.项目——属性——配置属性——链接器——常规——附加库目录(lib文件所在位置)
* 3.项目——属性——配置属性——链接器——常规——输入——附加依赖项(填所需lib)
* 路径要么用\\要么用/避免产生转义

8-03 教程造谣,我服了,配置属性 -> C/C++ -> 代码生成 -> 运行库 -> MDd(教程说选MTd我服了...),现在能成功加载模型了,但是给输入还是报错

8-04 下载opencv配置环境,把图片读成tensor送进模型推理
* C++的方法一般输入输出都要给进去: 方法(input, output), python里是 output = 方法(input)
* enum{a, b, c, d}相当于把{}里的数映射成了整数

8-05 把找轮廓、轮廓排序、图片切割功能移植过来

8-06 
* GLUE:NLP上的训练库
* semi-supervise-learning:有大量未标注的资料&少量标注的资料,未标注的用来pre-train,例如一句话遮住一个词来填词,标注资料用来fine-tune(具体的任务上用的训练集)
* python堆模块: heapq(小根堆,最小值在堆顶)
  * 方法:heapify(li):li -> 堆,
  * heappush(heap, x):把x压入堆
  * heappop(heap):弹出堆顶元素
* C++ vector添加元素(append):v.push_back()
* C++ vector的拼接(extend):v1.insert(v1.end(), v2.begin(), v2.end())

8-09 实现中心点排序&图片分割,现在想不保存小图,直接堆成tensor送入模型,但是堆叠tensor(cat)时碰到了报错,一开始的total不能是none,i=1的时候要把small_img赋给total

8-10 初步实现pytorch模型在C++中的部署,后序还有改进空间,这周先去交大写论文

######################## 至此部署模型初步完成,开始论文周 ########################

8-13 缩小延迟量差距(50/100 -> 50/70),3-bits依旧100%,传统方法只能做到95%

8-19 16分类偏振角,加上延迟量2分类,做4-bits、5-bits分类图,单纯的颜色可以做到100%,带上大小就只有98%了

8-24 训练32偏振角图片分类模型,准确率大概90%,用不了,导出中间层特征图片

8-26 交接代码
