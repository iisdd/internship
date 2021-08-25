# 集成各模块的主程序,原图的名字取名要用这种格式: 1_delay.tiff, 1_plr.tiff(从1开始)
from split import split
from txt2csv import txt2csv
from gen_dataset import gen_dataset
from train import train
from visualize import visualize

################# 切割图片超参数 ###################
PIC_PATH = '210616_pic'             # 原图路径(文件夹)
PIC_NUM = 10                        # 原图数量
SAVE_PATH = 'pic'                   # 切出来的小图保存路径
SUFFIX = 'tiff'                     # 图片后缀,ex:'tif', 'jpg', 'png'
PX = 10                             # 切出来的小图的边宽,ex:PX=10切出来就是20X20
################# 切割图片超参数 ###################
################# txt2csv超参数 ###################
TXT_NAME = '20210612-5.txt'         # txt文件名
OUTPUT_FILE_NAME = 'y.csv'          # 输出csv名
INDEX = ['x','y','z','delay','plr'] # txt列顺序,如果是先偏振再延迟量就掉换位置
DELAY = [50, 100]                   # 延迟量可选档位(从小到大)
PLR = [5, 20, 35, 50]               # 偏振角可选角度(从小到大)
################# txt2csv超参数 ###################
################ 生成训练集超参数 ##################
TYPE = 'plr'                        # 要训练的模型类型
TEST_SIZE = 0.2                     # 测试集占总数据的比例
################ 生成训练集超参数 ##################
################### 训练超参数 ####################
if TYPE == 'delay':
    N_O = len(DELAY)
elif TYPE == 'plr':
    N_O = len(PLR)
elif TYPE == 'both':
    N_O = len(PLR)*len(DELAY)
    
LR = 0.0005
BATCH_SIZE = 16
EPOCH = 3
################### 训练超参数 ####################
################### 画图超参数 ####################
PLOT_ONLY = 300                     # 画多少个点在聚类图上
################### 画图超参数 ####################

print('====================== 切割图片中 ===========================')
n_split = split(pic_path=PIC_PATH, pic_num=PIC_NUM, save_path=SAVE_PATH, suffix=SUFFIX, px=PX)
print('\n====================== 生成csv中 ============================')
txt2csv(txt_name=TXT_NAME, output_file_name=OUTPUT_FILE_NAME, index=INDEX, delay=DELAY, plr=PLR)
print('\n===================== 生成训练集中 ===========================')
pic_tensor, labels, decode, (train_x, test_x, train_y, test_y) = gen_dataset(pic_path=SAVE_PATH, pic_num=n_split, filename=OUTPUT_FILE_NAME, type=TYPE, test_size=TEST_SIZE)
print('\n===================== 模型训练中 ============================')
cnn = train(train_x, test_x, train_y, test_y, pic_px=PX*2, n_o=N_O, lr=LR, batch_size=BATCH_SIZE, epoch=EPOCH, type=TYPE)
print('\n===================== 可视化聚类中 ===========================')
visualize(cnn, input=pic_tensor, label=decode, plot_only=PLOT_ONLY, n_labels=N_O, title='3-bits')


