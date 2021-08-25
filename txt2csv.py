# 把txt里的结果转成csv保存
import pandas as pd

def txt2csv(txt_name, output_file_name, index=['x','y','z','delay','plr'], delay=[50, 100], plr=[5, 20, 35, 50]):
    data = pd.read_csv(txt_name, sep='	', header=None, names=index)
    # 替换延迟量
    for d in range(len(delay)):
        data['delay'] = data['delay'].replace(delay[d], d)
    # 替换偏振角
    for p in range(len(plr)):
        data['plr'] = data['plr'].replace(plr[p], p)


    data['decode'] = data['plr'] + data['delay'] * len(plr)      # 用来画聚类图的总标签
    print(data.info())
    data.to_csv(output_file_name, index=False)
    print(data.head())
