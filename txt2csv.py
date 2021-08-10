# 把txt里的结果转成csv保存
import pandas as pd

def txt2csv(txt_name, output_file_name, index=['x','y','z','delay','plr'], delay=[50, 100], plr=[5, 20, 35, 50]):
    data = pd.read_csv(txt_name, sep='	', header=None, names=index)
    data['delay']=data['delay'].replace(delay[0],0)
    data['delay']=data['delay'].replace(delay[1],1)

    data['plr']=data['plr'].replace(plr[0],0)
    data['plr']=data['plr'].replace(plr[1],1)
    data['plr']=data['plr'].replace(plr[2],2)
    data['plr']=data['plr'].replace(plr[3],3)

    data['decode'] = data['plr'] + data['delay'] * 4      # 用来画聚类图的总标签
    data.to_csv(output_file_name, index=False)
