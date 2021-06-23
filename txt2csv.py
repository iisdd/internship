# 把txt里的结果转成csv保存
import pandas as pd

filename = '20210612-5.txt'
x = pd.read_csv(filename, sep = '	', header=None, names=['x','y','z','delay','plr'])

x['delay']=x['delay'].replace(50,0)
x['delay']=x['delay'].replace(100,1)

x['plr']=x['plr'].replace(5,0)
x['plr']=x['plr'].replace(20,1)
x['plr']=x['plr'].replace(35,2)
x['plr']=x['plr'].replace(50,3)
# 最后一行标签不对
x[:1290].to_csv(r'y.csv', index=False)