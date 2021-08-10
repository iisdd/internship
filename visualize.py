# 降维可视化
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
from matplotlib import cm
import matplotlib as mpl
from sklearn.manifold import TSNE
import model
import utils
import pandas as pd


def plot_with_labels(df, n_labels, title=''):
    cmap = mpl.colors.ListedColormap(["navy", "crimson", "limegreen", "gold", 'm', 'c', 'k', 'g', 'y', 'r'][:n_labels])
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, n_labels), n_labels)
    fig, ax = plt.subplots()

    scatter = ax.scatter(x='x', y='y', c='cluster', marker='.', data=df,
                             cmap=cmap, norm=norm, s=100, edgecolor='none', alpha=0.70)
    fig.colorbar(scatter, ticks=np.linspace(0, n_labels-1, n_labels))
    plt.title(title)
    plt.xticks([])      # 去掉坐标轴
    plt.yticks([])
    plt.savefig(title+'.jpg', dpi=1000)
    plt.show()


def reduce_dimension(cnn, input, label, plot_only):    # 把CNN最后一层输出拿出来降维
    # cnn为训练好的模型,input为堆叠好的图片tensor,label为标签的tensor,plot_only为绘制的点数量
    _, last_layer, _ = cnn(input)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, random_state=63)
    low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
    labels = label.numpy()[:plot_only]
    df = pd.DataFrame({"x": low_dim_embs[:, 0],
                       "y": low_dim_embs[:, 1],
                       "cluster": labels
                       })
    return df


def visualize(cnn, input, label, plot_only, n_labels, title=''):
    df = reduce_dimension(cnn, input, label, plot_only)
    plot_with_labels(df, n_labels, title)
