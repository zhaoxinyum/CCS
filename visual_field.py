import numpy as np
from matplotlib import pyplot as plt


def Vis_Field(vector_field_np, filename):
    height, width = vector_field_np.shape[0:2]
    sample_step = 10
    Y, X = np.meshgrid(np.arange(height), np.arange(width))
    dpi = 100  # 假设 DPI = 100
    figsize = (width / dpi, height / dpi)

    plt.figure(figsize=figsize)
    plt.quiver(Y[::sample_step, ::sample_step], X[::sample_step, ::sample_step],
               vector_field_np[::sample_step, ::sample_step, 0], vector_field_np[::sample_step, ::sample_step, 1],
               angles='xy',
               scale_units='xy', scale=0.1, color='b')

    plt.gca().invert_yaxis()
    plt.axis('off')
    # 保存图片
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)

    # 关闭 figure，避免显示
    plt.close()
