from matplotlib import pyplot as plt
import numpy as np


def matrix_heatmap(harvest: np.array, save_path='./heatmap_of_matrix.png'):
    plt.imshow(harvest)
    plt.tight_layout()
    plt.colorbar()
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
