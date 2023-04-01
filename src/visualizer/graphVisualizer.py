import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

class GraphVisualizer(object):
    
    def __init__(self) -> None:
        pass
    
    def draw(self, h, color):
        z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

        plt.figure(figsize=(10,10))
        plt.xticks([])
        plt.yticks([])

        plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
        plt.show()