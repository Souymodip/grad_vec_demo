import numpy as np
import matplotlib.pyplot as plt
import config as cg
from scipy.ndimage import gaussian_filter1d, convolve1d


def gradient(img):
    GrY, GrX = np.gradient(img[:,:,0])
    GgY, GgX = np.gradient(img[:,:,1])
    GbY, GbX = np.gradient(img[:,:,2])

    Gr = np.stack([GrX, GrY], axis=2)
    Gg = np.stack([GgX, GgY], axis=2)
    Gb = np.stack([GbX, GbY], axis=2)

    return np.stack([Gr, Gg, Gb], axis=3)


def draw_gradient(img, Gs, axs):
    color = ['red', 'blue', 'green']
    show = False
    if axs is None or len(axs) != cg.CHANNELS:
        fig, axs = plt.subplots(1,3)
        show = True
    for c in range(cg.CHANNELS):
        axs[c].set_title(color[c])
        axs[c].imshow(img)
    H, W = Gs.shape[:2]
    S = 5
    L=2
    for i in range(H):
        for j in range(W):
            if i % S != 0 or j % S != 0: continue
            for c in range(cg.CHANNELS):
                g = np.array([Gs[i,j,0,c], Gs[i,j,1,c]])
                m = np.linalg.norm(g)
                if m > cg.ESP:
                    g = g /m
                    axs[c].plot([j, j + L*g[0]], [i, i + L*g[1]], '-', color=color[c])

    if show:
        fig.suptitle("Gradients")
        plt.show()


def avg_smoothen(s, filter=3):
    w = np.array([*np.arange(1, int(filter/2)+1), 0, *np.arange(int(filter/2), 0, -1)])
    w = w / np.sum(w)
    # return gaussian_filter1d(s, sigma=0.5)
    return convolve1d(gaussian_filter1d(s, sigma=0.6), w)


def poly_fit(colors, pos, n, sep=0.5):
    assert len(colors) == len(pos) > 1
    d = np.linalg.norm(pos[0] - pos[-1])
    n = min(n, int(d/0.5))
    if n < 1: return np.mean(colors, axis=0), np.mean(pos, axis=0)

    def box(x0, l):
        def h(x):
            return (np.sign(x) + 1) / 2
        def f(x):
            return h(x-x0) * h(x0 + l - x) *  x
        return f

    def fun(x):
        ls = x[:n]
        cs = x[n:].reshape(n, cg.CHANNELS)

        cx = []
        D = [
            np.sum(np.linalg.norm(colors[i] - cx[i], axis=0)) for i in range(n)
        ]