import numpy as np
from scipy.ndimage import convolve1d
import config as cg
import matplotlib.pyplot as plt


def laplacian1d(series, size=21):
    left = -1 * np.ones(size)
    right = -1 * np.ones(size)
    filter = np.array([*left, 2 * size, *right])
    return convolve1d(series, filter)


def get_peaks(s):
    size = 10

    def lSample(i):
        return s[max(0, i-size):i]

    def rSample(i):
        return s[i+1: min(len(s), i+size)]

    candidates = []
    for i in range(len(s)):
        if i == 0:
            if s[0] > np.mean(rSample(i)):
                candidates.append(i)
        elif i == len(s)-1:
            if s[-1] > np.mean(lSample(i)):
                candidates.append(i)
        elif s[i] > np.mean(lSample(i)) and s[i] > np.mean(rSample(i)):
            candidates.append(i)

    return np.array(candidates)



def cluster(profile, s):
    cluster = []
    window = 1
    filtered_s = []
    for i in range(len(s)):
        if len(cluster) == 0:
            cluster.append(s[i])
        elif cluster[-1] + window >= s[i]:
            cluster.append(s[i])
        else:
            max_i = np.argmax(profile[cluster])
            filtered_s.append(cluster[max_i])
            cluster = [s[i]]
    if len(cluster) >0 :
        max_i = np.argmax(profile[cluster])
        filtered_s.append(cluster[max_i])
    return filtered_s


def co_linearity(profile, s, thresold):
    #FixMe: color distance ratio must also match space distance ratio
    f = [s[0]]

    def non_colinearity(i):
        prev = profile[f[-1]]
        curr = profile[s[i]]
        next = profile[s[i + 1]]
        t = (s[i] - f[-1]) / (s[i+1] - f[-1])
        return np.linalg.norm(curr - ((1-t)*prev + t*next))

    for i in range(1, len(s)-1):
        q = non_colinearity(i)
        if q > thresold: f.append(s[i])
    f.append(s[-1])
    return f


def stop_extract(avg_colors, positions, debug):
    bumps = [np.abs(laplacian1d(avg_colors[:, i])) for i in range(cg.CHANNELS)]
    peaks = [get_peaks(bumps[i]) for i in range(3)]

    if debug:
        fig, axs = plt.subplots(3, 3)
        x = np.arange(0, len(avg_colors))
        for i in range(3):
            axs[0, i].plot(x, avg_colors[:, i])
            axs[1, i].plot(x, bumps[i])
            peak = np.array(peaks[i]).astype(int)
            f_peak = cluster(bumps[i], peak)
            axs[1, i].plot(peak, bumps[i][peak], 'o', color='red')
            axs[1, i].plot(f_peak, bumps[i][f_peak], 'o', color='blue')

    peaks = np.sort(list(set(cluster(bumps[0], peaks[0]) + cluster(bumps[1], peaks[1]) + cluster(bumps[2], peaks[2]))))
    threshold = 0.005 * np.max(bumps)
    if len(peaks) == 0:
        assert len(avg_colors) > 4
        offset = max(2, int(len(avg_colors)/100))
        peaks =[offset, len(avg_colors) - offset]
    peaks = co_linearity(avg_colors, peaks, thresold=threshold)

    if debug:
        x = np.arange(0, len(avg_colors))
        for i in range(3):
            axs[2, i].plot(x, bumps[i])
            axs[2, i].plot(peaks, bumps[i][peaks], 'o', color='red')

        fig.suptitle("Co-Lin:{:.3f}, peaks:{}".format(threshold, peaks))
        plt.show()

    return positions[peaks], avg_colors[peaks]


def render(H, W, stop_pos, stop_color):
    assert len(stop_pos) == len(stop_color) > 1
    img = np.zeros((H, W, cg.CHANNELS))
    direction = stop_pos[-1] - stop_pos[0]
    m = np.linalg.norm(direction)
    assert m > cg.ESP
    direction = direction / m
    D = np.array([np.linalg.norm(p- stop_pos[0]) for p in stop_pos])
    for i in range(H):
        for j in range(W):
            xy = np.array([j, i])
            d = (xy - stop_pos[0]).dot(direction)
            if d <= D[0] : img[i, j] = stop_color[0]
            elif d >=D[-1]: img[i, j] = stop_color[-1]
            else:
                for k in range(0, len(D)-1):
                    if D[k] <= d < D[k+1]:
                        r = (d - D[k]) / (D[k+1] - D[k])
                        assert 0 <= r < 1
                        img[i, j] = (1 - r) * stop_color[k] + r * stop_color[k+1]
                        break
    return img
