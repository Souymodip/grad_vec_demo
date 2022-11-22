import numpy as np
from scipy.ndimage import convolve1d
import config as cg
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P

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


def get_robust_peaks(color_series, kl, debug=False):
    from scipy import stats
    if debug: print(f'get_robust_peaks : {len(color_series)}')
    n = len(color_series)
    peak_pos = set()
    def diff(angle1, angle2):
        if angle1 < 0 : angle1 = angle1+np.pi
        if angle2 < 0 : angle2 = angle2+np.pi
        min_a,max_a = min(angle1, angle2), max(angle1, angle2)
        return min( max_a - min_a,  np.pi - (max_a - min_a))

    diffs3 = [] if debug else None
    offset = 5
    peak_pos.add(0+offset)
    peak_pos.add(n - 1-offset)
    for ch in range(cg.CHANNELS):
        if n >=  3*kl:
            y = color_series[:, ch]
            diffs = []
            for j in np.arange(kl, len(y)-kl):
                left_x = np.arange(j-kl, j+1)
                right_x = np.arange(j, j+kl+1)
                res_left = stats.linregress(left_x, y[left_x[0]:left_x[-1]+1])
                res_right = stats.linregress(right_x, y[right_x[0]:right_x[-1]+1])

                left_a, right_a = np.arctan(res_left.slope), np.arctan(res_right.slope)
                d = diff(left_a, right_a)
                if d >= 0.006: peak_pos.add(j)
                diffs.append(d)
            if debug: diffs3.append(diffs)

    if debug:
        fig, axs = plt.subplots(2, 3, tight_layout=True)
        for ch in range(3):
            axs[0, ch].plot(color_series[:,ch])
            if diffs3 is not None and len(diffs3[ch]) > 0:
                axs[1, ch].plot(diffs3[ch], 'r')
        fig.suptitle('Robust Peaks')
        plt.show()

    peaks = list(peak_pos)
    peaks.sort()
    return peaks


def cluster(peaks, window_size=10):
    window = []
    new_sites = []
    for pos in peaks:
        if len(window) == 0 or (pos - window[-1] < window_size and len(window) < window_size):
            window.append(pos)
        else:
            new_sites.append(int(np.median(window)))
            window = [pos]
    if len(window) > 0:
        new_sites.append(int(np.median(window)))
    return new_sites

def stop_extract(avg_colors, positions, debug):
    peaks = get_robust_peaks(color_series=avg_colors, kl=10, debug=False)
    peaks = cluster(peaks, window_size=10)
    # peaks = co_linearity(avg_colors, peaks, thresold=0.005)
    if debug:
        fig, axs = plt.subplots(3, tight_layout=True)
        x = np.arange(0, len(avg_colors))
        fig.suptitle(f'Peaks :{len(peaks)}')
        for i in range(3):
            axs[i].plot(x, avg_colors[:, i])
            axs[i].plot(x[peaks], avg_colors[peaks, i], 'ro')
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
