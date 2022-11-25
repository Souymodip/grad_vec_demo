import numpy as np
from collections import Counter
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt


def get_features(img, flat_R, W, reg_ids, scale_pos, bandwidth):
    assert len(flat_R) % W == 0 and len(flat_R.shape) == 1 and len(reg_ids) > 0

    def ij(pidx):
        return int(pidx / W), pidx % W

    reg_ids = set(reg_ids)
    H = int(len(flat_R) / W)
    features = []
    pixel_idx = []
    for pidx, reg_idx in enumerate(flat_R):
        if reg_idx in reg_ids:
            i, j = ij(pidx)
            features.append([i / H, j / W, *img[i, j]])
            pixel_idx.append(pidx)
    features = np.array(features) * np.array([scale_pos, scale_pos, 1, 1, 1])[None, :]
    return features, pixel_idx, bandwidth


def squeeze_reg_idx(flat_R):
    ordered_reg_ix = sorted(set(flat_R))
    for new_idx, old_idx in enumerate(ordered_reg_ix):
        flat_R[flat_R == old_idx] = new_idx
    # print("- squeezed :", Counter(flat_R))
    return flat_R


def cluster(img, flat_R, settings, debug):
    reg_count = Counter(flat_R)
    OneTwos = set([reg_id for reg_id in reg_count if 1 <= reg_count[reg_id] <= settings['LARGE_REGIONS']])
    features, pixel_idx, bandwidth = get_features(img, flat_R, img.shape[1], reg_ids=OneTwos,
                                                  scale_pos=settings['SCALE_POSITION_FEATURE'],
                                                  bandwidth=settings['BANDWIDTH'])
    print("- Mean shift with bandwidth:{} and {} features".format(bandwidth, features.shape))
    print("- small regions :{}".format(len(OneTwos)))
    clustering = MeanShift(bandwidth=bandwidth).fit(features)
    print("- Initial Segs :{}, Clustered segs: {}".format(len(OneTwos), len(clustering.cluster_centers_)))
    R_ = np.ones(flat_R.shape) * -1
    R_[pixel_idx] = clustering.labels_
    R_[R_ == -1] = len(clustering.cluster_centers_) + flat_R[R_ == -1]
    R_ = squeeze_reg_idx(R_)
    if debug:
        fig, ax = plt.subplots(1, 2)
        R = np.ones(flat_R.shape) * -1
        R[pixel_idx] = flat_R[pixel_idx]
        ax[0].set_title("Original")
        ax[0].imshow(img)
        ax_ = ax[1]
        ax_.set_title("Clustering")
        ax_.imshow(R_.reshape(img.shape[:2]))
        plt.show()
    return R_.flatten()


def neighbourhood(HW, pidx):
    H, W = HW
    i, j = int(pidx / W), pidx % W
    return [(k * W + l) for k, l in [
        [i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]  # , [i-1, j-1], [i+1, j-1], [i-1, j+1], [i+1, j+1]
    ] if 0 <= k < H and 0 <= l < W]


def anti_aliased(img, flat_R, settings):
    large = settings['LARGE_REGIONS']
    reg_count = Counter(flat_R)
    for pidx, reg_idx in enumerate(flat_R):
        if reg_count[reg_idx] == 1:
            large_nh = [(flat_R[nidx], reg_count[flat_R[nidx]])
                        for nidx in neighbourhood(img.shape[:2], pidx) if reg_count[flat_R[nidx]] >= large
                        ]
            if len(large_nh) != 0:
                max_reg_idx = max(large_nh, key=lambda x: x[1])[0]
                flat_R[pidx] = max_reg_idx
    return flat_R


def consolidate(flat_R, W):
    assert len(flat_R) % W == 0 and len(flat_R.shape) == 1
    reg_count = Counter(flat_R)
    print("One: {}".format(len([reg_id for reg_id in reg_count if reg_count[reg_id] == 1])))
    print("Two: {}".format(len([reg_id for reg_id in reg_count if reg_count[reg_id] == 2])))
    print("3 - 7: {}".format(len([reg_id for reg_id in reg_count if reg_count[reg_id] > 2 and reg_count[reg_id] <= 7])))
    print(
        "8 - 15: {}".format(len([reg_id for reg_id in reg_count if reg_count[reg_id] > 7 and reg_count[reg_id] <= 15])))
    print("> 15: {}".format(len([reg_id for reg_id in reg_count if reg_count[reg_id] > 15])))


def main(img, flat_R, debug):
    settings = {
        'SCALE_POSITION_FEATURE': 0.1,
        'BANDWIDTH': 0.4,
        'LARGE_REGIONS': 25
    }
    # flat_R = anti_aliased(img, flat_R, settings)
    return cluster(img, flat_R, settings, debug=debug)


# def mainV1(img, flat_R, app_settings, debug):

