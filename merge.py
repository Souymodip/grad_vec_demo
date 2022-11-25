import networkx as nx
import numpy as np
from collections import Counter
from config import CHANNELS
from lin_grad import fit_grad
from helpers import gradient, draw_gradient
import matplotlib.pyplot as plt
from progress.bar import ChargingBar

def neighbourhood(HW, pidx):
    H, W = HW
    i, j = int(pidx/W), pidx % W
    return [ (k*W + l) for k, l in [
        [i-1, j], [i+1, j], [i, j-1], [i, j+1]
    ] if 0 <= k < H and 0 <= l < W]


def boundary(R_flat, HW, reg_id):
    assert len(R_flat.shape) == 1
    pixels = np.arange(len(R_flat))[R_flat == reg_id]
    if len(pixels) == 0: return None
    ns = []
    for pidx in pixels:
        ns.extend(neighbourhood(HW, pidx))
    return np.array(ns, dtype='int')


def merge(R, candidate_regions, is_mergable):
    # g = nx.Graph()
    cc_dict = np.arange(0, np.max(R)+1)

    candidate_regs = candidate_regions()
    if len(candidate_regs) > 0:
        with ChargingBar("\t- Merge: ", max=len(candidate_regs)) as cb:
            for reg_ids in candidate_regs:
                reg_heads = set([cc_dict[reg] for reg in reg_ids])
                closure = [reg_idx for reg_idx, head in enumerate(cc_dict) if head in reg_heads]
                if is_mergable(closure) is not None: cc_dict[closure] = min(closure)
                cb.next()

    head_map = dict()
    for reg_idx in np.arange(0, np.max(R)+1):
        head = cc_dict[reg_idx]
        if head not in head_map: head_map[head] = []
        head_map[head].append(reg_idx)

    R_ = np.ones(R.shape, dtype='int') * -1
    for new_idx, head in enumerate(head_map):
        for reg_idx in head_map[head]:
            R_[R == reg_idx] = new_idx
            assert len(head_map[head]) != 1 or head_map[head][0] == reg_idx

    return R_


def get_candidate_regions(R, settings):
    assert len(R.shape) == 2
    H, W = R.shape[:2]
    R_flat = R.flatten()
    RegionSizes = Counter(R_flat)

    def large_adj_region_pairs():
        pairs = set()
        with ChargingBar("\t- Get Candidate for Merge: ", max=np.max(R) + 1) as cb:
            for reg_id in range(np.max(R) + 1):
                if reg_id not in RegionSizes or RegionSizes[reg_id] >= settings['SMALL_REGION']:
                    boundary_pidxs = boundary(R_flat, (H,W), reg_id)
                    CR = Counter(R_flat[boundary_pidxs])
                    for neighbour_id in CR:
                        if neighbour_id != reg_id and CR[neighbour_id] > settings['SIGNIFICANT_CONTACT'] \
                                and RegionSizes[neighbour_id] > settings['SMALL_REGION']:
                            pairs.add((reg_id, neighbour_id) if reg_id < neighbour_id else (neighbour_id, reg_id))
                cb.next()

        return pairs

    return large_adj_region_pairs


def get_mergability(R, I, G, settings):
    assert I.shape[:2] == R.shape[:2] == G.shape[:2] and I.shape[2] == CHANNELS \
           and G.shape[2] == 2 and G.shape[3] == CHANNELS
    H, W = R.shape[:2]
    X, Y = np.einsum('j, ij -> ij', np.arange(0, W), np.ones((H, W), dtype='int')), \
                         np.einsum('i, ij -> ij', np.arange(0, H), np.ones((H, W), dtype='int'))

    def is_lin_mergable(r_idxs):
        debug = False #1031 in r_idxs

        r_idxs = list(r_idxs)
        if len(r_idxs) == 0: return None
        M = (R == r_idxs[0]).astype(int)
        for r_idx in r_idxs[1:]:
            M += (R == r_idx).astype(int)
        M = np.clip(M, a_min=0, a_max=1)
        X_, Y_ = X[M > 0], Y[M > 0]
        range_i = np.min(Y_), np.max(Y_)
        range_j = np.min(X_), np.max(X_)
        if debug:
            print("\n-Regions :{}".format(r_idxs))
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(I*M[:, :, None])
            RM = R * M
            for i, r in enumerate(r_idxs):
                RM[R==r] = i+1
            axs[1].imshow(RM)
            fig.suptitle("Merge ? {}".format(r_idxs))
            plt.show()

        return fit_grad(I=I[range_i[0]:range_i[1]+1, range_j[0]:range_j[1]+1],
                        M=M[range_i[0]:range_i[1]+1, range_j[0]:range_j[1]+1],
                        G=G[range_i[0]:range_i[1]+1, range_j[0]:range_j[1]+1],
                        settings=settings, debug=debug)
    return is_lin_mergable


def lin_merge(img, flat_segment, app_settings, debug=True):
    print("Linear Merge of regions")
    settings = {
        'SMALL_REGION': app_settings['MERGE_SMALL_REGION_THRESHOLD'],
        'SIGNIFICANT_CONTACT':7,
        'TRIM_BOUNDARY': True,
        'LIN_GRAD_QUALIFICATION_RATIO': 0.03,
        'SOLID_MERGE_THRESHOLD': 0.005,
        'SAMPLE_STEP': 1,
        'EXCEPTABLE_DEVIATION': 0.02 # 0.5
    }

    R = flat_segment.reshape(*img.shape[:2])
    G = gradient(img)

    if debug and False: draw_gradient(img, G, axs=None)
    R_ = merge(R, candidate_regions=get_candidate_regions(R, settings=settings),
              is_mergable=get_mergability(R, img, G, settings=settings))
    if debug:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(R)
        axs[1].imshow(R_)
        axs[0].set_title("Input :{}".format(np.max(R)+1))
        axs[1].set_title("Merged: {}".format(np.max(R_)+1))
        fig.suptitle("Merge")
        plt.show()
    return R_.flatten()


if __name__ == '__main__':
    pass

