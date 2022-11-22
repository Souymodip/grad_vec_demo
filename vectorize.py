import numpy as np
import cv2
from collections import Counter
import matplotlib.pyplot as plt
from config import ESP, SQ_RT2
from lin_grad import grad_direction, flatten_gradient, get_lin_samples
from helpers import gradient, draw_gradient, avg_smoothen
import config as cg
from svgfy import get_linear_grad_fill, polyline_grad, create_svg
from stop_extraction import stop_extract, render


def draw_contours(R_idx, ow_, reg_idx):
    fig, ax = plt.subplots()
    ax.imshow(R_idx)
    ax.set_title("Outer Walk for R:{}".format(reg_idx))
    ax.plot(ow_[:, 0], ow_[:, 1], 'o-', markersize=3)
    ax.plot(ow_[0, 0], ow_[0, 1], 'ro', markersize=3)
    plt.show()


def get_range(flat_R, flat_I, flat_J, reg_idx):
    assert flat_R.shape == flat_I.shape == flat_J.shape
    flat_I, flat_J = flat_I[flat_R == reg_idx], flat_J[flat_R == reg_idx]
    return [np.min(flat_I), np.max(flat_I)+1], [np.min(flat_J), np.max(flat_J)+1]


def outer_walk(R, c, debug):
    ow = []
    def add_point(p):
        if len(ow) == 0:
            ow.append(np.round(p,1))
        else:
            if np.linalg.norm(ow[-1] - p) > ESP:
                ow.append(np.round(p,1))

    for idx, p in enumerate(c):
        q = c[(idx+1)%len(c)]
        v = q - p
        theta = np.arctan2(v[1], v[0]) + np.pi / 2
        normal = np.array([np.cos(theta), np.sin(theta)])
        if np.linalg.norm(v) > 1.1:
            add_point(p + normal*0.5 * SQ_RT2)
            add_point((p+q)*0.5)
            add_point(q + normal*0.5 * SQ_RT2)
        else:
            theta2 = theta + np.pi/4
            normal2 = np.array([np.cos(theta2), np.sin(theta2)])
            add_point(p + normal2*0.5*SQ_RT2)
            add_point((p + q) * 0.5 + normal*0.5)
            theta2 = theta - np.pi / 4
            normal2 = np.array([np.cos(theta2), np.sin(theta2)])
            add_point(q + normal2 * 0.5 * SQ_RT2)
        if debug:
            plt.imshow(R)
            plt.plot(c[:,0], c[:,1], 'ko-')
            plt.plot(p[0], p[1], 'ro')
            plt.plot(q[0], q[1], 'bo')
            ow_ = np.array(ow)
            plt.plot(ow_[:,0], ow_[:,1], 'go-', markersize=4)
            plt.show()
    return np.array(ow)


def outerwalks_contours(img, flat_R, small_regions, debug):
    reg_counter = Counter(flat_R)
    R = flat_R.reshape(*img.shape[:2])
    I, J = np.meshgrid(np.arange(0, img.shape[0]), np.arange(0, img.shape[1]), indexing='ij')
    flat_I, flat_J = I.flatten(), J.flatten()

    def get_outers(cs, R_idx, r_i, r_j, debug):
        ows = []
        pixels = []
        for c_idx, c in enumerate(cs):
            ow = outer_walk(R_idx, c, debug=False) + np.array([r_j[0], r_i[0]])
            ows.append(ow)
            pixels.append(c + np.array([r_j[0], r_i[0]]))
            if debug: draw_contours(R_idx, ow - np.array([r_j[0], r_i[0]]), reg_idx)
        return ows, pixels

    R2OW = dict()
    R2CS = dict()
    for reg_idx in reg_counter:
        if reg_counter[reg_idx] <= small_regions: continue
        r_i, r_j = get_range(flat_R, flat_I, flat_J, reg_idx)
        R_idx = (R[r_i[0]:r_i[1], r_j[0] : r_j[1]] == reg_idx).astype(np.uint8)*255
        cs, _ = cv2.findContours(R_idx, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cs = [c.squeeze() for c in cs if len(c.squeeze().shape) == 2 and c.squeeze().shape[1] == 2]
        R2OW[reg_idx], R2CS[reg_idx] = get_outers(cs, R_idx, r_i, r_j, debug=False)

    if debug:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_title('Contours')
        for reg_idx in R2OW:
            for ow in R2OW[reg_idx]:
                ax.plot(ow[:,0], -ow[:,1], '-')
        plt.show()

    return R2OW, R2CS


def fit_linear_grad(img: object, R: object, G: object, reg_idx: object, settings: object, debug: object, ows: object) -> object:
    M = (R == reg_idx).astype(int)
    direction = grad_direction(flatten_gradient(G, M), settings['LIN_GRAD_QUALIFICATION_RATIO'], debug=False)
    if direction is None:
        if debug: print("\t- fit_linear_grad: direction un-determined for reg_{}.".format(reg_idx))
        return []
    sample_mass = get_lin_samples(img, M, direction, sample_step=settings['SAMPLE_STEP'], debug=False)
    if len(sample_mass) <= settings['MIN_COLOR_SAMPLES']: return []

    avg_sample = np.array([np.mean(s, axis=0) for s, p in sample_mass])
    sample_pos = np.array([p for _, p in sample_mass])

    if debug:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img * M[:, :, None])
        for i, c in enumerate(cg.RGB):
            axs[1].plot(np.arange(0, len(avg_sample)), avg_sample[:, i], '-', color=c)
        fig.suptitle("Avg Color in direction")
        plt.show()
    if np.std(avg_sample) < 0.001 : return []
    stop_pos, stop_colors = stop_extract(avg_sample, sample_pos, debug=False)

    if debug:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img * M[:, :, None])
        axs[0].set_title("Orignal")
        r_img = render(*img.shape[:2], stop_pos=stop_pos, stop_color=stop_colors)
        axs[1].imshow(r_img * M[:, :, None])
        axs[1].set_title("Reconstructed")
        axs[1].plot(stop_pos[:,0], stop_pos[:, 1], 'ro-')
        fig.suptitle("Rendering Stops :{}".format(reg_idx))
        plt.show()

    return [stop_pos, stop_colors]


# def get_lin_grad_data(img, flat_R, reg_idxs):
#     settings = {
#         'SMALL_REGIONS': 25,
#         'SIGNIFICANT_CONTACT':7,
#         'MIN_COLOR_SAMPLES': 10,
#         'TRIM_BOUNDARY': True,
#         'LIN_GRAD_QUALIFICATION_RATIO': 0.03,
#         'SOLID_MERGE_THRESHOLD': 0.005,
#         'SAMPLE_STEP': 1,
#         'EXCEPTABLE_DEVIATION': 0.02 # 0.5
#     }
#
#     G = gradient(img)
#     R = flat_R.reshape(*img.shape[:2])
#     I, J = np.meshgrid(np.arange(0, img.shape[0]), np.arange(0, img.shape[1]), indexing='ij')
#     flat_I, flat_J = I.flatten(), J.flatten()
#
#     solid_fill = dict()
#     for reg_idx in reg_idxs:
#         r_i, r_j = get_range(flat_R, flat_I, flat_J, reg_idx)
#         rImg = img[r_i[0]:r_i[1] + 1, r_j[0]:r_j[1] + 1]
#         rR = R[r_i[0]:r_i[1] + 1, r_j[0]:r_j[1] + 1]
#         rG = G[r_i[0]:r_i[1] + 1, r_j[0]:r_j[1] + 1]
#         stops_colors = fit_linear_grad(rImg, rR, rG, reg_idx, settings, False, R2OW[reg_idx])
#         if len(stops_colors) == 0:

def main(img, flat_R, svg=False, debug=False):
    print("Vectorizing Patches")
    settings = {
        'SMALL_REGIONS': 25,
        'SIGNIFICANT_CONTACT':7,
        'MIN_COLOR_SAMPLES': 10,
        'TRIM_BOUNDARY': True,
        'LIN_GRAD_QUALIFICATION_RATIO': 0.03,
        'SOLID_MERGE_THRESHOLD': 0.005,
        'SAMPLE_STEP': 1,
        'EXCEPTABLE_DEVIATION': 0.02 # 0.5
    }
    G = gradient(img)
    R2OW, R2CS = outerwalks_contours(img, flat_R, small_regions=settings['SMALL_REGIONS'], debug=False)

    R = flat_R.reshape(*img.shape[:2])
    I, J = np.meshgrid(np.arange(0, img.shape[0]), np.arange(0, img.shape[1]), indexing='ij')
    flat_I, flat_J = I.flatten(), J.flatten()

    solid_fill = dict()
    lin_grad_fill = dict()

    for reg_idx in R2OW:
        r_i, r_j = get_range(flat_R, flat_I, flat_J, reg_idx)
        rImg = img[r_i[0]:r_i[1] + 1, r_j[0]:r_j[1] + 1]
        rR = R[r_i[0]:r_i[1] + 1, r_j[0]:r_j[1] + 1]
        rG = G[r_i[0]:r_i[1] + 1, r_j[0]:r_j[1] + 1]

        stops_colors = fit_linear_grad(rImg, rR, rG, reg_idx, settings, debug, R2OW[reg_idx])
        if len(stops_colors) > 0:
            for i, s in enumerate(stops_colors[0]):
                stops_colors[0][i] = np.array([s[0] + r_j[0], s[1] + r_i[0]])
            lin_grad_fill[reg_idx] = stops_colors
        elif len(stops_colors) == 0:
            solid_fill[reg_idx] = np.mean(rImg[rR == reg_idx], axis=0)


    # svgfy(*img.shape[:2], reg2ow=R2OW, reg2solid=solid_fill, reg2lin_grad=lin_grad_fill, reg2rad_grad=None, debug=debug)
    if svg:
        create_svg(*img.shape[:2], reg2ow=R2OW, reg2solid=solid_fill, reg2lin_grad=lin_grad_fill, reg2rad_grad=None, debug=debug)
    return lin_grad_fill, solid_fill


