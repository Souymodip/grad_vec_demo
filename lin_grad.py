import numpy as np
from scipy.ndimage import minimum_filter
from config import CHANNELS, ESP
import matplotlib.pyplot as plt
from helpers import draw_gradient


def theta_V1(Gs, debug):
    """
    Todo: Valid the energy functions. The maxima-minima points are getting mixed
    """
    assert len(Gs.shape) == 2 and Gs.shape[1] == 2
    Gx, Gy = Gs[:, 0], Gs[:, 1]
    a = np.sum(Gx * Gy)
    b = np.sum(Gx ** 2 - Gy ** 2)
    c = -np.sum(Gx * Gy)
    disc = b ** 2 - 4 * a * c
    # print("a:{:.3f}, b:{:.3f}, c:{:.3f}, discriminant:{:.3f}".format(a, b, c, disc))
    if np.abs(a) < ESP:
        gx = np.mean(np.abs(Gx))
        gy = np.mean(np.abs(Gy))
        if debug: print("\t Axis aligned: gx:{}, gy:{}".format(gx, gy))
        return np.arctan2(gy, gx)
    else:
        tt1 = np.arctan((-b + np.sqrt(disc)) / (2 * a))
        tt2 = np.arctan((-b - np.sqrt(disc)) / (2 * a))
        E1 = np.linalg.norm(- Gx * np.sin(tt1) + Gy * np.cos(tt1))
        E2 = np.linalg.norm(- Gx * np.sin(tt2) + Gy * np.cos(tt2))
        if debug: print(" tt1:{} -> {}, tt2:{} -> {}".format(np.rad2deg(tt1), E1, np.rad2deg(tt2), E2))
        return tt1 if E1 >= E2 else tt2


def theta(Gs, debug):
    thetas = np.arctan2(Gs[:, 1], Gs[:, 0])
    thetas = np.where(thetas < 0, thetas + np.pi, thetas)
    mean_t = np.mean(thetas)
    if debug: print(" Mean angle: {:.3f} deg".format(np.rad2deg(mean_t)))
    return mean_t


def flatten_gradient(G_, M_):
    assert M_.shape[:2] == G_.shape[:2] and G_.shape[2] == 2 and G_.shape[3] == CHANNELS
    G = G_ * M_[:, :, None, None]
    GxR, GxG, GxB = G[:, :, 0, 0], G[:, :, 0, 1], G[:, :, 0, 2]
    GyR, GyG, GyB = G[:, :, 1, 0], G[:, :, 1, 1], G[:, :, 1, 2]

    G = [np.stack([GxC.flatten(), GyC.flatten()]).T for GxC, GyC in zip([GxR, GxG, GxB], [GyR, GyG, GyB])]
    G = np.concatenate(G, axis=0)

    assert len(G.shape) == 2 and G.shape[1] == 2 and G.shape[0] == G_.shape[0] * G_.shape[1] * CHANNELS
    return G


def grad_direction(flat_G, lin_grad_qualification_ratio, debug):
    mods = np.linalg.norm(flat_G, axis=1)
    M = (mods > ESP).astype(int)
    if debug: print("-grad_direction. Grad:{} / Total:{} := {}  lin_grad_qualification_ratio:{}".format(
        np.sum(M), len(M), np.sum(M) / len(M), lin_grad_qualification_ratio))

    if np.sum(M) / len(M) <= lin_grad_qualification_ratio:
        return None

    t = theta(flat_G[mods > ESP], debug)
    return np.array([np.cos(t), np.sin(t)]) if t is not None else None


def get_samples(I, M, start, direction, sample_step = 1, debug=False):
    H, W = I.shape[:2]
    assert np.round(np.linalg.norm(direction)) == 1
    s = []
    ls = []
    L = np.sqrt(H**2 + W**2)*0.5

    def get(sign):
        for l in np.arange(0, L, sample_step):
            p = start + l * sample_step * sign * direction
            i, j = int(p[1]), int(p[0])
            if 0<= i < H and 0 <= j < W and M[i, j] > 0:
                s.append(I[i,j])
                ls.append(l * sample_step * sign)

    get(1), get(-1)
    s = np.array(s)

    if debug and len(s) > 0:
        print("-STD : {}".format(np.std(s, axis=0)))
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(I * M[:,:,None])
        ls = np.array(ls)
        points = start + ls[:, None] * direction
        axs[0].plot(points[:,0], points[:,1], 'o')
        axs[0].plot(start[0], start[1], 'ro')
        axs[1].plot(np.arange(0, len(s)), s, '-')
        plt.show()

    return s if len(s) > 3 else None


def lin_fit_possible(S, settings, debug):
    if len(S) <= settings['SIGNIFICANT_CONTACT']:
        if debug: print("-- return: insignificant sample size {} < {}".format(len(S), settings['SIGNIFICANT_CONTACT']))
        return False
    exceptable_deviation = settings['EXCEPTABLE_DEVIATION']
    deviations = [
        int(np.std(s[:,0]) < exceptable_deviation and
            np.std(s[:,1]) < exceptable_deviation and
            np.std(s[:,2]) < exceptable_deviation)
        for s, p in S
    ]

    if debug:
        for d, (s, p) in zip(deviations, S):
            print("* std:{}, {}, {} := {}".format(np.std(s[:,0]), np.std(s[:,1]), np.std(s[:,2]), d))
        print("-lin_fit_possible :{} / {} := {}".format(np.sum(deviations), len(deviations), np.sum(deviations) / len(deviations)))

    return np.sum(deviations) / len(deviations) > 0.9


def is_solid(I, M, solid_merge_threshold, debug):
    color_deviation = np.std(I[M > 0], axis=0)
    if debug: print ("-Solid color threshold :{}. Deviation:{}".format(solid_merge_threshold, color_deviation))
    return (color_deviation <= solid_merge_threshold).all()


def get_lin_samples(I, M, direction, sample_step, debug):
    ortho_direction = np.array([-direction[1], direction[0]])
    L = np.sqrt(I.shape[0] ** 2 + I.shape[1] ** 2) * 0.5
    mid = np.array([I.shape[1], I.shape[0]]) * 0.5
    def get(d):
        ss = []
        for l in np.arange(0, L, sample_step):
            start = mid + sample_step * d * l
            s = get_samples(I, M, start, ortho_direction, sample_step=sample_step, debug=debug)
            if s is not None:
                ss.append((s, start))
        return ss

    ss_right = get(direction)
    ss_left = get(-1 * direction)
    ss_left.reverse()
    ss = ss_left + ss_right
    return ss


def fit_grad(I, M, G, settings, debug):
    assert I.shape[:2] == M.shape[:2] == G.shape[:2] and I.shape[2] == CHANNELS \
           and G.shape[2] == 2 and G.shape[3] == CHANNELS
    if 'TRIM_BOUNDARY' in settings:
        M = minimum_filter(M, size=3)
    if is_solid(I, M, settings['SOLID_MERGE_THRESHOLD'], debug=debug):
        if debug: print("\t -- return Solid.")
        return True

    flat_G = flatten_gradient(G, M)

    direction = grad_direction(flat_G, settings['LIN_GRAD_QUALIFICATION_RATIO'], debug=debug)
    if debug:
        print("-direction : {} deg".format(np.rad2deg(np.arctan2(direction[1], direction[0]))))
    if direction is None:
        if debug: print("\t --return. Direction null.")
        return None

    sample_step = settings['SAMPLE_STEP'] if 'SAMPLE_STEP' in settings else 1
    ss = get_lin_samples(I, M, direction, sample_step=sample_step, debug=False)
    lin_fit = lin_fit_possible(ss, settings, debug=debug)

    if debug and len(ss) != 0:
        L = np.sqrt(I.shape[0] ** 2 + I.shape[1] ** 2) * 0.5
        mid = np.array([I.shape[1], I.shape[0]]) * 0.5
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(I * M[:, :, None])
        axs[0].plot([mid[0], mid[0] + 0.25*L*direction[0]],
                    [mid[1], mid[1] + 0.25*L*direction[1]], 'r-')

        if len(ss):
            avg = np.array([np.mean(s, axis=0) for s, p in ss])
            l = np.arange(0, len(avg))
            axs_ = axs[1]
            axs_.plot(l, avg[:,0], 'r-')
            axs_.plot(l, avg[:,1], 'g-')
            axs_.plot(l, avg[:,2], 'b-')

        draw_gradient(I * M[:, :, None], G, [axs[2], axs[2], axs[2]])
        fig.suptitle("Avg Colour along the gradient")
        plt.show()

    if not lin_fit: return None
    # todo: generate stops and color values
    return ss

