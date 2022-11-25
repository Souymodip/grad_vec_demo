import numpy as np
import config as cg
from collections import Counter
from copy import deepcopy
import matplotlib.pyplot as plt
from progress.bar import ChargingBar
from scipy.stats import entropy
from skimage import color


def nbh(h, w, i, j, kernel_size)->[int]:
    s = []
    n = int(3 / 2)
    for k in range(i - n, i + n + 1):
        if 0 <= k < h:
            for l in range(j - n, j + n + 1):
                if 0 <= l < w:
                    s.append(k * w + l)
    return s


def entropy_kernel(img, flat_reg, pIxd, fill, kernel_size=3):
    h, w = img.shape[:2]
    pi, pj = int(pIxd/w), pIxd%w

    signals = []
    for nIdx in nbh(h, w, pi, pj, kernel_size=kernel_size):
        nRegIdx = flat_reg[nIdx]
        ni, nj = int(nIdx/w), nIdx%w
        signals.append(np.linalg.norm(img[ni, nj] - fill(nRegIdx, nIdx)))

    signals = np.array(signals) / np.sum(signals)
    return entropy(pk=signals)


def entropy_test(img, flat_reg, fill, kernel):
    h, w = img.shape[:2]
    d = 250
    pIdxs = [d*h + i for i in range(w)]
    E = [entropy_kernel(img, flat_reg, pIdx, fill, kernel) for pIdx in pIdxs]
    fig, (ax1, ax2) = plt.subplots(1,2, tight_layout=True)
    ax1.imshow(img)
    ax1.set_title('Original Image')
    line = np.array([[pIdx%w, int(pIdx/w)] for pIdx in pIdxs])
    ax1.plot(line[:,0], line[:,1])
    ax2.scatter(np.arange(len(E)), E)
    ax2.set_title(f'Entropy: Kernel size: {kernel}')
    plt.show()


def render(img, flat_reg, fill_function):
    rImg_flat = np.zeros((len(flat_reg), cg.CHANNELS))
    for pIdx, rIdx in enumerate(flat_reg):
        rImg_flat[pIdx] = fill_function(rIdx, pIdx)
    return rImg_flat.reshape(*img.shape)


def luvDiff(rgb1, rgb2):
    luv1 = color.rgb2luv(rgb1.reshape(1,1,cg.CHANNELS)).squeeze() #cv2.cvtColor(rgb1, cv2.COLOR_RGB2Luv)
    luv2 = color.rgb2luv(rgb2.reshape(1,1,cg.CHANNELS)).squeeze() #cv2.cvtColor(rgb2, cv2.COLOR_RGB2Luv)
    diff = np.linalg.norm((luv1-luv2)*np.array([1/100, 1/200, 1/200]))
    # print(f'Luv1:{luv1}, Luv2:{luv2} := {diff}')
    return diff


def rgbDiff(rgb1, rgb2):
    return np.linalg.norm(rgb1 - rgb2)


def inflate_large_regions(img, flat_reg, fill_function, nbh, app_setting):
    size_map = Counter(flat_reg)
    P = np.arange(len(flat_reg))
    small_region = app_setting['ALIASED_SMALL_REGION']
    aliased_color_diff_threshold = app_setting['ALIASED_INFLATE_COLOR_DIFF_THRESHOLD']
    flat_img = img.reshape(-1, cg.CHANNELS)

    def is_boundary(rIdx, pIdx):
        for nIdx in nbh(pIdx):
            if flat_reg[nIdx] != rIdx: return True
        return False

    def grow(pIdxs, rIdx):
        new_Br = []
        for pIdx in pIdxs:
            for nIdx in nbh(pIdx):
                nRegIdx = flat_reg[nIdx]
                if size_map[nRegIdx] <= small_region:
                    diff = luvDiff(fill_function(rIdx, nIdx), flat_img[nIdx]) #np.linalg.norm(fill_function(rIdx, nIdx) - flat_img[nIdx])
                    if diff <= aliased_color_diff_threshold:
                        flat_reg[nIdx] = rIdx
                        new_Br.append(nIdx)
        return new_Br
    max_count = app_setting['ALIASED_ITERATIONS_MAX']
    assert max_count > 0
    with ChargingBar('\t- Inflate', max=len(size_map)) as cb:
        for rIdx in size_map:
            if size_map[rIdx] > small_region:
                Pr = P[flat_reg==rIdx]
                Br = [pIdx for pIdx in Pr if is_boundary(rIdx, pIdx)]
                count = 0
                while len(Br) > 0 and count < max_count:
                    Br = grow(Br, rIdx)
                    count +=1
            cb.next()
    return flat_reg


def get_lin_grad_functions(W, lin_grad_fill):
    def get_color(rIdx, pIdx):
        stops, colors = lin_grad_fill[rIdx]
        D = np.array([np.linalg.norm(p - stops[0]) for p in stops])
        direction = stops[-1] - stops[0]
        m = np.linalg.norm(direction)
        if m <= cg.ESP:
            return colors[0]
        direction = direction / m

        xy = np.array([pIdx % W, int(pIdx / W)])
        d = (xy - stops[0]).dot(direction)
        if d <= D[0]:
            return colors[0]
        elif d >= D[-1]:
            return colors[-1]
        else:
            for k in range(0, len(D) - 1):
                if D[k] <= d < D[k + 1]:
                    r = (d - D[k]) / (D[k + 1] - D[k])
                    assert 0 <= r < 1
                    return (1 - r) * colors[k] + r * colors[k + 1]
            assert 0
            return None
    return get_color


def get_solid_function(solid_fill):
    def get_color(rIdx):
        return solid_fill[rIdx]
    return get_color


def de_anti_alias(flat_img, flat_reg, nbh, shape, app_setting, debug):
    new_flat_reg = deepcopy(flat_reg)
    new_flat_img = deepcopy(flat_img)
    new_flat_img_luv = color.rgb2luv(new_flat_img.reshape(1, *new_flat_img.shape)).squeeze()
    small_region = app_setting['DE_ANTI_ALIAS_SMALL_REGION']
    def fun(new_flat_img_luv, new_flat_reg, size_map):
        for pIdx, rIdx in enumerate(new_flat_reg):
            if size_map[rIdx] < small_region:
                diffs = [
                    (nIdx, np.linalg.norm(new_flat_img_luv[nIdx]-new_flat_img_luv[pIdx])) for nIdx in nbh(pIdx)
                    if rIdx != new_flat_reg[nIdx]
                ]
                if len(diffs) > 0:
                    min_diff = min(diffs, key=lambda x:x[1])
                    new_flat_img_luv[pIdx] = new_flat_img_luv[min_diff[0]]
                    new_flat_reg[pIdx] = new_flat_reg[min_diff[0]]
        return new_flat_img_luv, new_flat_reg

    size_map = Counter(new_flat_reg)
    iter_count = app_setting['DE_ANTI_ALIAS_ITERATIONS']
    with ChargingBar(f'\t- De-Anti-Aliasing:', max=iter_count) as cb:
        for _ in range(iter_count):
            new_flat_img_luv, new_flat_reg = fun(new_flat_img_luv, new_flat_reg, size_map)
            last_count = len(size_map)
            size_map = Counter(new_flat_reg)
            cb.suffix = f'Reg Count: {len(size_map)}'
            if last_count <= len(size_map): break
            cb.next()

    new_flat_img = color.luv2rgb(new_flat_img_luv.reshape(1, *new_flat_img_luv.shape)).squeeze()

    if debug:
        fig, ((ax1, ax2), (ax11, ax12)) = plt.subplots(2,2, tight_layout=True)
        ax1.imshow(flat_img.reshape(shape))
        ax2.imshow(new_flat_img.reshape(shape))
        ax11.imshow(flat_reg.reshape(shape[:2]))
        ax12.imshow(new_flat_reg.reshape(shape[:2]))
        fig.suptitle(f'De-Anti-Aliasing: Iterations: {iter_count}')
        plt.show()
    # exit(0)
    return new_flat_reg


def create_fill_function(flat_img, flat_reg, W, lin_grad_fill, solid_fill):
    lin_grad = get_lin_grad_functions(W, lin_grad_fill)
    solid = get_solid_function(solid_fill)

    def fill_function(rIdx, pIdx):
        if rIdx in lin_grad_fill: return lin_grad(rIdx, pIdx)
        elif rIdx in solid_fill: return solid(rIdx)
        else: return np.mean(flat_img[flat_reg==rIdx], axis=0)

    return fill_function


def squeeze_thin_regions(img, flat_reg_inflated, fill_function, app_setting):
    def func(img, flat_reg, fill_function, small_region, aliased_color_diff_threshold):
        h, w = img.shape[:2]
        flat_img = img.reshape(-1, cg.CHANNELS)
        size_map = Counter(flat_reg)

        def left_reg(pIdx):
            pi, pj = int(pIdx / w), pIdx % w
            return None if pj == 0 else flat_reg[pi * w + pj - 1]

        def right_reg(pIdx):
            pi, pj = int(pIdx / w), pIdx % w
            return None if pj == w - 1 else flat_reg[pi * w + pj + 1]

        def top_reg(pIdx):
            pi, pj = int(pIdx / w), pIdx % w
            return None if pi == 0 else flat_reg[(pi - 1) * w + pj]

        def bot_reg(pIdx):
            pi, pj = int(pIdx / w), pIdx % w
            return None if pi == h - 1 else flat_reg[(pi + 1) * w + pj]

        def squeeze_x(pIdx, rIdx):
            lReg, rReg = left_reg(pIdx), right_reg(pIdx)
            if lReg != rIdx and rReg != rIdx:
                if (lReg is None or size_map[lReg] <= small_region) and (
                        rReg is None or size_map[rReg] <= small_region):
                    pass
                elif (lReg is None or size_map[lReg] <= small_region) and rReg is not None and size_map[
                    rReg] > small_region:
                    diff = np.linalg.norm(flat_img[pIdx] - fill_function(rReg, pIdx))
                    if diff <= aliased_color_diff_threshold:
                        flat_reg[pIdx] = rReg
                elif lReg is not None and size_map[lReg] > small_region and (
                        rReg is None or size_map[rReg] <= small_region):
                    diff = np.linalg.norm(flat_reg[pIdx] - fill_function(lReg, pIdx))
                    if diff <= aliased_color_diff_threshold:
                        flat_reg[pIdx] = lReg
                else:
                    diff_right = np.linalg.norm(flat_img[pIdx] - fill_function(rReg, pIdx))
                    diff_left = np.linalg.norm(flat_reg[pIdx] - fill_function(lReg, pIdx))
                    if diff_left <= aliased_color_diff_threshold and diff_left <= diff_right:
                        flat_reg[pIdx] = lReg
                    elif diff_right <= aliased_color_diff_threshold and diff_right <= diff_left:
                        flat_reg[pIdx] = rReg

        def squeeze_y(pIdx, rIdx):
            tReg, bReg = top_reg(pIdx), bot_reg(pIdx)
            if tReg != rIdx and bReg != rIdx:
                if (tReg is None or size_map[tReg] <= small_region) and (
                        bReg is None or size_map[bReg] <= small_region):
                    pass
                elif (tReg is None or size_map[tReg] <= small_region) and bReg is not None and size_map[
                    bReg] > small_region:
                    diff = np.linalg.norm(flat_img[pIdx] - fill_function(bReg, pIdx))
                    if diff <= aliased_color_diff_threshold:
                        flat_reg[pIdx] = bReg
                elif tReg is not None and size_map[tReg] > small_region and (
                        bReg is None or size_map[bReg] <= small_region):
                    diff = np.linalg.norm(flat_reg[pIdx] - fill_function(tReg, pIdx))
                    if diff <= aliased_color_diff_threshold:
                        flat_reg[pIdx] = tReg
                else:
                    diff_bot = np.linalg.norm(flat_img[pIdx] - fill_function(bReg, pIdx))
                    diff_top = np.linalg.norm(flat_reg[pIdx] - fill_function(tReg, pIdx))
                    if diff_top <= aliased_color_diff_threshold and diff_top <= diff_bot:
                        flat_reg[pIdx] = tReg
                    elif diff_bot <= aliased_color_diff_threshold and diff_bot <= diff_top:
                        flat_reg[pIdx] = bReg

        #
        with ChargingBar(f"\t- Squeeze: Th:{aliased_color_diff_threshold}", max=len(flat_reg)) as cb:
            for pIdx, rIdx in enumerate(flat_reg):
                squeeze_x(pIdx, rIdx)
                squeeze_y(pIdx, rIdx)
                cb.next()

        return flat_reg

    small_region = app_setting['ALIASED_SMALL_REGION']
    aliased_color_diff_threshold = app_setting['ALIASED_SQUEEZE_COLOR_DIFF_THRESHOLD']
    flat_reg_squeezed = func(img, flat_reg_inflated, fill_function, small_region, aliased_color_diff_threshold)
    flat_reg_squeezed = func(img, flat_reg_squeezed, fill_function, small_region, aliased_color_diff_threshold * 2)
    return flat_reg_squeezed


def remap(flat_img, flat_reg, lin_grad_fill, solid_fill, pad):
    rIdxs = set(flat_reg)
    print(f'Remap : {len(rIdxs)}')
    lin, solid = dict(), dict()
    new_flat_reg = deepcopy(flat_reg)
    for new_rIdx, old_rIdx in enumerate(rIdxs):
        nId = new_rIdx + pad
        if old_rIdx in lin_grad_fill:
            stops_colors = lin_grad_fill[old_rIdx]
            first, last = stops_colors[0][0], stops_colors[0][-1]
            if np.linalg.norm(last - first) <= cg.ESP:
                solid[nId] = stops_colors[1][0]
            else:
                lin[nId] = stops_colors
        elif old_rIdx in solid_fill:
            solid[nId] = solid_fill[old_rIdx]
        else:
            solid[nId] = np.mean(flat_img[flat_reg == old_rIdx], axis=0)
        new_flat_reg[flat_reg == old_rIdx] = nId
    return new_flat_reg, lin, solid


def main(img, flat_reg, lin_grad_fill, solid_fill, app_setting, debug):
    print(f'Handling small regions created by anti-aliasing.')
    if debug: flat_reg_old = deepcopy(flat_reg)
    flat_img = img.reshape(-1, cg.CHANNELS)
    fill_function = create_fill_function(flat_img, flat_reg, img.shape[1], lin_grad_fill, solid_fill)
    h, w = img.shape[:2]

    def nbh(pidx):
        i, j = int(pidx / w), pidx % w
        return [(k * w + l) for k, l in [
            [i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1] #, [i - 1, j - 1], [i + 1, j - 1], [i - 1, j + 1], [i + 1, j + 1]
        ] if 0 <= k < h and 0 <= l < w]

    if app_setting['DE_ANTI_ALIAS']:
        flat_reg = de_anti_alias(flat_img, flat_reg, nbh, img.shape, app_setting, debug=debug)

    flat_reg_inflated = inflate_large_regions(img, flat_reg, fill_function, nbh, app_setting)
    if debug:
        print(f'De-Anit-Aliasing : {len(Counter(flat_reg_old))} ~ {len(Counter(flat_reg_inflated))}')
        fig, ((ax1, ax2), (ax11, ax12)) = plt.subplots(2,2, tight_layout=True)
        ax1.imshow(flat_reg_old.reshape(*img.shape[:2]))
        ax2.imshow(flat_reg_inflated.reshape(*img.shape[:2]))
        fig.suptitle('Inflate')

        rImg = render(img, flat_reg_inflated, fill_function)
        ax11.imshow(img)
        ax12.imshow(rImg)
        plt.show()

    if app_setting['ALIASED_SQUEEZE_THIN_REGION']:
        new_flat_reg = squeeze_thin_regions(img, flat_reg_inflated, fill_function, app_setting)
    else:
        new_flat_reg = flat_reg_inflated

    new_flat_reg, new_lin_fill, new_solid_fill = remap(flat_img, new_flat_reg, lin_grad_fill, solid_fill, app_setting['PAD_REG_INDEX'])

    if debug:
        new_fill_function = create_fill_function(flat_img, new_flat_reg, img.shape[1], new_lin_fill, new_solid_fill)
        fig, ((ax1, ax2), (ax11, ax12)) = plt.subplots(2,2, tight_layout=True)
        ax1.imshow(flat_reg_old.reshape(*img.shape[:2]))
        ax2.imshow(new_flat_reg.reshape(*img.shape[:2]))
        fig.suptitle(f'Region Count: {len(set(new_flat_reg))}. '
              f'Linear Fill: {len(new_lin_fill)}, Solid Fill: {len(new_solid_fill)}')

        rImg = render(img, new_flat_reg, new_fill_function)
        ax11.imshow(img)
        ax12.imshow(rImg)
        plt.show()

    return new_flat_reg, new_lin_fill, new_solid_fill
    # entropy_test(img, flat_reg_inflated, fill_function, kernel=5)
    # exit(0)



