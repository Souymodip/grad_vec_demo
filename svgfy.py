import numpy as np
import config as cg
import os
import matplotlib.pyplot as plt
import drawSvg as draw

def rgb2hex(rgb):
    assert len(rgb) == 3
    return 'rgb({}, {}, {})'.format(int(255*rgb[0]), int(255*rgb[1]), int(255*rgb[2]))


def get_solid_fill(polyline, color):
    pass


def polyline_grad(poly_line, grad_id):
    assert len(poly_line) > 3
    points = ' '.join(['{},{}'.format(x,y) for x,y in poly_line])
    return '<polyline points=\"{}\" style=\"fill:url(#grad{});stroke:none\"/>'.format(
        points, int(grad_id)
    )


def polyline_solid(poly_line, color):
    assert len(poly_line) > 3
    points = ' '.join(['{},{}'.format(x,y) for x,y in poly_line])
    return '<polyline points=\"{}\" style=\"fill:{};stroke:none\"/>'.format(
        points, rgb2hex(color)
    )


def get_linear_grad_fill(stop_pos, stop_colors, idx):
    assert len(stop_pos) == len(stop_colors) > 1 and stop_pos.shape[1] == 2 and stop_colors.shape[1] == cg.CHANNELS
    D = np.array([np.linalg.norm(ps - stop_pos[0]) for ps in stop_pos])
    D = D * 100 / D[-1]
    assert D[-1] > cg.ESP
    stops = [
        '\t<stop offset=\"{}%\" style="stop-color:{};stop-opacity:1"/>'.format(per, rgb2hex(rgb)) for per, rgb in zip(D, stop_colors)
    ]

    grad_open = '<linearGradient id=\"grad{}" x1=\"{:.2f}\" y1=\"{:.2f}\" x2=\"{:.2f}\" y2=\"{:.2f}\" gradientUnits="userSpaceOnUse">\n'.format(
        int(idx), stop_pos[0][0], stop_pos[0][1], stop_pos[-1][0], stop_pos[-1][1]
    )
    grad_close ='\n</linearGradient>'
    return grad_open + '\n'.join(stops) + grad_close


def get_radial_grad_fill(polyline, focus, outer_radius, transform, stops, colors):
    pass


def create_svg(H, W, reg2ow, reg2solid, reg2lin_grad, reg2rad_grad, debug):
    def add_polyline(p, ow):
        p.M(ow[0,0], ow[0,1])
        for o in ow[1:]:
            p.L(o[0], o[1])
        p.Z()

    def lin_grad(stops, colors):
        grad = draw.LinearGradient(stops[0][0], stops[0][1], stops[-1][0], stops[-1][1])
        D = np.array([np.linalg.norm(ps - stops[0]) for ps in stops])
        D = D / D[-1]
        for per, rgb in zip(D, colors):
            grad.addStop(per, color=rgb2hex(rgb))
        return grad

    path = os.path.join(cg.SVG, 'out.svg')
    d = draw.Drawing(H, W, origin=(0, 0))
    for reg_idx in reg2ow:
        ows = reg2ow[reg_idx]
        if reg_idx in reg2solid:
            p = draw.Path(fill=rgb2hex(reg2solid[reg_idx]))
            for ow in ows:
                assert len(ow) > 1
                add_polyline(p, ow)
            d.append(p)
        elif reg_idx in reg2lin_grad:
            stop_pos, stop_colors = reg2lin_grad[reg_idx]
            grad = lin_grad(stop_pos, stop_colors)
            p = draw.Path(fill=grad)
            for ow in ows:
                assert len(ow) > 1
                add_polyline(p, ow)
            d.append(p)
        else:
            assert 0

    d.setPixelScale(1)
    d.saveSvg(path)