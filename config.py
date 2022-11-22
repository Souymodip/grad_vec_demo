import os
from math import sqrt

PRJ = 'grad_vec_demo'
ROOT = os.getcwd()
ROOT = ROOT[:ROOT.find(PRJ) + len(PRJ)]


def get_folder(path):
    if not os.path.exists(path): os.mkdir(path)
    return path


OUT_DIR = get_folder(os.path.join(ROOT, 'outputs'))
INPUT_DIR = get_folder(os.path.join(ROOT, 'input'))
SVG = get_folder(os.path.join(ROOT, 'svg'))
CHANNELS = 3
RGB = ['red', 'blue', 'green']
ESP = 1e-8
SQ_RT2 = sqrt(2)
