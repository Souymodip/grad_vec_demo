import numpy as np
import os
import config as cg
from cairosvg import svg2png
from PIL import Image
import io


paper2pixel_exe = os.path.join(cg.ROOT, 'PaperToPixel_EXE')

def call():
    cmd = f'{paper2pixel_exe} \"{cg.INPUT_DIR}\" \"{cg.OUT_DIR}\" imageTrace'
    os.system(cmd)
    svg_path = os.path.join(cg.SVG, 'out.svg')
    img = svg2png(url=svg_path)
    pilImage = Image.open(io.BytesIO(img))
    return np.array(pilImage)
    # print(type(img))
    # pilImage.show()


if __name__ == '__main__':
    call()

