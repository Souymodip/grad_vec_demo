from grad_mean_shift import main as segment
from paper2pixel_wrapper import call as svgfy
import numpy as np
import gradio as gr
import os
import config as cg
import cv2 as cv


def smoothen(input_img):
    return cv.bilateralFilter(input_img, 15, 75, 75)


def flip_image(x):
    return np.flipud(x)

def open():
    cmd = f'open {os.path.join(cg.OUT_DIR, "imageTrace", "input_image.svg")}'
    os.system(cmd)


def vec(npImg):
    print(f'Image:{npImg.shape}, {np.min(npImg)} ~ {np.max(npImg)}')
    if app_config['SMOOTHEN']: npImg = smoothen(npImg)
    # npImg = flip_image(npImg)
    msSeg, mergeSeg = segment(npImg, app_config, debug=False)
    svgPng = svgfy()
    return svgPng



if __name__ == '__main__':
    app_config = {
        'SMOOTHEN': False,
        'EDGIFY': True,
        'SMALL_REGION':0
    }
    with gr.Blocks(title='Grad2Vec') as demo:
        with gr.Row():
            image_input = gr.Image(label='Input Image', show_label=True)
            image_output = gr.Image(label='Vectorized Image', show_label=True)
        # with gr.Row():
        #     image_output2 = gr.Image()
        #     image_output3 = gr.Image()
        image_button1 = gr.Button("crunch!!")
        image_button2 = gr.Button("open in Ai")
        image_button1.click(vec, inputs=image_input, outputs=image_output)
        image_button2.click(open)
    demo.launch(share=False)