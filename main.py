from grad_mean_shift import main as segment
from paper2pixel_wrapper import call as svgfy
import numpy as np
import gradio as gr
import os
import config as cg


def sepia(input_img):
    print(f"Image: {input_img.shape}")
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img


def flip_image(x):
    return np.flipud(x)


def work(x):
    return flip_image(x), sepia(x), np.fliplr(x)


def open():
    cmd = f'open {os.path.join(cg.SVG, "out.svg")}'
    os.system(cmd)


def vec(npImg):
    npImg = flip_image(npImg)
    msSeg, mergeSeg = segment(npImg, app_config, debug=False)
    svgPng = svgfy()
    return svgPng



if __name__ == '__main__':
    app_config = {
        'EDGIFY': False,
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