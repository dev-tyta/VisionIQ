import gradio as gr
from faster_rcnn import img_detect, video_detection


with gr.Blocks() as demo:
    # choices = gr.Dropdown(["Image", "Video"],
    #                       label="What type of Object would you like to detect?", onchange=detection)


if __name__ == "__main__":
    demo.queue()
    demo.launch()
