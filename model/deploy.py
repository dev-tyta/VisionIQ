import gradio as gr
from faster_rcnn import img_detect, video_detection


with gr.Blocks() as demo:
    gr.Markdown("# Human Detector and Counter"
                "This app helps you detect humans present in your image."
                "Upload your choice of Image and the app loads the  ")
    # choices = gr.Dropdown(["Image", "Video"],
    #                       label="What type of Object would you like to detect?", onchange=detection)


if __name__ == "__main__":
    demo.queue()
    demo.launch()
