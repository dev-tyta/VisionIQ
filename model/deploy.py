import gradio as gr
# from faster_rcnn import img_detect, video_detection


with gr.Blocks() as demo:
    the_ = "# Human Detector and Counter" \
           "Using Faster-RCNN model, the app detects the people present in your image. " \
           "It returns the image with a bounding box over each person and the total number of people" \
           "## How to Use" \
           "1. "
    gr.Markdown(the_)
    # choices = gr.Dropdown(["Image", "Video"],
    #                       label="What type of Object would you like to detect?", onchange=detection)


if __name__ == "__main__":
    demo.queue()
    demo.launch()
