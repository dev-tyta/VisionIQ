import gradio as gr
# from faster_rcnn import img_detect, video_detection


# test function
def test(inp):
    out = inp
    return out


with gr.Blocks() as demo:
    the_ = "# Human Detector and Counter" \
           "Using Faster-RCNN model, the app detects the people present in your image. " \
           "It returns the image with a bounding box over each person and the total number of people" \
           "## How to Use" \
           "1. Upload Image File" \
           "2. Click on Detect to get the output. NOTE: This would take a while." \
           "3. Download the processed Image."
    gr.Markdown(the_)
    # choices = gr.Dropdown(["Image", "Video"],
    #                       label="What type of Object would you like to detect?", onchange=detection)
    inp_img = gr.Image()
    out_img = gr.Image()

    det = gr.Button(value="Detect")
    det.click(fn=test, inputs=inp_img, outputs=out_img)

if __name__ == "__main__":
    demo.launch()
