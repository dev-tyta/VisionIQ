import gradio as gr
from faster_rcnn import img_detect, video_detection


inp_image = gr.Image(type="filepath")
out_image = gr.Image()
inp_video = gr.Video()
out_video = gr.PlayableVideo


def detection(choice):
    if choice == "Image":
        gr.update(input=[inp_image], output=out_image, function=img_detect)
    elif choice == "Video":
        gr.update(inputs=[inp_video], output=out_video, function=video_detection)
    # elif choice == "Web-Cam":
    #     inputs = [input1]
    #     output = outputs
    #     function = function1


with gr.Blocks() as demo:
    choices = gr.Dropdown(["Image", "Video", "Web-Cam"],
                          label="What type of Object would you like to detect?", onchange=detection)

    
    det = gr.Button("Detect")
    det.click(detect, inputs, outputs)

if __name__ == "__main__":
    demo.queue()
    demo.launch()
