import gradio as gr

def cls_demo(img):
    # TODO: run classification + Grad-CAM + SAM overlay
    return "blast (demo)", 0.93, img

def seg_demo(img):
    # TODO: run segmentation + compute %area
    return img, "12.3%"

with gr.Blocks(title="Rice Leaf Health") as demo:
    gr.Markdown("# Rice Leaf Health Demo")
    with gr.Tab("Classification"):
        inp = gr.Image(type="pil", label="Upload leaf image")
        pred = gr.Label(num_top_classes=1, label="Prediction (demo)")
        score = gr.Number(label="Confidence")
        vis = gr.Image(label="Explanation Overlay")
        btn = gr.Button("Run")
        btn.click(fn=cls_demo, inputs=inp, outputs=[pred, score, vis])

    with gr.Tab("Segmentation"):
        inp2 = gr.Image(type="pil", label="Upload leaf image")
        mask = gr.Image(label="Segmentation (demo)")
        area = gr.Textbox(label="% Diseased Area")
        btn2 = gr.Button("Run")
        btn2.click(fn=seg_demo, inputs=inp2, outputs=[mask, area])

if __name__ == "__main__":
    demo.launch()
