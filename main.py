from pathlib import Path
from time import perf_counter

import gradio as gr

from pose2d import run_2d_pose
from pose3d import run_3d_pose


def get_3d_pose(video_name):
    start_2d = perf_counter()
    path2d = run_2d_pose(video_name)
    print(f'2D pose estimation took {perf_counter() - start_2d:.2f}s')

    start_3d = perf_counter()
    run_3d_pose(video_name)
    print(f'3D pose estimation took {perf_counter() - start_3d:.2f}s')
    print(f'Total time: {perf_counter() - start_2d:.2f}s')

    stem = Path(video_name).stem
    print(f'Saved 3d video to output/{stem}/{stem}_3D.mp4')

    return path2d, f'output/{stem}/{stem}_3D.mp4'


def create_ui():
    with gr.Blocks(
            css=".wrap svelte-1voqrms {max-height: 500px; box-sizing: border-box}"
                ".block.svelte-1voqrms {overflow: scroll}") as demo:
        with gr.Row(scale=0.8):
            with gr.Column():
                input_video = gr.Video(type="mp4", label="Upload a video", autosize=True)
                with gr.Row():
                    submit_btn = gr.Button("Submit").style(full_width=True).style(size='sm')
                    clear_btn = gr.Button("Clear").style(size='sm')
                # with gr.Row():
                #     text1 = gr.Textbox(label="")
            with gr.Column(scale=0.5):
                output_video2d = gr.Video(label="2D Pose Estimation", autosize=True)
                output_video3d = gr.Video(label="3D Pose Estimation", autosize=True)
        clear_btn.click(lambda: (None, None, None), inputs=[], outputs=[input_video, output_video2d, output_video3d])
        submit_btn.click(get_3d_pose, inputs=input_video, outputs=[output_video2d, output_video3d])

    return demo

if __name__ == '__main__':
    # get_3d_pose('./videos/kunkun_cut.mp4')
    create_ui().launch()
