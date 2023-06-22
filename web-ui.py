import os
import time
import soundfile as sf
import numpy as np
import tempfile
from scipy.io import wavfile
from pytube import YouTube
from gradio import Interface, components as gr
from moviepy.editor import AudioFileClip
from inference import EnsembleDemucsMDXMusicSeparationModel, predict_with_model
import torch

def download_youtube_video_as_wav(youtube_url):
    output_dir = "downloads"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "temp.mp4")

    try:
        yt = YouTube(youtube_url)
        yt.streams.filter(only_audio=True).first().download(filename=output_file)
        print("Download completed successfully.")
    except Exception as e:
        print(f"An error occurred while downloading the video: {e}")
        return None

    # Convert mp4 audio to wav
    wav_file = os.path.join(output_dir, "mixture.wav")
    clip = AudioFileClip(output_file)
    clip.write_audiofile(wav_file)

    return wav_file


def check_file_readiness(filepath):
    num_same_size_checks = 0
    last_size = -1

    while num_same_size_checks < 5:
        current_size = os.path.getsize(filepath)

        if current_size == last_size:
            num_same_size_checks += 1
        else:
            num_same_size_checks = 0
            last_size = current_size

        time.sleep(1)

    # If the loop finished, it means the file size has not changed for 5 seconds
    # which indicates that the file is ready
    return True



def separate_music_file_wrapper(input_string, use_cpu, use_single_onnx, large_overlap, small_overlap, chunk_size, use_large_gpu):
    input_files = []

    if input_string.startswith("https://www.youtube.com") or input_string.startswith("https://youtu.be"):
        output_file = download_youtube_video_as_wav(input_string)
        if output_file is not None:
            input_files.append(output_file)
    elif os.path.isdir(input_string):
        input_directory = input_string
        input_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.wav')]
    else:
        raise ValueError("Invalid input! Please provide a valid YouTube link or a directory path.")

    options = {
        'input_audio': input_files,
        'output_folder': 'results',
        'cpu': use_cpu,
        'single_onnx': use_single_onnx,
        'overlap_large': large_overlap,
        'overlap_small': small_overlap,
        'chunk_size': chunk_size,
        'large_gpu': use_large_gpu,
    }

    predict_with_model(options)

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_files = {}
    for f in input_files:
        audio_file_name = os.path.splitext(os.path.basename(f))[0]
        output_files["vocals"] = os.path.join(options['output_folder'], audio_file_name + "_vocals.wav")
        output_files["instrumental"] = os.path.join(options['output_folder'], audio_file_name + "_instrum.wav")
        output_files["instrumental2"] = os.path.join(options['output_folder'], audio_file_name + "_instrum2.wav") # For the second instrumental output
        output_files["bass"] = os.path.join(options['output_folder'], audio_file_name + "_bass.wav")
        output_files["drums"] = os.path.join(options['output_folder'], audio_file_name + "_drums.wav")
        output_files["other"] = os.path.join(options['output_folder'], audio_file_name + "_other.wav")


    # Check the readiness of the files
    output_files_ready = []
    for k, v in output_files.items():
        if os.path.exists(v) and check_file_readiness(v):
            output_files_ready.append(v)
        else:
            empty_data = np.zeros((44100, 2)) # 2 channels, 1 second of silence at 44100Hz
            empty_file = tempfile.mktemp('.wav')
            wavfile.write(empty_file, 44100, empty_data.astype(np.int16))  # Cast to int16 as wavfile does not support float32
            output_files_ready.append(empty_file)

    return tuple(output_files_ready)

description = """
# ZFTurbo Web-UI
Web-UI by [Ma5onic](https://github.com/Ma5onic)
## Options:
- **Use CPU Only:** Select this if you have not enough GPU memory. It will be slower.
- **Use Single ONNX:** Select this to use a single ONNX model. It will decrease quality a little bit but can help with GPU memory usage.
- **Large Overlap:** The overlap for large chunks. Adjust as needed.
- **Small Overlap:** The overlap for small chunks. Adjust as needed.
- **Chunk Size:** The size of chunks to be processed at a time. Reduce this if facing memory issues.
- **Use Fast Large GPU Version:** Select this to use the old fast method that requires > 11 GB of GPU memory. It will work faster.
"""

iface = Interface(
    fn=separate_music_file_wrapper,
    inputs=[
        gr.Text(label="Input Directory or YouTube Link"),
        gr.Checkbox(label="Use CPU Only", value=False),
        gr.Checkbox(label="Use Single ONNX", value=False),
        gr.Number(label="Large Overlap", value=0.6),
        gr.Number(label="Small Overlap", value=0.5),
        gr.Number(label="Chunk Size", value=1000000),
        gr.Checkbox(label="Use Fast Large GPU Version", value=False)
    ],
    outputs=[
        gr.Audio(label="Vocals"),
        gr.Audio(label="Instrumental"),
        gr.Audio(label="Instrumental 2"),
        gr.Audio(label="Bass"),
        gr.Audio(label="Drums"),
        gr.Audio(label="Other"),
    ],
    description=description,
)

iface.queue().launch(debug=True, share=False)