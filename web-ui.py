import os
from gradio import Interface, components as gr
import soundfile as sf
import numpy as np
import tempfile
from inference import EnsembleDemucsMDXMusicSeparationModel, predict_with_model

options = {
    'cpu': False,
    'single_onnx': False,
    'overlap_large': 0.6,
    'overlap_small': 0.5,
    'chunk_size': 1000000,  # adjust as needed
}

# Create an instance of EnsembleDemucsMDXMusicSeparationModel
def separate_music_file_wrapper(input_directory, use_cpu, use_single_onnx, large_overlap, small_overlap, chunk_size):
    # Get all .wav files in the directory
    input_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.wav')]

    options = {
        'input_audio': input_files,
        'output_folder': './results/',  # change this as necessary
        'cpu': use_cpu,
        'single_onnx': use_single_onnx,
        'overlap_large': large_overlap,
        'overlap_small': small_overlap,
        'chunk_size': chunk_size,
        'large_gpu': not use_cpu,  # change this as necessary
    }

    # Separate the sources
    predict_with_model(options)
    
    # Assuming the separated files are saved in the same directory with the source name appended,
    # we can return a list of the output file paths
    output_files = {}
    for f in input_files:
        audio_file_name = os.path.splitext(os.path.basename(f))[0]
        output_files["vocals"] = os.path.abspath(options['output_folder'] + audio_file_name + "_vocals.wav")
        output_files["instrumental"] = os.path.abspath(options['output_folder'] + audio_file_name + "_instrum.wav")
        output_files["bass"] = os.path.abspath(options['output_folder'] + audio_file_name + "_bass.wav")
        output_files["drums"] = os.path.abspath(options['output_folder'] + audio_file_name + "_drums.wav")
        output_files["other"] = os.path.abspath(options['output_folder'] + audio_file_name + "_other.wav")

    # return individual paths
    return output_files["vocals"], output_files["instrumental"], output_files["bass"], output_files["drums"], output_files["other"]


iface = Interface(
    fn=separate_music_file_wrapper,
    inputs=[
        gr.Text(label="Input Directory"),
        gr.Checkbox(label="Use CPU Only", value=False),
        gr.Checkbox(label="Use Single ONNX", value=False),
        gr.Number(label="Large Overlap", value=0.6),
        gr.Number(label="Small Overlap", value=0.5),
        gr.Number(label="Chunk Size", value=1000000),
    ],
    outputs=[
        gr.Audio(label="Vocals"),
        gr.Audio(label="Instrumental"),
        gr.Audio(label="Bass"),
        gr.Audio(label="Drums"),
        gr.Audio(label="Other"),
    ]
)

iface.launch(debug=True)
