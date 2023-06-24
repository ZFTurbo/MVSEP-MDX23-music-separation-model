import os
import time
import numpy as np
import tempfile
from scipy.io import wavfile
import gradio as gr
from inference import EnsembleDemucsMDXMusicSeparationModel, predict_with_model
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import asyncio

# prevent connection from being closed after inference (windows Error)
if os.name == 'nt':
    # Change  event loop policy to SelectorEventLoop instead of the default ProactorEventLoop
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")

def check_file_readiness(filepath):
    # If the loop finished, it means the file size has not changed for 5 seconds
    # which indicates that the file is ready
    num_same_size_checks = 0
    last_size = -1
    while num_same_size_checks < 5:
        current_size = os.path.getsize(filepath)
        if current_size == last_size:
            num_same_size_checks += 1
        else:
            num_same_size_checks = 0
            last_size = current_size
        time.sleep(0.5)
    return True

def generate_spectrogram(audio_file_path):
    y, sr = librosa.load(audio_file_path)
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                             y_axis='mel', fmax=22050, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    image_path = tempfile.mktemp('.png')
    plt.savefig(image_path)
    plt.close()
    return image_path

def generate_spectrograms(audio_files):
    output_spectrograms = []
    for audio_file in audio_files:
        output_spectrograms.append(generate_spectrogram(audio_file))
    return tuple(output_spectrograms)

def separate_music_file_wrapper(input_audio, use_cpu, use_single_onnx, large_overlap, small_overlap, chunk_size, use_large_gpu):
    print(f"type(input_audio): {type(input_audio)}, input_audio: {input_audio[:10]}") # truncate printout
    sample_rate, audio_data = input_audio
    output_file = "input_audio.wav"
    if isinstance(audio_data, np.ndarray):
        audio_data = audio_data.astype(np.int16)
    wavfile.write(output_file, sample_rate, audio_data)


    input_files = [output_file]

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

    print(f'use_cpu: {use_cpu}, use_large_gpu: {use_large_gpu}')
    predict_with_model(options)

    # Clear GPU cache once the separation finishes
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
    
    # Generate spectrograms right after separating the audio
    output_spectrograms = generate_spectrograms(output_files_ready)

    #print(len(output_files_ready)) # should print 6
    #print(len(output_spectrograms)) # should print 6
    return tuple(output_files_ready) + output_spectrograms

separation_description = """
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

theme = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="cyan",
)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown(separation_description)
    input_audio = gr.Audio(label="Upload Audio", interactive=True)
    use_cpu = gr.Checkbox(label="Use CPU Only", value=False)
    use_single_onnx = gr.Checkbox(label="Use Single ONNX", value=False)
    large_overlap = gr.Number(label="Large Overlap", value=0.6)
    small_overlap = gr.Number(label="Small Overlap", value=0.5)
    chunk_size = gr.Number(label="Chunk Size", value=1000000)
    use_large_gpu = gr.Checkbox(label="Use Fast Large GPU Version", value=False)    
    process_button = gr.Button("Process Audio")

    vocals = gr.Audio(label="Vocals")
    vocals_spectrogram = gr.Image(label="Vocals Spectrogram")
    instrumental = gr.Audio(label="Instrumental")
    instrumental_spectrogram = gr.Image(label="Instrumental Spectrogram")
    instrumental2 = gr.Audio(label="Instrumental 2")
    instrumental2_spectrogram = gr.Image(label="Instrumental 2 Spectrogram")
    bass = gr.Audio(label="Bass")
    bass_spectrogram = gr.Image(label="Bass Spectrogram")
    drums = gr.Audio(label="Drums")
    drums_spectrogram = gr.Image(label="Drums Spectrogram")
    other = gr.Audio(label="Other")
    other_spectrogram = gr.Image(label="Other Spectrogram")
    
    process_button.click(
        separate_music_file_wrapper,
        inputs=[input_audio, use_cpu, use_single_onnx, large_overlap, small_overlap, chunk_size, use_large_gpu],
        outputs=[vocals, instrumental, instrumental2, bass, drums, other, vocals_spectrogram, instrumental_spectrogram, instrumental2_spectrogram, bass_spectrogram, drums_spectrogram, other_spectrogram],
    )

demo.queue().launch(debug=True, share=False)