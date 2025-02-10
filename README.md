# MVSEP-MDX23-music-separation-model
Model for [Sound demixing challenge 2023: Music Demixing Track - MDX'23](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023). Model perform separation of music into 4 stems "bass", "drums", "vocals", "other". Model won 3rd place in challenge (Leaderboard C).

Model based on [Demucs4](https://github.com/facebookresearch/demucs), [MDX](https://github.com/kuielab/mdx-net) neural net architectures and some MDX weights from [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) project (thanks [Kimberley Jensen](https://github.com/KimberleyJensen) for great high quality vocal models). Thanks [@Ma5onic](https://github.com/Ma5onic) for web UI & helping with dataset augmentation techniques. Brought to you by [MVSep.com](https://mvsep.com).
## Usage

```
    python inference.py --input_audio mixture1.wav mixture2.wav --output_folder ./results/
```

With this command audios with names "mixture1.wav" and "mixture2.wav" will be processed and results will be stored in `./results/` folder in WAV format.

### All available keys
* `--input_audio` - input audio location. You can provide multiple files at once. **Required**
* `--output_folder` - output audio folder. **Required**
* `--cpu` - choose CPU instead of GPU for processing. Can be very slow.
* `--overlap_large` - overlap of splitted audio for light models. Closer to 1.0 - slower, but better quality. Default: 0.6.
* `--overlap_small` - overlap of splitted audio for heavy models. Closer to 1.0 - slower, but better quality. Default: 0.5.
* `--single_onnx` - only use single ONNX model for vocals. Can be useful if you have not enough GPU memory.
* `--chunk_size` - chunk size for ONNX models. Set lower to reduce GPU memory consumption. Default: 1000000.
* `--large_gpu` - it will store all models on GPU for faster processing of multiple audio files. Requires at least 11 GB of free GPU memory.
* `--use_kim_model_1` - use first version of Kim model (as it was on contest).
* `--only_vocals` - only create vocals and instrumental. Skip bass, drums, other. Processing will be faster.

### Notes
* If you have not enough GPU memory you can use CPU (`--cpu`), but it will be slow. Additionally you can use single ONNX (`--single_onnx`), but it will decrease quality a little bit. Also reduce of chunk size can help (`--chunk_size 200000`).
* In current revision code requires less GPU memory, but it process multiple files slower. If you want old fast method use argument `--large_gpu`. It will require > 11 GB of GPU memory, but will work faster.
* There is [Google.Collab version](https://colab.research.google.com/github/jarredou/MVSEP-MDX23-Colab_v2/blob/main/MVSep-MDX23-Colab.ipynb) of this code.  

## Quality comparison

Quality comparison with best separation models performed on [MultiSong Dataset](https://mvsep.com/quality_checker/leaderboard2.php?sort=bass). 

| Algorithm     | SDR bass  | SDR drums  | SDR other  | SDR vocals  | SDR instrumental  |
| ------------- |:---------:|:----------:|:----------:|:----------:|:------------------:|
| MVSEP MDX23   | 12.5034   | 11.6870    | 6.5378     |  9.5138    | 15.8213            |
| Demucs HT 4   | 12.1006   | 11.3037    | 5.7728     |  8.3555    | 13.9902            |
| Demucs 3      | 10.6947   | 10.2744    | 5.3580     |  8.1335    | 14.4409            |
| MDX B         | ---       | ----       | ---        |  8.5118    | 14.8192            |

* Note: SDR - signal to distortion ratio. Larger is better.

## GUI

![GUI Window](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model/blob/main/images/MVSep-Window.png)

* Script for GUI (based on PyQt5): [gui.py](gui.py).
* You can download [standalone program for Windows here](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model/releases/download/v1.0.1/MVSep-MDX23_v1.0.1.zip) (~730 MB). Unzip archive and to start program double click `run.bat`. On first run it will download pytorch with CUDA support (~2.8 GB) and some Neural Net models.
* Program will download all needed neural net models from internet at the first run.
* GUI supports Drag & Drop of multiple files.
* Progress bar available.

## Web Interface
executing `web-ui.py` with python will start the web interface locally on `localhost` (127.0.0.1).
You'll see what port it is running on within the terminal output.

![image](https://github.com/Ma5onic/MVSEP-MDX23-music-separation-model/assets/18509613/ae7130a5-60a4-4095-abbd-5290e84dcf7c)

* Browser-Based user interface
* Program will download all needed neural net models from internet at the first run.
* supports Drag & Drop for audio upload (single file)

![Web-UI Window](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model/assets/18509613/4872f6aa-5896-44e9-8885-eaee1de3f4ee)


## Changes

### v1.0.1
* Settings in GUI updated, now you can control all possible options
* Kim vocal model updated from version 1 to version 2, you still can use version 1 using parameter `--use_kim_model_1`
* Added possibility to generate only vocals/instrumental pair if you don't need bass, drums and other stems. Use parameter `--only_vocals`
* Standalone program was updated. It has less size now. GUI will download torch/cuda on the first run. 

## Citation

* [arxiv paper](https://arxiv.org/abs/2305.07489)

```
@misc{solovyev2023benchmarks,
      title={Benchmarks and leaderboards for sound demixing tasks}, 
      author={Roman Solovyev and Alexander Stempkovskiy and Tatiana Habruseva},
      year={2023},
      eprint={2305.07489},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}

@article{fabbro2024sound,
    title={The Sound Demixing Challenge 2023-Music Demixing Track.},
    author={Fabbro, G., Uhlich, S., Lai, C.-H., Choi, W., Martínez-Ramírez, M., Liao, W., Gadelha, I., Ramos, G., Hsu, E., Rodrigues, H., Stöter, F.-R.,
    Défossez, A., Luo, Y., Yu, J., Chakraborty, D., Mohanty, S., Solovyev, R., Stempkovskiy, A., Habruseva, T., Goswami, N., Harada, T., Kim, M.,
    Lee, J. H., Dong, Y., Zhang, X., Liu, J., & Mitsufuji, Y},
    journal={Trans. Int. Soc. Music. Inf. Retr.},
    volume={7},
    number={1},
    pages={63--84},
    year={2024}
}
```
