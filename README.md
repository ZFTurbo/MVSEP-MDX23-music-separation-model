# MVSEP-MDX23-music-separation-model
Model for MDX23 music separation contest. Model perform separation of music into 4 stems "bass", "drums", "vocals", "other".

It based on [Demucs4](https://github.com/facebookresearch/demucs), [MDX](https://github.com/kuielab/mdx-net) and some MDX weights from [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui). Brought to you by [MVSep.com](https://mvsep.com)
## Usage

```
    python inference.py --input_audio mixture1.wav mixture2.wav --output_folder ./result/
```

With this command audios with names "mixture1.wav" and "mixture2.wav" will be processed and results will be stored in `./result/` folder in WAV format.

## Quality comparison

TBD 

## GUI

TBD

## Citation

TBD
