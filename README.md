# MVSEP-MDX23-music-separation-model
Model for MDX23 music separation contest. Model perform separation of music into 4 stems "bass", "drums", "vocals", "other".

It based on [Demucs4](https://github.com/facebookresearch/demucs), [MDX](https://github.com/kuielab/mdx-net) and some MDX weights from [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui). Brought to you by [MVSep.com](https://mvsep.com)
## Usage

```
    python inference.py --input_audio mixture1.wav mixture2.wav --output_folder ./result/
```

With this command audios with names "mixture1.wav" and "mixture2.wav" will be processed and results will be stored in `./result/` folder in WAV format.

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

## Citation

TBD
