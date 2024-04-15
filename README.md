# Training code of Diffused Heads

### [Project](https://mstypulkowski.github.io/diffusedheads/) | [Paper](https://arxiv.org/abs/2301.03396) | [Demo](https://youtu.be/DSipIDj-5q0)
Here you can find information about training and evaluation of Diffused Heads. If you want to test our model on CREMA, please switch back to [main](https://github.com/MStypulkowski/diffused-heads/tree/main).

**Note**: No checkpoints or datasets are provided. This code was roughly cleaned and can have bugs. Please raise an issue to open discussion on your problem. We apologize for the delay in publishing the code.

## Checkpoints
CREMA checkpoint can be downloaded [here](https://drive.google.com/file/d/1QiUbnV4MfCYXTtICinDW4Frd_di7xphx/view?usp=sharing). No LRW checkpoint will be provided due to the license.

## Data
### Alignment
Our model works best on videos with the same alignment. To prepare videos, please use [face processor](https://github.com/DinoMan/face-processor). You can experiment with different offset values.

### Audio embeddings
Precompute audio embeddings for your dataset.

We worked with the one from [SDA](https://github.com/DinoMan/speech-driven-animation?tab=readme-ov-file#using-the-encodings). You can use part of the demo code from [main](https://github.com/MStypulkowski/diffused-heads/tree/main) where a scripted checkpoint is provided.

You are free to use any suitable audio encoder. Perhaps a better (and easier) choice is [Whisper Large](https://huggingface.co/openai/whisper-large-v3). Remember to change the dimension of audio embeddings in the config file, if needed.

### Folder structure
The provided dataset class works on predefined `file_list.txt` containing relative paths to video clips. Examples can be found in `datasets/` The data folder should contain subfolders `audio/` and `video/` with separate audio and video files.

## Scripts
To train the model, specify paths and parameters in `./configs/config.yaml`.
```
python train.py
```

To generate multiple test videos, specify paths and parameters in `./configs/config_gen_test.yaml`. 
```
python generate.py
```

To generate a video from any image/video and audio, specify paths and parameters in `./configs/config_gen_custom.yaml`. 
```
python custom_video.py
```

## Evaluation
The test splits for CREMA and LRW we used can be found in `datasets/`.

Metrics used:
* FVD: [Laughing Matters repo](https://github.com/antonibigata/Laughing-Matters/blob/6f0296d39ddf624c5b1e71214a311e7273dfb237/src/models/components/modules/metrics.py#L103)
* FID: [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html)
* Blinks/s and Blink duration: https://github.com/DinoMan/blink-detector
* OFM and F-MSE: `./smoothness_eval.py`
* AV offset and AV Confidence: https://github.com/joonson/syncnet_python
* WER: a pretrained lipreading model that we cannot share. You can use any available one.

## W&B
Our code supports W&B login. We left the code in the main scripts commented.

## Citation
```
@inproceedings{stypulkowski2024diffused,
  title={Diffused heads: Diffusion models beat gans on talking-face generation},
  author={Stypu{\l}kowski, Micha{\l} and Vougioukas, Konstantinos and He, Sen and Zi{\k{e}}ba, Maciej and Petridis, Stavros and Pantic, Maja},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5091--5100},
  year={2024}
}
```
## License
This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
