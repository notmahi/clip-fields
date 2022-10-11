# CLIP-Fields: Weakly Supervised Semantic Fields for Robotic Memory

[[Paper]]() [[Website]](https://mahis.life/clip-fields/) [[Code]](https://github.com/notmahi/clip-fields) [[Data]](https://osf.io/famgv)

Authors: [Mahi Shafiullah](https://mahis.life), [Chris Paxton](https://cpaxton.github.io/), [Lerrel Pinto](https://lerrelpinto.com), [Soumith Chintala](https://soumith.ch), Arthur Szlam.

https://user-images.githubusercontent.com/3000253/195213301-43eae6e8-4516-4b8d-98e7-633c607c6616.mp4

**Tl;dr** CLIP-Field is a novel weakly supervised approach for learning a semantic robot memory that can respond to natural language queries solely from raw RGB-D and odometry data with no extra human labelling. It combines the image and language understanding capabilites of novel vision-language models (VLMs) like CLIP, large language models like sentence BERT, and open-label object detection models like Detic, and with spatial understanding capabilites of neural radiance field (NeRF) style architectures to build a spatial database that holds semantic information in it.

## Installation
To properly install this repo and all the dependencies, follow these instructions.

```
# Clone this repo.
git clone --recursive https://github.com/notmahi/clip-fields
cd clip-fields

# Create conda environment and install the dependencies.
conda create -n cf python=3.8
conda activate cf
conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install -r requirements.txt

# Install the hashgrid encoder with the relevant cuda module.
cd gridencoder
# For this part, it may be necessary to find out what your nvcc path is and use that, 
# For me $which nvcc gives public/apps/cuda/11.1/bin/nvcc, so I used the following part
# export CUDA_HOME=/public/apps/cuda/11.1
python setup.py install
cd ..
```

## Training
Once you have the dependencies installed, you can run the training script `train.py` with any [.r3d](https://record3d.app/) files that you have! If you just want to try out a sample, download the [sample data](https://osf.io/famgv) `nyu.r3d` and run the following command.

```
python train.py dataset_path=nyu.r3d
```

If you want to use LSeg as an additional source of open-label annotations, you should download the [LSeg demo model](https://github.com/isl-org/lang-seg#-try-demo-now) and place it in the `path_to_LSeg/checkpoints/demo_e200.ckpt`. Then, you can run the following command.

```
python train.py dataset_path=nyu.r3d use_lseg=true
```

You can check out the `config/train.yaml` for a list of possible configuration options. In particular, if you want to train with any particular set of labels, you can specify them in the `custom_labels` field in `config/train.yaml`.

## Interactive Tutorial and Evaluation
We have an interactive tutorial and evaluation notebook that you can use to explore the model and evaluate it on your own data. You can find them in the `demo/` directory, that you can run after installing the dependencies.

## Acknowledgements
We would like to thank the following projects for making their code and models available, which we relied upon heavily in this work.
* [CLIP](https://github.com/openai/CLIP) with [MIT License](https://github.com/openai/CLIP/blob/main/LICENSE)
* [Detic](https://github.com/facebookresearch/Detic/) with [Apache License 2.0](https://github.com/facebookresearch/Detic/blob/main/LICENSE)
* [Torch NGP](https://github.com/ashawkey/torch-ngp) with [MIT License](https://github.com/ashawkey/torch-ngp/blob/main/LICENSE)
* [LSeg](https://github.com/isl-org/lang-seg) with [MIT License](https://github.com/isl-org/lang-seg/blob/main/LICENSE)
* [Sentence BERT](https://www.sbert.net/) with [Apache License 2.0](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE)
