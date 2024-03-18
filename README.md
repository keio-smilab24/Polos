# 🌟 [CVPR24] Polos: Multimodal Metric Learning from Human Feedback for Image Captioning
[![Conference](https://img.shields.io/badge/CVPR-2024-bbeaff.svg)](https://arxiv.org/abs/2402.18091)
[![arXiv](https://img.shields.io/badge/arXiv-2402.18091-b31b1b.svg)](https://arxiv.org/abs/2402.18091)

- Accepted at CVPR 2024
- [project page](https://yuiga.dev/polos)
- [arXiv](https://arxiv.org/abs/2402.18091)

*Establishing an automatic evaluation metric that closely aligns with human judgements is essential for the effective development of image captioning models. Data-driven metrics have recently gained prominence in this field, demonstrating a stronger correlation with human judgements than classic metrics such as CIDEr and SPICE. However, these approaches pose challenges; for instance, they lack sufficient capabilities to handle hallucinations and to generalize across various types of images and texts. This limitation is partly attributed to the fact that existing approaches compute scalar similarities merely using embeddings learned from tasks that are not directly related to image captioning evaluation. In this study, we propose Polos, a supervised automatic evaluation metric tailored for image captioning models. To enhance robustness and practicality, we also present Multimodal Metric Learning from Human Feedback (M
LHF), a novel framework for developing metrics based on human feedback. In line with the principles of M
LHF, Polos is trained directly from human feedback and computes evaluation scores using multimodal inputs, employing a parallel feature extraction mechanism that leverages SimCSE and CLIP. This mechanism enables our metric to effectively model intricate relationships within the vector space of text-image pairs as well as text-text pairs. In addition, we have constructed a large-scale dataset for M
LHF, which comprises 131K human judgements collected from 550 evaluators. Our dataset further distinguishes itself from existing datasets in terms of the inclusion of diverse captions, which are collected from humans and generated from ten image captioning models, including modern models. Our approach has achieved state-of-the-art performance on various image captioning benchmarks, including Composite, Flickr8K-Expert, Flickr8K-CF, FOIL, and our dataset, demonstrating its effectiveness and robustness.*

## Quick Installation

To install the Polos library, run the following command in your terminal:
```bash
pip install polos
```

To use the Polos library, follow the steps outlined below. This example demonstrates how to use the library with an image and associated text data for machine learning prediction tasks.

### Step 1: Importing Required Libraries
Import the necessary libraries from Polos and other dependencies.
```python
from polos.models import download_model, load_checkpoint
from PIL import Image
```

### Step 2: Downloading and Loading the Model
Download and load the model using the Polos library functions.
```python
model_path = download_model("polos")
model = load_checkpoint(model_path)
```

### Step 3: Preparing the Data
Prepare your data in the specified format. Here, we are using an example image `test.jpg` and associated text data.
```python
data = [
    {
        "img": Image.open("test.jpg").convert("RGB"),
        "mt": "a dog with a person",
        "refs": ["there is a dog sitting on a couch with a person reaching out", "a dog laying on a couch with a person", "a dog is laying on a couch with a person"],
    }
]
```

### Step 4: Making Predictions
Use the model to make predictions on your data.
```python
_, scores = model.predict(data, batch_size=8, cuda=True)
print(scores)
```

### Note
- Ensure that the image `test.jpg` is correctly located in your working directory. ([test.jpg](https://upload.wikimedia.org/wikipedia/en/5/5f/Original_Doge_meme.jpg))
- The `cuda=True` argument in the `predict` function indicates the use of CUDA for GPU acceleration. Make sure you have a compatible NVIDIA GPU and CUDA installed.

```python
from polos.models import download_model, load_checkpoint
from PIL import Image

model_path = download_model("polos")
model = load_checkpoint(model_path)

# Data must be in the following format:
data = [
    {
        "img": Image.open("test.jpg").convert("RGB"),
        "mt": "a dog with a person",
        "refs":["there is a dog sitting on a couch with a person reaching out", "a dog laying on a couch with a person", 'a dog is laying on a couch with a person'],
    }
]

_, scores = model.predict(data, batch_size=8, cuda=True)
print(scores)
```

## Instructions (for researchers)

<img height="500" alt="system" src="https://github.com/keio-smilab24/Polos/assets/51681991/30b12191-224a-4293-b424-54a5e1948173">


We assume the following environment for our experiments:

- Python 3.10.0 (pyenv is strongly recommended)
- [Poetry](https://github.com/python-poetry/poetry) for dependency management (refer to Poetry documentation)
- PyTorch version 2.1.0 with CUDA 11.8 support
- PyTorch Lightning for model training facilitation

### Clone & Install

```bash
git clone git@github.com:keio-smilab24/Polos.git
cd Polos
```

```bash
pyenv virtualenv 3.10.0 polos
pyenv local polos
sh install.sh # cuda=11.8
```

### Datasets

- Polaris
  - The Polaris dataset can be downloaded at [this link](https://polos-polaris.s3.ap-northeast-1.amazonaws.com/polaris.zip).
  - Unzip and extract the contents into the `data_en` directory.
- Flickr8k
  - We evaluate Flickr8K according to the [PAC-S](https://github.com/aimagelab/pacscore) pre-processing.
  - Download the dataset from [this link](https://drive.google.com/drive/folders/1oQY8zVCmf0ZGUfsJQ_OnqP2_kw1jGIXp) provided by the PAC-S authors.
  - Once you have downloaded the dataset, place them under the `data_en/flickr8k` folder.
- Composite / PASCAL-50S / FOIL
  - For the Composite, PASCAL-50S, and FOIL datasets, download them from the following links:
  - [Composite](https://imagesdg.wordpress.com/image-to-scene-description-graph/)
  - [PASCAL-50S](https://vrama91.github.io/cider/)
  - [FOIL](https://foilunitn.github.io/)


### Checkpoint

The best checkpoint can be downloaded at [this link](https://polos-polaris.s3.ap-northeast-1.amazonaws.com/reprod.zip). Unzip and extract the `checkpoints`.


### Train

```bash
sh train.sh
```

### Evaluation

PAC-S checkpoints are required to assess PAC-S. 

Download the checkpoints according to the instructions on the [authors' github](https://github.com/aimagelab/pacscore) and place them in the specified locations.

```bash
sh validate.sh
```

## Bibtex

```
@inproceedings{wada2024,
    title     = {{Polos: Multimodal Metric Learning from Human Feedback for Image Captioning}},
    author    = {Wada, Yuiga  and Kaneda, Kanta and Saito, Daichi and Sugiura, Komei},
    year      = 2024,
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
}
```

## License

This work is licensed under the BSD-3-Clause-Clear license. To view a copy of this license, see [LICENSE](LICENSE).
