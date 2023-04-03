# NumPyCLIP

This is a pure NumPy implementation of https://github.com/openai/CLIP.

You can use NumPyCLIP to embed images or text as a 512-dimensional feature vectors. The cosine similarity between feature vectors should be high when a text corresponds to an image.

For example, the image [`data/CLIP.png`](https://github.com/99991/NumPyCLIP/blob/main/data/CLIP.png) shows a diagram. When embedding the texts `"a diagram"`, `"a dog"` and `"a cat"` as feature vectors, the similarity of the image feature vector to the text feature vector for the text `"a diagram"` will be largest.

Possible applications include image classification, image captioning, visual question answering, text-based image search, image-based text search and image-filtering.

# [Example](https://github.com/99991/NumPyCLIP/blob/main/example.py)

```python
import numpyclip
import numpy as np
from PIL import Image

model, preprocess = numpyclip.load("ViT-B/32")

image = preprocess(Image.open("data/CLIP.png"))[np.newaxis, :, :, :]

text = numpyclip.tokenize(["a diagram", "a dog", "a cat"])

image_features = model.encode_image(image)
text_features = model.encode_text(text)

logits_per_image, logits_per_text = model(image, text)
probs = numpyclip.softmax(logits_per_image, axis=-1)

print("Label probs:", probs)  # prints: [[0.99279356 0.00421067 0.00299573]]
```

# Install dependencies, download and run on Debian/Ubuntu

```
sudo apt update
sudo apt install python3 python3-pip git
pip3 install numpy pillow
git clone --depth 1 https://github.com/99991/NumPyCLIP.git
cd NumPyCLIP
python3 example.py
python3 tests.py
```

This will install Python, git, NumPy and Pillow (for image loading). Once the dependencies are installed, it will download NumPyCLIP and run [`example.py`](https://github.com/99991/NumPyCLIP/blob/main/example.py). The first time, the file `~/.cache/clip/ViT-B-32.pt` (337.6 MiB) will be downloaded, which may take a few minutes.

By default, the model weights will be downloaded to `~/.cache/CLIP`, but you can also specify the directory with the `CLIP_DIR` environment variable:

```bash
# Download weights to "my/weights/directory"
CLIP_DIR=my/weights/directory python3 tests.py
```

# Limitations

* NumPyCLIP is slower than the original PyTorch implementation if you have a powerful GPU.
* To reduce dependencies, [`ftfy.fix_text`](https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/simple_tokenizer.py#L51) has been removed from the tokenization step. This may cause differences when running NumPyCLIP on badly formatted text.
* The preprocessing of the image might differ from the official implementation by an offset of 1 pixel or so.
* So far, only the ViT-B/32 model has been ported.
* This library has not been tested much yet.
* During pre-processing, the input image is resized to 224x224 and center-cropped. For best results, make sure that all important content is in the centre of the image and of a reasonable size so that it is not lost when the image is scaled down.

# TODO

* Test more images and text with weird unicode symbols
* Implement other models
* Package for PyPi
* Investigate whether it is safe to ignore the RuntimeWarning about infinity during computation of [`sigmoid`](https://github.com/99991/NumPyCLIP/blob/68cbd9254d4696d9ab5b4cd39e7d150547251740/numpyclip.py#L108)
