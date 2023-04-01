# NumPyCLIP

This is a pure NumPy implementation of https://github.com/openai/CLIP.

You can use NumPyCLIP to embed images or text as a 512-dimensional feature vectors. The cosine similarity between feature vectors should be high when a text corresponds to an image.

For example, the image [`data/CLIP.png`](https://github.com/99991/NumPyCLIP/blob/main/data/CLIP.png) shows a diagram. When embedding the texts `"a diagram"`, `"a dog"` and `"a cat"` as feature vectors, the similarity of the image feature vector to the text feature vector for the text `"a diagram"` will be largest.

Possible applications include image classification, image captioning, visual question answering, text-based image search, image-based text search and image-filtering.

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

# Limitations

* NumPyCLIP is slower than the original PyTorch implementation if you have a powerful GPU.
* To reduce dependencies, [`ftfy.fix_text`](https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/simple_tokenizer.py#L51) has been removed from the tokenization step. This may cause differences when running NumPyCLIP on badly formatted text.
* The preprocessing of the image might differ from the official implementation by an offset of 1 pixel or so.
* So far, only the ViT-B/32 model has been ported.
* This library has not been tested much yet.
* During pre-processing, the input image is resized to 224x224 and center-cropped. For best results, make sure that all important content is in the centre of the image and of a reasonable size so that it is not lost when the image is scaled down.
