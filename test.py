import numpyclip
from PIL import Image
import numpy as np

model, preprocess = numpyclip.load("ViT-B/32")

image = preprocess(Image.open("data/CLIP.png"))[np.newaxis, :, :, :]

text = numpyclip.tokenize(["a diagram", "a dog", "a cat"])

image_features = model.encode_image(image)
text_features = model.encode_text(text)

logits_per_image, logits_per_text = model(image, text)
probs = numpyclip.softmax(logits_per_image, axis=-1)

expected_text_features = np.load("data/text_features.npy")
expected_image_features = np.load("data/image_features.npy")
expected_probs = np.array([[0.9927937, 0.00421068, 0.00299572]])

assert np.abs(expected_text_features - text_features).max() < 1e-5
assert np.abs(image_features - expected_image_features).max() < 1e-5
assert np.abs(probs - expected_probs).max() < 1e-5

print("Test passed :)")
