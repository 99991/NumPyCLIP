import numpyclip
from PIL import Image
import numpy as np
import csv

model, preprocess = numpyclip.load("ViT-B/32")

def test_features():
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

with open("data/labels.csv", encoding="utf-8") as f:
    data = list(csv.reader(f))

def test_classification():
    d = {label: filename for filename, label in data}
    labels = list(d.keys())
    text = numpyclip.tokenize(["a photo of " + label for label in labels])

    text_features = model.encode_text(text)
    text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

    num_correct = 0

    for n, (filename, label) in enumerate(data, 1):
        filename = "data/images/" + filename
        image = preprocess(Image.open(filename))[np.newaxis, :, :, :]

        image_features = model.encode_image(image)
        image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)

        scale = np.exp(model.params["logit_scale"])
        logits_per_image = scale * image_features @ text_features.T

        probs = numpyclip.softmax(logits_per_image, axis=-1)

        index = np.argmax(probs)

        num_correct += index == labels.index(label)

        print(f"{num_correct:3d} of {n:3d} correct, predicted label {labels[index]}, expected label {label}")

    assert num_correct == 100

if __name__ == "__main__":
    test_features()
    test_classification()
    print("Tests passed :)")
