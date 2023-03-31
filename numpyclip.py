import os
import numpy as np
from PIL import Image
import json
import zipfile
import urllib.request
import warnings
import typing
from simple_tokenizer import SimpleTokenizer


def download(url: str, filename: str) -> None:
    # Create directories if they don't exist yet
    directories = os.path.dirname(filename)
    if directories:
        os.makedirs(directories, exist_ok=True)

    # Download the file
    with urllib.request.urlopen(url) as response:
        total = int(response.info()["Content-Length"])

        buf = b""
        while True:
            data = response.read(10**6)
            if not data:
                break
            buf += data
            print(f"Downloading model... {len(buf) / total * 100:.2f} %")

    # Write the downloaded data to the file
    with open(filename, "wb") as f:
        f.write(buf)


def load_zip(path: str) -> typing.Dict[str, bytes]:
    files = {}

    with zipfile.ZipFile(path) as z:
        for file_info in z.infolist():
            with z.open(file_info) as f:
                path = file_info.filename
                files[path] = f.read()

    return files


model_url = "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"

model_path = os.path.expanduser("~/.cache/clip/ViT-B-32.pt")

if not os.path.isfile(model_path):
    download(model_url, model_path)

files = load_zip(model_path)

with open("data/ViT-B-32.json") as f:
    weights_info = json.load(f)


def get_weights(name: str):
    info = weights_info[name]

    if "value" in info:
        return info["value"]

    path = info["path"]
    dtype = info["dtype"]
    shape = info["shape"]
    start = info["start"]
    end = info["end"]

    assert dtype in ["float16", "float32"]

    data = files[path][start:end]

    weights = np.frombuffer(data, dtype=dtype).reshape(shape)
    weights = weights.astype(np.float32)

    return weights


def softmax(x, axis: int):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def multi_head_attention(x, name, attention_mask=None):
    W_qkv = get_weights(name + ".in_proj_weight")
    b_qkv = get_weights(name + ".in_proj_bias")
    W_out = get_weights(name + ".out_proj.weight")
    b_out = get_weights(name + ".out_proj.bias")
    num_heads = get_weights(name + ".num_heads")

    assert W_qkv.shape[0] == W_qkv.shape[1] * 3 == x.shape[-1] * 3
    assert W_out.shape[0] == W_out.shape[1] == x.shape[-1]
    assert x.shape[-1] % num_heads == 0

    x = x.transpose(1, 0, 2)[np.newaxis, :, :, :]
    d = x.shape[-1]
    W_q, W_k, W_v = W_qkv.reshape(3, num_heads, 1, -1, d).swapaxes(-2, -1)
    b_q, b_k, b_v = b_qkv.reshape(3, num_heads, 1, 1, -1)

    # Scaled dot product attention for all heads at once
    scale = 1.0 / (d // num_heads) ** 0.5
    qk = scale * (x @ W_q + b_q) @ (x @ W_k + b_k).swapaxes(-2, -1)
    if attention_mask is not None:
        qk += attention_mask
    heads = softmax(qk, axis=-1) @ (x @ W_v + b_v)

    # Concatenate heads by concatenating first and last axis
    heads = np.concatenate(tuple(heads), axis=2)

    out = heads.swapaxes(0, 1) @ W_out.T + b_out

    return out


def build_attention_mask(context_length):
    mask = np.full((context_length, context_length), fill_value=-np.inf, dtype=np.float32)
    mask = np.triu(mask, 1)
    return mask


def layer_norm(x, ln, eps=1e-5):
    weight = get_weights(ln + ".weight")
    bias = get_weights(ln + ".bias")

    mean = x.mean(axis=-1, keepdims=True)
    var = np.square(x - mean).mean(axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(var + eps) * weight + bias
    return x


def patch_project(x, kernel):
    # Decompose images into 32x32 patches and multiply all patches by matrix.

    n, c, h, w = x.shape
    d, pc, ph, pw = kernel.shape
    p = pc * ph * pw
    gh = h // ph
    gw = w // pw

    assert c == pc and h % ph == 0 and w % pw == 0

    # (d, pc, ph, pw) -> (pc, ph, pw, d) -> (pc * ph * pw, d) = (p, d)
    W = kernel.transpose(1, 2, 3, 0).reshape(p, d)
    # (n, c, h, w) -> (n, c, gh, ph, gw, pw) -> (n, gh, gw, c, ph, pw) -> (n, gh, gw, p)
    x = x.reshape(n, c, gh, ph, gw, pw).transpose(0, 2, 4, 1, 3, 5).reshape(n, gh, gw, p)
    # (n, gh, gw, p) @ (p, d) = (n, gh, gw, d)
    x = x @ W
    # (n, gh, gw, d) -> (n, gh * gw, d)
    x = x.reshape(n, gh * gw, d)

    return x


def sigmoid(x):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return 1.0 / (1.0 + np.exp(-x))


def mlp(x, name):
    c_fc_w = get_weights(name + ".c_fc.weight")
    c_fc_b = get_weights(name + ".c_fc.bias")
    c_proj_w = get_weights(name + ".c_proj.weight")
    c_proj_b = get_weights(name + ".c_proj.bias")

    x = x @ c_fc_w.T + c_fc_b

    # QuickGELU activation
    x = x * sigmoid(1.702 * x)

    x = x @ c_proj_w.T + c_proj_b

    return x


def residual_attention_block(x, name, attention_mask=None):
    # Residual attention
    x = x + multi_head_attention(layer_norm(x, name + ".ln_1"), name + ".attn", attention_mask=attention_mask)
    x = x + mlp(layer_norm(x, name + ".ln_2"), name + ".mlp")
    return x


def _preprocess(image, image_size=224):
    # Scale image such that length of smaller side is 224
    width, height = image.size
    scale = image_size / min(width, height)
    width = int(scale * width)
    height = int(scale * height)
    # Some Pillow versions have different interface
    if hasattr(Image, "Resampling"):
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    else:
        image = image.resize((width, height), Image.BICUBIC)

    # Crop center
    x = round((width - image_size) / 2)
    y = round((height - image_size) / 2)
    image = image.crop((x, y, x + image_size, y + image_size))

    image = image.convert("RGB")

    # Normalize
    image = np.array(image, dtype=np.float32) / 255.0
    mean = np.float32([0.48145466, 0.4578275, 0.40821073])
    std = np.float32([0.26862954, 0.26130258, 0.27577711])
    image = (image - mean) / std

    # HWC -> CHW
    image = image.transpose(2, 0, 1)

    return image


def compute_text_features(text):
    x = get_weights("token_embedding.weight")[text]

    positional_embedding = get_weights("positional_embedding")

    x = x + positional_embedding
    x = x.transpose(1, 0, 2)  # NLD -> LND
    attention_mask = build_attention_mask(x.shape[0])
    for i in range(12):
        x = residual_attention_block(x, f"transformer.resblocks.{i}", attention_mask=attention_mask)
    x = x.transpose(1, 0, 2)  # LND -> NLD
    x = layer_norm(x, "ln_final")

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[np.arange(x.shape[0]), np.argmax(text, axis=-1)] @ get_weights("text_projection")

    return x


def compute_image_features(image):
    class_embedding = get_weights("visual.class_embedding").reshape(1, 1, 768)
    positional_embedding = get_weights("visual.positional_embedding")

    x = patch_project(image, get_weights("visual.conv1.weight"))
    x = np.concatenate([class_embedding, x], axis=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + positional_embedding
    x = layer_norm(x, "visual.ln_pre")

    x = x.transpose(1, 0, 2)  # NLD -> LND

    for i in range(12):
        x = residual_attention_block(x, f"visual.transformer.resblocks.{i}")

    x = x.transpose(1, 0, 2)  # LND -> NLD

    x = layer_norm(x[:, 0, :], "visual.ln_post")

    x = x @ get_weights("visual.proj")

    return x


def tokenize(texts, context_length=77):
    tokenizer = SimpleTokenizer()

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]

    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]

    result = np.zeros((len(all_tokens), context_length), dtype=np.int32)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")

        result[i, : len(tokens)] = tokens

    return result


class Model:
    def __init__(self):
        pass

    def encode_image(self, image):
        return compute_image_features(image)

    def encode_text(self, text):
        return compute_text_features(text)

    def __call__(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # Normalize features
        image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

        scale = np.exp(get_weights("logit_scale"))
        logits_per_image = scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text


def load(name: str):
    return Model(), _preprocess
