import os
import numpy as np
import numpy.typing
import json
import zipfile
import urllib.request
import warnings
import typing
from simple_tokenizer import SimpleTokenizer
from PIL import Image


NDFloat32 = numpy.typing.NDArray[np.float32]
NDInt64 = numpy.typing.NDArray[np.int64]


def download(url: str, filename: str, chunk_size: int = 10**6) -> None:
    # Create directories if they don't exist yet
    directories = os.path.dirname(filename)
    if directories:
        os.makedirs(directories, exist_ok=True)

    # Download the file
    with urllib.request.urlopen(url) as response:
        total = int(response.info()["Content-Length"])

        buf = b""
        while True:
            data = response.read(chunk_size)
            if not data:
                break
            buf += data
            print(f"Downloading {filename} {len(buf) / total * 100:.2f} %")

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


class Params:
    def __init__(self, name: str, download_root: str = None) -> None:
        assert name == "ViT-B/32", f"Model {name} not supported yet. Only ViT-B-32 currently supported."

        model_urls = {
            "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
            "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
            "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
            "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
            "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
            "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
            "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
            "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
            "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
        }

        model_url = model_urls[name]

        name = name.replace("/", "-")

        if download_root is None:
            download_root = os.path.expanduser(f"~/.cache/clip")
            download_root = os.environ.get("CLIP_DIR", download_root)

        model_path = os.path.join(download_root, f"{name}.pt")

        if not os.path.isfile(model_path):
            print(f"Downloading {model_path} from {model_url}")
            download(model_url, model_path)

        self.files = load_zip(model_path)

        with open(f"data/{name}.json") as f:
            self.info = json.load(f)

    def get_int(self, name: str) -> int:
        info = self.info[name]

        value: int = info["value"]

        return value

    def __getitem__(self, name: str) -> NDFloat32:
        info = self.info[name]

        path = info["path"]
        dtype = info["dtype"]
        shape = info["shape"]
        start = info["start"]
        end = info["end"]

        assert dtype in ["float16", "float32"]

        data = self.files[path][start:end]

        arr = np.frombuffer(data, dtype=dtype).reshape(shape)
        arr = arr.astype(np.float32)

        return arr


def sigmoid(x: NDFloat32) -> NDFloat32:
    # Guard against overflow
    # 88.7 is a little bit less than np.log(np.finfo(np.float32).max)
    return 1.0 / (1.0 + np.exp(np.minimum(-x, 88.7)))


def softmax(x: NDFloat32, axis: int) -> NDFloat32:
    x = np.exp(x)
    x = x / x.sum(axis=axis, keepdims=True)
    return x


def build_attention_mask(context_length: int) -> NDFloat32:
    mask = np.full((context_length, context_length), fill_value=-np.inf, dtype=np.float32)
    mask = np.triu(mask, 1)
    return mask


def patch_project(x: NDFloat32, kernel: NDFloat32) -> NDFloat32:
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


def preprocess(image: Image.Image, image_size: int = 224) -> NDFloat32:
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
    x0 = round((width - image_size) / 2)
    y0 = round((height - image_size) / 2)
    x1 = x0 + image_size
    y1 = y0 + image_size
    image = image.crop((x0, y0, x1, y1))

    image = image.convert("RGB")

    # Normalize
    x = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    x = (x - mean) / std

    # HWC -> CHW
    x = x.transpose(2, 0, 1)

    return x


def tokenize(texts: typing.List[str], context_length: int = 77) -> NDInt64:
    tokenizer = SimpleTokenizer()

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]

    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]

    result = np.zeros((len(all_tokens), context_length), dtype=np.int64)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")

        result[i, : len(tokens)] = tokens

    return result


def multi_head_attention(x: NDFloat32, name: str, params: Params, attention_mask: typing.Optional[NDFloat32] = None) -> NDFloat32:
    W_qkv = params[name + ".in_proj_weight"]
    b_qkv = params[name + ".in_proj_bias"]
    W_out = params[name + ".out_proj.weight"]
    b_out = params[name + ".out_proj.bias"]
    num_heads = params.get_int(name + ".num_heads")

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

    out: NDFloat32 = heads.swapaxes(0, 1) @ W_out.T + b_out

    return out


def layer_norm(x: NDFloat32, name: str, params: Params, eps: float = 1e-5) -> NDFloat32:
    weight = params[name + ".weight"]
    bias = params[name + ".bias"]

    mean = x.mean(axis=-1, keepdims=True)
    var = np.square(x - mean).mean(axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(var + eps) * weight + bias

    return x


def mlp(x: NDFloat32, name: str, params: Params) -> NDFloat32:
    c_fc_w = params[name + ".c_fc.weight"]
    c_fc_b = params[name + ".c_fc.bias"]
    c_proj_w = params[name + ".c_proj.weight"]
    c_proj_b = params[name + ".c_proj.bias"]

    x = x @ c_fc_w.T + c_fc_b

    # QuickGELU activation
    x = x * sigmoid(1.702 * x)

    x = x @ c_proj_w.T + c_proj_b

    return x


def residual_attention_block(x: NDFloat32, name: str, params: Params, attention_mask: typing.Optional[NDFloat32] = None) -> NDFloat32:
    # Residual attention
    x = x + multi_head_attention(layer_norm(x, name + ".ln_1", params), name + ".attn", params, attention_mask=attention_mask)
    x = x + mlp(layer_norm(x, name + ".ln_2", params), name + ".mlp", params)
    return x


def encode_image(image: NDFloat32, params: Params) -> NDFloat32:
    class_embedding = params["visual.class_embedding"].reshape(1, 1, 768)
    positional_embedding = params["visual.positional_embedding"]

    x = patch_project(image, params["visual.conv1.weight"])
    x = np.concatenate([class_embedding, x], axis=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + positional_embedding
    x = layer_norm(x, "visual.ln_pre", params)

    x = x.transpose(1, 0, 2)  # NLD -> LND

    for i in range(12):
        x = residual_attention_block(x, f"visual.transformer.resblocks.{i}", params)

    x = x.transpose(1, 0, 2)  # LND -> NLD

    x = layer_norm(x[:, 0, :], "visual.ln_post", params)

    x = x @ params["visual.proj"]

    return x


def encode_text(text: NDInt64, params: Params) -> NDFloat32:
    x: NDFloat32 = params["token_embedding.weight"][text]

    positional_embedding = params["positional_embedding"]

    x = x + positional_embedding
    x = x.transpose(1, 0, 2)  # NLD -> LND
    attention_mask = build_attention_mask(x.shape[0])
    for i in range(12):
        x = residual_attention_block(x, f"transformer.resblocks.{i}", params, attention_mask=attention_mask)
    x = x.transpose(1, 0, 2)  # LND -> NLD
    x = layer_norm(x, "ln_final", params)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[np.arange(x.shape[0]), np.argmax(text, axis=-1)] @ params["text_projection"]

    return x


def image_text_logits(image: NDFloat32, text: NDInt64, params: Params) -> typing.Tuple[NDFloat32, NDFloat32]:
    image_features = encode_image(image, params)
    text_features = encode_text(text, params)

    # Normalize features
    image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
    text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

    scale = np.exp(params["logit_scale"])
    logits_per_image = scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    return logits_per_image, logits_per_text


class Model:
    def __init__(self, params: Params) -> None:
        self.params = params

    def encode_image(self, image: NDFloat32) -> NDFloat32:
        return encode_image(image, self.params)

    def encode_text(self, text: NDInt64) -> NDFloat32:
        return encode_text(text, self.params)

    def __call__(self, image: NDFloat32, text: NDInt64) -> typing.Tuple[NDFloat32, NDFloat32]:
        return image_text_logits(image, text, self.params)


def load(name: str, download_root: str = None) -> typing.Tuple[Model, typing.Callable[[Image.Image, int], NDFloat32]]:
    return Model(Params(name, download_root)), preprocess
