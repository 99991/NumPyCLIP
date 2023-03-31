# The original weights are stored as pth files, but we want as few dependencies
# as possible.
# This file loads the original CLIP model and scans the files in the pth file
# (which also is a zip file) for the file and byte offset of the weights.
# The offsets, shape, dtype and file names are stored as JSON.
import zipfile
import torch
import json
import re
import os
from clip.model import CLIP


def load_zip(path):
    files = {}

    with zipfile.ZipFile(path) as zip_file:
        for file_info in zip_file.infolist():
            with zip_file.open(file_info) as f:
                path = file_info.filename
                files[path] = f.read()

    return files


def find_weights(files, weights):
    weights_data = weights.cpu().numpy().tobytes()

    start_data = weights_data[:16]
    end_data = weights_data[-16:]

    for path, data in files.items():
        start = data.find(start_data)
        end = data.find(end_data)

        if start != -1 and end != -1:
            end += len(end_data)

            if data[start:end] != weights_data:
                raise ValueError(f"Found weights in {path}, but not all bytes match :( {weights.shape} {type(weights)} {weights.dtype}")

            # Delete file so future calls because weights have been found.
            # There should only be one set of weights per file.
            del files[path]

            return start, end, path

    raise ValueError(f"Did not find weights :( {weights.shape} {type(weights)} {weights.dtype}")


def main():
    path = os.path.expanduser("~/.cache/clip/ViT-B-32.pt")
    model = torch.jit.load(path, map_location="cpu").eval()
    state_dict = model.state_dict()

    files = load_zip(path)

    info = {}

    for key, value in state_dict.items():
        start, end, path = find_weights(files, value)
        info[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype).split(".")[-1],
            "start": start,
            "end": end,
            "path": path,
        }
        print(key, info[key])

        # TODO load this info from file instead of hacking it in here
        if key.endswith("attn.in_proj_weight"):
            key = ".".join(key.split(".")[:-1] + ["num_heads"])

            if key.startswith("visual."):
                num_heads = 12
            else:
                num_heads = 8

            info[key] = {
                "value": num_heads,
            }

    with open("ViT-B-32.json", "w") as f:
        json.dump(info, f, indent=4)


if __name__ == "__main__":
    main()
