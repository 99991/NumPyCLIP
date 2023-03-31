# NumPyCLIP

Pure NumPy implementation of https://github.com/openai/CLIP

# Limitations

* NumPyCLIP is slower than the original PyTorch implementation if you have a powerful GPU.
* To reduce dependencies, [`ftfy.fix_text`](https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/simple_tokenizer.py#L51) has been removed from the tokenization step. This may cause differences when running NumPyCLIP on badly formatted text.
* The preprocessing of the image might differ from the official implementation by an offset of 1 pixel or so.
* So far, only the ViT-B/32 model has been ported.

