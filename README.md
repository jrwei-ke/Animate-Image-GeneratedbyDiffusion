# Animate-Image-GeneratedbyDiffusion

This repo is based on ["Stable Diffusion"](https://arxiv.org/abs/2112.10752), 
code on ["diffusers"](https://github.com/huggingface/diffusers), 
model on ["anything-v5"](https://huggingface.co/stablediffusionapi/anything-v5) and 
["sdxl"](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

# Environment
```bash
conda create --name AIGD python=3.10
```

# Install
```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers/
pip install -e .
cd text_to_image/
pip install -r requirements.txt
```
If there is a problem of peft version, directly specify the version of peft.
```bash
pip install peft==0.17.0
```
