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

# Inference

Using ["anything-v5"](https://huggingface.co/stablediffusionapi/anything-v5) as example. One can choose any interested pre-train model found on [hugginface](https://huggingface.co/) supported by diffusers.

```python
import torch
from diffusers import StableDiffusionPipeline

# loda model
model_id = "stablediffusionapi/anything-v5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# move to cuda
pipe = pipe.to("cuda")

# prompt
prompt = "1girl"
negative_prompt = ""

# using pipeline
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50, # NFE 
    guidance_scale=7.5,     # CFG
    safety_checker=None,  # Disable safety checker
    requires_safety_checker=False
).images[0]

# save result
image.save("result.png")
print("result saved!")
```
