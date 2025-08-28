# Animate-Image-GeneratedbyDiffusion

This repo is based on ["Stable Diffusion"](https://arxiv.org/abs/2112.10752), 
["LoRA"](https://arxiv.org/abs/2106.09685), 
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

# Finetune

Using Lora:  

Build one's own dataset first.  
```bash
cd /diffusers/examples/text_to_image
mkdir datasets
cd datasets
touch metadata.jsonl
```
Put the images in this created "datasets" folder and edit metadata.jsonl like:
```json
{"file_name": "image1.jpeg", "text": "1girl, animate girl."}
{"file_name": "image2.jpeg", "text": "1girl, animate girl."}
```
file_names should be the same as one's images, and text, the prompts descrbing the correponding image, should be well prompt!!!  
  
You can prompt your training images by yourself. If you have lots experence on prompting, you will get a high quality datasets.  

On the other hand, one can use VLLM to captioning images and producing prompts which is an efficient way but maybe unstable.  

Here is an example to ask an image to Gemini and gets the prompts:
```
I want to build a list of prompts describing this image for LoRA training of stable diffusion. please look at this image and combine the acknowlege of prompt engineering to give me the best prompts of this image.
```
You can copy and paste the provided prompts to the position of "text" directly; however, a liitle bit modified is recommended.  

After dataset done, be sure in the virtual environment and install the dependencies.
```bash
cd ..
```
or be sure in the correct folder.
```bash
cd /diffusers/examples/text_to_image
```
According to the base model, install the dependencies coreespond to the requirements.
```bash
pip install -r requirements.txt
pip install -r requirements_flax.txt
pip install -r requirements_sdxl.txt
```
Start Training.  
The following hyperparameters mean:  
pretrained_model_name_or_path -> base model  
train_data_dir -> name of dataset folder  
caption_column -> prompts  
resolution -> image input size (512 x 512)  
train_batch_size -> every grad update based on the number of images  
num_train_epochs -> how many trains based on the whole dataset  
checkpointing_steps -> every checkpoint saves the weights  
learning_rate -> the length every step taken by the optimizer to update the weights  
lr_scheduler -> learning rate modified in the training or not  
lr_warmup_steps -> learning rate starts from a very small number or not  
output_dir -> where the LoRA weights saved  
```bash
train_text_to_image_lora.py   --pretrained_model_name_or_path="stablediffusionapi/anything-v5"   --train_data_dir="datasets" --caption_column="text"   --resolution=512 --random_flip   --train_batch_size=1 --num_train_epochs=10 --checkpointing_steps=10   --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0   --output_dir="sd15-lora
```
```bash
accelerate launch train_text_to_image_lora.py   --pretrained_model_name_or_path="stablediffusionapi/anything-v5"   --train_data_dir="datasets" --caption_column="text"   --resolution=512 --random_flip   --train_batch_size=1 --num_train_epochs=100 --checkpointing_steps=10   --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0   --output_dir="sd15-lora" 
```

