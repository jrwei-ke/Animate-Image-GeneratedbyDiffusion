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
