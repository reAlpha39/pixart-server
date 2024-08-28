import torch
import random
import numpy as np
import ollama
import re
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from queue import Queue
from threading import Thread, Lock
import time
from pydantic import BaseModel
from typing import Optional
from diffusers import PixArtSigmaPipeline
from transformers import T5EncoderModel, BitsAndBytesConfig
from PIL import Image
from io import BytesIO
import base64
from uuid import uuid4
import os
import gc

app = FastAPI()

# Directory to save images
IMAGE_DIR = "./generated_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload= True,
)


MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "low quality, bad, watermark, human, people, person"
    seed: Optional[int] = 42
    randomize_seed: Optional[bool] = False
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    guidance_scale: Optional[float] = 3.5
    num_inference_steps: Optional[int] = 28
    url_output_type: Optional[bool] = False
    generate_prompt: Optional[bool] = False


def flush():
    gc.collect()
    torch.cuda.empty_cache()

@app.post("/generate-image/")
def generate_image(request: ImageRequest):
    if request.randomize_seed:
        request.seed = random.randint(0, MAX_SEED)
    
    generator = torch.Generator().manual_seed(request.seed)
    
    try:
        if request.generate_prompt:
            response = ollama.chat(
                model='gemma2:latest',
                keep_alive=0,
                messages=[
                    {
                        'role': 'user',
                        'content': request.prompt
                    }
                ]
            )
            generated_prompt = response['message']['content']
            generated_prompt = re.sub('\n', '', generated_prompt)
        else:
            generated_prompt = request.prompt

        text_encoder = T5EncoderModel.from_pretrained(
            "./PixArt-Sigma-900M",
            subfolder="text_encoder",
            quantization_config=quantization_config,
            device_map="auto",
        )

        pipe = PixArtSigmaPipeline.from_pretrained(
            "./PixArt-Sigma-900M",
            torch_dtype=torch.float16,
            text_encoder=text_encoder,
            transformer=None,
            device_map="balanced"
        )

        with torch.no_grad():
            prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(
                generated_prompt,
                negative_prompt=request.negative_prompt,
            )

        del text_encoder
        del pipe
        flush()

        pipe = PixArtSigmaPipeline.from_pretrained(
            "./PixArt-Sigma-900M",
            text_encoder=None,
            torch_dtype=torch.float16,
        ).to("cuda")

        # pipe.transformer = torch.compile(
        #     pipe.transformer, mode="reduce-overhead", fullgraph=True)


        latents = pipe(
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            num_images_per_prompt=1,
            output_type="latent",
            width=request.width,
            height=request.height,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            generator=generator
        ).images

        del pipe.transformer
        flush()

        with torch.no_grad():
            image = pipe.vae.decode(
                latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]

        # image = pipe(
        #     # prompt=request.prompt,
        #     negative_prompt=request.negative_prompt,
        #     width=request.width,
        #     height=request.height,
        #     guidance_scale=request.guidance_scale,
        #     num_inference_steps=request.num_inference_steps,
        #     generator=generator
        # ).images[0]

        if request.url_output_type:
            del pipe
            flush()
            # Save the image with a unique filename
            image_filename = f"{uuid4()}.png"
            image_path = os.path.join(IMAGE_DIR, image_filename)
            image.save(image_path)

            # Return the image URL
            image_url = f"/images/{image_filename}"
            return {
                "image_url": image_url,
                "generated_prompt":  generated_prompt,
                "seed": request.seed
            }
        
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        del pipe
        flush()
        
        return {
            "image": img_str,
            "generated_prompt":  generated_prompt,
            "seed": request.seed
        }
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to serve the images
@app.get("/images/{image_filename}")
def get_image(image_filename: str):
    image_path = os.path.join(IMAGE_DIR, image_filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)


@app.get("/")
def read_root():
    return {"message": "Welcome to the PixArt Sigma 900M Image Generation API"}

