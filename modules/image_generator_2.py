from modules.generate_image_request_model import ImageRequest
from modules.prompt_generator import PromptGenerator

import torch
import random
import numpy as np
import ollama
from diffusers import PixArtSigmaPipeline, Transformer2DModel
from transformers import T5EncoderModel, BitsAndBytesConfig
from PIL import Image
from uuid import uuid4
import os
import gc

# Directory to save images
IMAGE_DIR = "./generated_images"
os.makedirs(IMAGE_DIR, exist_ok=True)


class ImageGenerator:
    _pipe = None
    MAX_SEED = np.iinfo(np.int32).max

    @classmethod
    def initialize_pipeline(cls):
        if cls._pipe is None:
            cls._pipe = PixArtSigmaPipeline.from_pretrained(
                "./PixArt-Sigma-900M",
                torch_dtype=torch.float16,
            ).to("cuda")
        return cls._pipe

    @classmethod
    def generate(cls, request: ImageRequest):
        # Ensure the pipeline is initialized
        pipe = cls.initialize_pipeline()

        if request.randomize_seed:
            request.seed = random.randint(0, cls.MAX_SEED)

        generator = torch.Generator().manual_seed(request.seed)

        try:
            if request.generate_prompt:
                generated_prompt = PromptGenerator.generate(
                    prompt=request.prompt,
                )
            else:
                generated_prompt = request.prompt

            image = pipe(
                prompt=generated_prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                generator=generator
            ).images[0]

            # Save the image with a unique filename
            image_filename = f"{uuid4()}.png"
            image_path = os.path.join(IMAGE_DIR, image_filename)
            image.save(image_path)

            return {
                "filename": image_filename,
                "generated_prompt":  generated_prompt,
                "seed": request.seed,
                "model": request.model
            }

        except Exception as e:
            print(e)
            raise Exception(str(e))
