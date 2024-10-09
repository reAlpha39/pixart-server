from modules.generate_image_request_model import ImageRequest
from modules.prompt_generator import PromptGenerator

import gc
import torch
import random
import numpy as np
from diffusers import PixArtSigmaPipeline, Transformer2DModel
from PIL import Image
from uuid import uuid4
import os

# Directory to save images
IMAGE_DIR = "./generated_images"
os.makedirs(IMAGE_DIR, exist_ok=True)


class ImageGenerator:
    def __init__(self):
        try:
            self.MAX_SEED = np.iinfo(np.int32).max
            self.T5_token_max_length = 300
            self.weight_dtype = torch.float16

            self.model_paths = {
                "PixArt-Sigma-900M": "./PixArt-Sigma-900M",
                "PixArt-Sigma-XL-2-1024-MS": "./PixArt-Sigma-XL-2-1024-MS"
            }

            self.current_model = "PixArt-Sigma-900M"
            self.pipe = self._load_model(self.current_model)

            self.prompt_generator = PromptGenerator()
        except Exception as e:
            print(e)
            raise Exception(str(e))

    def _load_model(self, model_name):

        if model_name == "PixArt-Sigma-XL-2-1024-MS":
            transformer = Transformer2DModel.from_pretrained(
                self.model_paths[model_name],
                subfolder='transformer',
                torch_dtype=self.weight_dtype,
            )
            pipe = PixArtSigmaPipeline.from_pretrained(
                "./pixart_sigma_sdxlvae_T5_diffusers",
                transformer=transformer,
                torch_dtype=self.weight_dtype,
                use_safetensors=True,
            )
        else:
            pipe = PixArtSigmaPipeline.from_pretrained(
                self.model_paths[model_name],
                torch_dtype=self.weight_dtype,
            )

        pipe.to("cuda")
        pipe.text_encoder.to_bettertransformer()
        return pipe

    def generate(self, request: ImageRequest):
        if request.randomize_seed:
            request.seed = random.randint(0, self.MAX_SEED)

        generator = torch.Generator().manual_seed(request.seed)

        try:
            # Change model if the loaded one is not the requested one
            if request.model != self.current_model:
                if request.model in self.model_paths:
                    print(f"Switching to model: {request.model}")
                    self.pipe = self._load_model(request.model)
                    self.current_model = request.model
                else:
                    print(f"Requested model {request.model} not available. Using {self.current_model}.")

            if request.generate_prompt:
                generated_prompt = self.prompt_generator.generate(
                    prompt_model=request.prompt_model,
                    generate_prompt_temperature=request.generate_prompt_temperature,
                    keep_alive_prompt_model=request.keep_alive_prompt_model,
                    prompt=request.prompt,
                )
                print(f'generated prompt: {generated_prompt}')
            else:
                generated_prompt = request.prompt

            with torch.no_grad():
                image = self.pipe(
                    prompt=generated_prompt,
                    negative_prompt=request.negative_prompt,
                    width=request.width,
                    height=request.height,
                    guidance_scale=request.guidance_scale,
                    num_inference_steps=request.num_inference_steps,
                    generator=generator,
                    max_sequence_length=self.T5_token_max_length,
                ).images[0]

            gc.collect()
            torch.cuda.empty_cache()

            image_filename = f"{uuid4()}.png"
            image_path = os.path.join(IMAGE_DIR, image_filename)
            image.save(image_path)

            print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024} GB")

            return {
                "filename": image_filename,
                "generated_prompt": generated_prompt,
                "seed": request.seed,
                "model": self.current_model
            }

        except Exception as e:
            print(e)
            raise Exception(str(e))
