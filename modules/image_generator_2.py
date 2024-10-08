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
            weight_dtype = torch.float16

            transformer = Transformer2DModel.from_pretrained(
                "./PixArt-Sigma-900M",
                subfolder='transformer',
                torch_dtype=weight_dtype,
            )
            self.pipe = PixArtSigmaPipeline.from_pretrained(
                "./PixArt-Sigma-900M",
                transformer=transformer,
                torch_dtype=weight_dtype,
                use_safetensors=True,
            )

            self.pipe.to("cuda")

            # self.pipe = PixArtSigmaPipeline.from_pretrained(
            #     "./PixArt-Sigma-900M",
            #     torch_dtype=torch.float16,
            # ).to("cuda")

            # speed-up T5
            self.pipe.text_encoder.to_bettertransformer()

            self.prompt_generator = PromptGenerator()
        except Exception as e:
            print(e)
            raise Exception(str(e))

    def generate(self, request: ImageRequest):

        if request.randomize_seed:
            request.seed = random.randint(0, self.MAX_SEED)

        generator = torch.Generator().manual_seed(request.seed)

        try:
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

            image = self.pipe(
                prompt=generated_prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                generator=generator,
                max_sequence_length=self.T5_token_max_length,
                use_resolution_binning=request.use_resolution_binning,
            ).images[0]

            gc.collect()
            torch.cuda.empty_cache()

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
