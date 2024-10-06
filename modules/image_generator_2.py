from modules.generate_image_request_model import ImageRequest
from modules.prompt_generator import PromptGenerator

import torch
import random
import numpy as np
from diffusers import PixArtSigmaPipeline
from PIL import Image
from uuid import uuid4
import os

# Directory to save images
IMAGE_DIR = "./generated_images"
os.makedirs(IMAGE_DIR, exist_ok=True)


class ImageGenerator:
    def __init__(self):
        self._pipe = PixArtSigmaPipeline.from_pretrained(
            "./PixArt-Sigma-900M",
            torch_dtype=torch.float16,
        ).to("cuda")

        # # speed-up T5
        # self._pipe.text_encoder.to_bettertransformer()

        # # Compile Model
        # self._pipe.transformer = torch.compile(
        #     self._pipe.transformer,
        #     mode="reduce-overhead",
        #     fullgraph=True,
        # )

        self.MAX_SEED = np.iinfo(np.int32).max

    def generate(self, request: ImageRequest):

        if request.randomize_seed:
            request.seed = random.randint(0, self.MAX_SEED)

        generator = torch.Generator().manual_seed(request.seed)

        try:
            if request.generate_prompt:
                generated_prompt = PromptGenerator.generate(
                    prompt=request.prompt,
                )
            else:
                generated_prompt = request.prompt

            image = self._pipe(
                prompt=generated_prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                generator=generator
            ).images[0]

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
