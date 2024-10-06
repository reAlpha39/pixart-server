from modules.generate_image_request_model import ImageRequest
from modules.prompt_generator import PromptGenerator

import torch
import random
import numpy as np
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

    def __init__(self):
        self.MAX_SEED = np.iinfo(np.int32).max
        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

    def flush(self):
        gc.collect()
        torch.cuda.empty_cache()

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

            text_encoder = T5EncoderModel.from_pretrained(
                f"./{request.model}",
                subfolder="text_encoder",
                quantization_config=self.quantization_config,
                device_map="auto",
                use_safetensors=True,
            )

            pipe = PixArtSigmaPipeline.from_pretrained(
                f"./{request.model}",
                torch_dtype=torch.float16,
                text_encoder=text_encoder,
                transformer=None,
                device_map="balanced",
                use_safetensors=True,
            )

            pipe.text_encoder.to_bettertransformer()

            with torch.no_grad():
                prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(
                    generated_prompt,
                    negative_prompt=request.negative_prompt,
                )

            del text_encoder
            del pipe
            self.flush()

            transformer = Transformer2DModel.from_pretrained(
                f"./{request.model}",
                subfolder='transformer',
                torch_dtype=torch.float16,
                use_safetensors=True,
            )

            pipe = PixArtSigmaPipeline.from_pretrained(
                f"./pixart_sigma_sdxlvae_T5_diffusers",
                text_encoder=None,
                transformer=transformer,
                torch_dtype=torch.float16,
                use_safetensors=True,
            ).to("cuda")

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
            del transformer
            self.flush()

            with torch.no_grad():
                image = pipe.vae.decode(
                    latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            image = pipe.image_processor.postprocess(
                image, output_type="pil")[0]

            del pipe
            self.flush()

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
