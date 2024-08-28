from modules.generate_image_request_model import ImageRequest
from modules.prompt_generator import PromptGenerator 

import torch
import random
import numpy as np
import ollama
from diffusers import PixArtSigmaPipeline
from transformers import T5EncoderModel, BitsAndBytesConfig
from PIL import Image
from io import BytesIO
import base64
from uuid import uuid4
import os
import gc

# Directory to save images
IMAGE_DIR = "./generated_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

class ImageGenerator:
    @classmethod
    def flush(self):
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def generate(self, request: ImageRequest):
        if request.randomize_seed:
            request.seed = random.randint(0, MAX_SEED)

        generator = torch.Generator().manual_seed(request.seed)

        try:
            if request.generate_prompt:
                generated_prompt = PromptGenerator.generate(
                    prompt=request.prompt,
                )
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
            self.flush()

            pipe = PixArtSigmaPipeline.from_pretrained(
                "./PixArt-Sigma-900M",
                text_encoder=None,
                torch_dtype=torch.float16,
            ).to("cuda")

            # pipe.transformer = torch.compile(
            #     pipe.transformer, mode="reduce-overhead",
            #     fullgraph=True,
            # )

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
            self.flush()

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
                self.flush()
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
            self.flush()

            return {
                "image": img_str,
                "generated_prompt":  generated_prompt,
                "seed": request.seed
            }

        except Exception as e:
            print(e)
            raise Exception(str(e))
